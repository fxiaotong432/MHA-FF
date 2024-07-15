from functools import partial
from typing import Callable, Union
from typing import Sequence
import pandas as pd
import SimpleITK as sitk
import numpy as np
import glob
import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers import get_pool_layer
from monai.networks.layers.factories import Conv, Norm
from torchinfo import summary
import radiomics
import logging
from radiomics import featureextractor
radiomics.logger.setLevel(logging.ERROR)
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import os
import time
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
import torch
from monai.losses import DiceCELoss
from monai.metrics import compute_meandice
from torch.nn import CrossEntropyLoss

from common import create_dir_if_not_exists
from radio_utils import *
from Models2D import resnet50, MultiModalMIL_2d,MultiModalMIL_2d_dimchange
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'




from sklearn.metrics import confusion_matrix,plot_roc_curve,auc,roc_auc_score,roc_curve
import pandas as pd
import numpy as np
from skimage import measure
from sklearn.metrics import accuracy_score,roc_curve,recall_score,roc_auc_score,auc,confusion_matrix,cohen_kappa_score, f1_score, precision_score,matthews_corrcoef 
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, SelectFromModel, SelectKBest
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats
from sklearn import svm, linear_model
from sklearn.svm import SVC,LinearSVC
from sklearn.feature_selection import SelectKBest, f_classif, chi2, SelectFromModel,mutual_info_classif
from statsmodels.discrete.discrete_model import MNLogit
from statistics import mean
from sklearn.preprocessing import label_binarize
import multiprocessing

def cal_multi_ft(Y_true, Y_score, n_classes=3):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_true[:, i], Y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_true.ravel(), Y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return fpr, tpr, roc_auc

def call_ml(datflag_file, data_file, model_stage, test_id, valid_id, n_sim, cv_num, feature_file, save_dir):
    views= ['16','32','48']
    train_path_1 = 'task2_Lung_Features.csv'
    train_path_2 = 'task2_Node_Features_16.csv'
    train_path_3 = 'task2_Node_Features_32.csv'
    train_path_4 = 'task2_Node_Features_48.csv'
    Feature_list_1 = pd.read_csv(train_path_1).fillna(0)
    Feature_list_1.columns =['lung_'+i if (i!='index') and (i!='flag') else i for i in Feature_list_1.columns]
    Feature_list_2 = pd.read_csv(train_path_2)
    Feature_list_2.columns =['node_'+i+'_'+views[0] if (i!='index') and (i!='flag') else i for i in Feature_list_2.columns]
    Feature_list_3 = pd.read_csv(train_path_3)
    Feature_list_3.columns =['node_'+i+'_'+views[1] if (i!='index') and (i!='flag') else i for i in Feature_list_3.columns]
    Feature_list_4 = pd.read_csv(train_path_4)
    Feature_list_4.columns =['node_'+i+'_'+views[2] if (i!='index') and (i!='flag') else i for i in Feature_list_4.columns]
    Feature_list = pd.merge(Feature_list_1, Feature_list_2, on=['index','flag'])
    Feature_list = pd.merge(Feature_list, Feature_list_3, on=['index','flag'])
    Feature_list = pd.merge(Feature_list, Feature_list_4, on=['index','flag'])
    Feature_list['index'] = Feature_list.pop('index')
    Feature_list['flag'] = Feature_list.pop('flag')
    train_index, test_index, valid_index = get_par_index_valid(datflag_file=datflag_file, data_file=data_file, stage = model_stage, 
                                                  test_id=test_id,valid_id=valid_id)
    print('train: ',len(train_index),' test:',len(test_index),' valid:',len(valid_index))

    Flag_ls = Feature_list['flag'].tolist()
    if model_stage=='task1':
        Lable_dict ={'良性':0,'其他恶性癌':1,'浸润前病变':1,'grade1':1,'grade2':1,'grade3':1}
        n_classes = 2
    elif model_stage=='task2':
        Lable_dict ={'浸润前病变':0,'其他恶性癌':1,'grade1':1,'grade2':1,'grade3':1}
        n_classes = 2
    elif model_stage=='task3':
        Lable_dict ={'grade1':0,'grade2':1,'grade3':2}
        n_classes = 3

    Lable_ls = np.array([Lable_dict[x] for x in Flag_ls ])
    Feature_Name = np.array(list(Feature_list.head(0))[:-2])
    train_idd = Feature_list.index[[i in train_index for i in Feature_list['index'] ]]
    test_idd = Feature_list.index[[i in test_index for i in Feature_list['index'] ]]
    valid_idd = Feature_list.index[[i in valid_index for i in Feature_list['index'] ]]

    Feature_train = Feature_list.iloc[train_idd,:-2]
    Lable_train = Lable_ls[train_idd]
    Feature_test = Feature_list.iloc[test_idd,:-2]
    Lable_test = Lable_ls[test_idd]
    Feature_valid = Feature_list.iloc[valid_idd,:-2]
    Lable_valid = Lable_ls[valid_idd]

    min_max_scaler = MinMaxScaler()
    Feature_train = pd.DataFrame(min_max_scaler.fit_transform(Feature_train))
    Feature_train.columns = Feature_Name
    Feature_test = pd.DataFrame(min_max_scaler.transform(Feature_test))
    Feature_test.columns = Feature_Name
    Feature_valid = pd.DataFrame(min_max_scaler.transform(Feature_valid))
    Feature_valid.columns = Feature_Name

    feat_dict = {'shape':[],'firstorder':[],'glcm':[],'glrlm':[],'glszm':[],
                 'gldm':[],'ngtdm':[]}
    for i in feat_dict:
        for j in  Feature_list.columns:
            if i in j:
                feat_dict[i].append(j)
    auc_group = {}
    nan_feature_name = []
    for i in feat_dict:
        print('current stage: ', i)
        auc_group[i]=[]
        for j in feat_dict[i]:
            x_slct = Feature_train[j]
            y_slct = Lable_train
            skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=0)
            crt_auc=[]
            for train_id_cv, test_id_cv in skf.split(x_slct, y_slct):
                X_train, X_test = x_slct[train_id_cv], x_slct[test_id_cv]
                y_train, y_test = y_slct[train_id_cv], y_slct[test_id_cv]
                y_test_b = label_binarize(y_test, classes=list(range(n_classes)))
                model_LR = MNLogit(y_train,X_train.values,missing='drop').fit(method='bfgs',maxiter = 1000)
                y_pred = model_LR.predict(X_test.tolist())
                if np.isnan(y_pred).any():
                    crt_auc.append(0)
                    nan_feature_name.append(j)
                else:
                    if n_classes==2:
                        cal_auc = roc_auc_score(y_test,y_pred[:, 1])
                        crt_auc.append(cal_auc)
                    else:
                        _,_,cal_auc = cal_multi_ft(y_test_b,y_pred, n_classes)
                        crt_auc.append(cal_auc['macro'])
            auc_group[i].append(mean(crt_auc))
    k = 10
    Feat_name_selected = []
    for i in feat_dict:
        Feat_name = [feat_dict[i][k] for k in np.argsort(np.array(auc_group[i]))[::-1][0:k]]
        Feat_name_selected.extend(Feat_name)
    Feature_train_selected = Feature_train.loc[:,Feat_name_selected]
    Feature_test_selected = Feature_test.loc[:,Feat_name_selected]
    Feature_valid_selected = Feature_valid.loc[:,Feat_name_selected]
    Feat_name_selected.append('index')
    Feature_list.loc[:,Feat_name_selected].to_csv(feature_file+'/selected_feature_'+str(n_sim)+'_'+str(cv_num)+'.csv',index=0)

    # SVM
    clf_OS = svm.SVC(kernel="rbf", probability=True, random_state=48)
    clf_OS.fit(Feature_train_selected, Lable_train)
    import joblib 
    joblib.dump(clf_OS, save_dir+"/SVM.m")
    pred_prob = clf_OS.predict_proba(Feature_valid_selected)
    Y_hat = []
    for j in range(len(Lable_valid)):
        Y_hat.append(list(pred_prob[j]).index(max(pred_prob[j])))
    Acc_clf_valid = accuracy_score(Lable_valid, Y_hat)
    print('Valid acc: ',Acc_clf_valid)
    clf_loss_function = CrossEntropyLoss()
    loss = clf_loss_function(torch.from_numpy(np.array(pred_prob)), torch.from_numpy(np.array(Lable_valid)))
    loss_valid = loss.item()
    pred_prob = clf_OS.predict_proba(Feature_test_selected)
    Y_hat = []
    for j in range(len(Lable_test)):
        Y_hat.append(list(pred_prob[j]).index(max(pred_prob[j])))
    Acc_clf_test =accuracy_score(Lable_test, Y_hat)
    print('Test acc: ',Acc_clf_test)
    clf_loss_function = CrossEntropyLoss()
    loss = clf_loss_function(torch.from_numpy(np.array(pred_prob)), torch.from_numpy(np.array(Lable_test)))
    # acc
    metric={}
    metric['acc'] = Acc_clf_test
    # auc
    metric['auc'] = sklearn.metrics.roc_auc_score(Lable_test, np.array(pred_prob)[:,1])
    auc_l, auc_h, z = cal_AUC_each(Lable_test, np.array(pred_prob)[:,1])
    metric['auc_l'] = auc_l
    metric['auc_h'] = auc_h
    metric['auc_std'] = z
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(Lable_test, Y_hat).ravel()
    metric['sensitivity'] = tp / (tp+fn)
    metric['specificity'] = tn / (tn+fp)
    metric['precision'] = tp / (tp+fp)
    f1 = 2*metric['precision']*metric['sensitivity']/(metric['precision']+metric['sensitivity'])
    metric['f1']=f1
    metric['loss'] = float(loss.item())
    
    return metric, pred_prob




def call_model(datflag_file, data_file, model_stage, test_id, valid_id, node_size, ct_num, mode, n_sim, cv_num, feature_file, 
                   hidden_dim, num_mil, save_dir):
    if model_stage=='task3':
        Class_num = 3
    else:
        Class_num = 2
    radiomics_file_adr = feature_file+'/selected_feature_'+str(n_sim)+'_'+str(cv_num)+'.csv'
    
    radio_dim = 70
    
    train_index, test_index, valid_index = get_par_index_valid(datflag_file=datflag_file, data_file=data_file, stage = model_stage, 
                                                               test_id=test_id, valid_id=valid_id)
    datflag_0 = pd.read_csv(datflag_file)
    train_id = datflag_0.loc[train_index,'ID'].unique()

    datflag_train=datflag_0.loc[train_index,:]
    datflag_test=datflag_0.loc[test_index,:]
    datflag_valid=datflag_0.loc[valid_index,:]
    
    data_file = '/course75/RealData/'
    partition, labels=get_par_lab_valid(datflag_file=datflag_file, data_file=data_file, stage = model_stage, 
                                        test_id=test_id,valid_id=valid_id, no_trans=no_trans)
    train_label = [labels[i] for i in partition['train']]
    train_anno = pd.DataFrame({'dir':partition['train'],'label':train_label,'index':partition['train_index']})
    test_label = [labels[i] for i in partition['test']]
    test_anno = pd.DataFrame({'dir':partition['test'],'label':test_label,'index':partition['test_index']})
    valid_label = [labels[i] for i in partition['valid']]
    valid_anno = pd.DataFrame({'dir':partition['valid'],'label':valid_label,'index':partition['valid_index']})
    radiomics_file = pd.read_csv(radiomics_file_adr)
    selected_feature_name =radiomics_file.drop(['index'], axis=1).columns
    
    # 2. DataLoader & Model
    training_data = CustomImageDataset_2d_nslice(train_anno,radiomics_file,data_file, input_size=node_size, n=ct_num)
    test_data = CustomImageDataset_2d_nslice(test_anno,radiomics_file, data_file, input_size=node_size, n=ct_num)
    valid_data = CustomImageDataset_2d_nslice(valid_anno,radiomics_file, data_file, input_size=node_size, n=ct_num)

    train_workers =12
    test_workers =12
    valid_workers =12
    
    train_dataloader = DataLoader(training_data, batch_size=96, shuffle=True, num_workers=train_workers)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=test_workers)
    valid_dataloader = DataLoader(valid_data, batch_size=64, shuffle=False, num_workers=valid_workers)
    # GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_configs = {
        'n_classes': 3,
        'n_input_channels': 1,
        'shortcut_type': 'B',
        'conv1_t_size': 7,
        'conv1_t_stride': 1,
        'no_max_pool': False,
        'widen_factor': 1.0,
    }
    if mode =='mil':
        # 128：23,785,580
        # 64： 23,645,788
        model = MultiModalMIL_2d_dimchange(num_classes=Class_num, num_mil=num_mil, pretrained=False, hidden_dim=hidden_dim, 
                                           att_type='normal', with_drop=None, dim_change = False, 
                                           radio_dim = radio_dim,n_slice=1,**model_configs).to(device)
        model_name = 'MIL_merge'
    elif mode =='res':
        # 23,514,179
        model = resnet50(pretrained=False, num_classes=Class_num, wo_class=False).to(device)
        model_name = 'Resnet'
    print('model_name: ',model_name)
    learning_rate = 0.001
    train_epochs = 20
    val_interval = 1

    iters_verbose = 20

    clf_loss_function = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epochs)
    best_acc = 0
    best_metric_epoch = 0
    best_metric_test = 0
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    test_loss = []
    test_acc = []
    for epoch in range(train_epochs):
        crt_loss, crt_acc= train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, clf_loss_function, device,
                        epoch + 1, iters_verbose)
        if epoch % val_interval == 0:
            # valid
            clf_metrc, clf_scores = evaluate(valid_dataloader, model, clf_loss_function,device)
            clf_loss = clf_metrc['loss']
            clf_acc = clf_metrc['acc']

            # test
            clf_metric_test, clf_scores_test = evaluate(test_dataloader, model, clf_loss_function,device)
            clf_loss_test = clf_metric_test['loss']
            clf_acc_test = clf_metric_test['acc']

            print(f" Valid Clf Loss: {clf_loss}, Valid Clf Acc: {clf_acc}")
            print(f" Test Clf Loss: {clf_loss_test}, Test Clf Acc: {clf_acc_test}")
            if clf_acc > best_acc:
                best_metric_valid = clf_metrc
                best_acc = best_metric_valid['acc']
                best_metric_loss = clf_loss
                best_metric_epoch = epoch + 1
                best_metric_test = clf_metric_test
                best_score_test = clf_scores_test
                print(f"\nCurrent Test mean acc: {clf_acc_test:.4f} ")
                PATH = save_dir+'/'+model_name+'_'+str(num_mil)+'dim_'+str(hidden_dim)+'_model.pt'
                torch.save(model, PATH)
            print(f"current epoch: {epoch + 1}"
                  f"\nValid best acc: {best_metric_valid['acc']:.4f}  Test acc: {best_metric_test['acc']:.4f} at epoch: {best_metric_epoch}")
            
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return best_metric_test,best_score_test


node_size = 32
ct_num = 7
select_radio = True
no_trans = True
if_radio = True
if_mil = False
if_deep = False


model_stage ='task2'
feature_file = model_stage+'_features'
datflag_file = 'datflag641-plus.csv'
data_file = '/course75/RealData/CT/'

if model_stage=='task3':
    Class_num = 3
else:
    Class_num = 2



dat_csv = pd.read_csv(datflag_file)


if model_stage=='task1':
        datflag_crt = pd.DataFrame(dat_csv)
        datflag_crt.loc[:, 'label'] = 1
        datflag_crt.loc[datflag_crt['flag']=='良性','label'] = 0
elif model_stage=='task2':
    datflag_crt = pd.DataFrame(dat_csv.loc[dat_csv['flag']!='良性',])
    datflag_crt.loc[:, 'label'] = 1
    datflag_crt.loc[datflag_crt['flag']=='浸润前病变','label'] = 0
elif model_stage=='task3':
    datflag_crt = pd.DataFrame(dat_csv.loc[dat_csv.apply(lambda x: x['flag'] in ['grade1','grade2','grade3'], axis=1),])
    datflag_crt.loc[:, 'label'] = 0
    datflag_crt.loc[datflag_crt['flag']=='grade2','label'] = 1
    datflag_crt.loc[datflag_crt['flag']=='grade3','label'] = 2
print('patients:',len(datflag_crt['ID'].unique()),'  nodes: ',len(datflag_crt))

patient_ID = datflag_crt['ID'].drop_duplicates()
patient_Lable = datflag_crt.loc[patient_ID.index,'label']
patient_ID = np.array(patient_ID)
patient_Lable = np.array(patient_Lable)
Sim_times= 10
hidden_dim = 128
num_mil = 4

outcome_radio_list = []
outcome_radio_pred = []
outcome_mil_list = []
outcome_mil_pred =[]
outcome_res_list = []
outcome_res_pred =[]
save_dir = 'Task2_final_model/'
save_dir = create_dir_if_not_exists(save_dir)
start = time.perf_counter()
n_sim=1
split_1 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=n_sim)
split_2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=n_sim)
for cv_num, [train_id_index, test_val_id_index] in enumerate(split_1.split(patient_ID, patient_Lable)):
    for valid_id_index, test_id_index in split_2.split(patient_ID[test_val_id_index], patient_Lable[test_val_id_index]):
        test_id = patient_ID[test_val_id_index][test_id_index]
        valid_id = patient_ID[test_val_id_index][valid_id_index]
        if if_radio:
            metric_radio,score_radio = call_ml(datflag_file=datflag_file, data_file=data_file, model_stage=model_stage, 
                                  test_id=test_id, valid_id=valid_id, n_sim=n_sim, cv_num=cv_num, feature_file=feature_file,save_dir=save_dir)
            outcome_radio_list.append(metric_radio)
            outcome_radio_pred.append(score_radio)
            pd.DataFrame(outcome_radio_list).to_csv(os.path.join(save_dir, 'radio.csv'),index=0)
            pd.DataFrame(outcome_radio_pred[0]).T.to_csv(os.path.join(save_dir, 'radio_pred.csv'),index=0)

        if if_mil:
            metric_mil,score_mil = call_model(datflag_file=datflag_file, data_file=data_file, model_stage=model_stage, 
                                            test_id=test_id, valid_id=valid_id,node_size=node_size, ct_num=ct_num,
                                            mode = 'mil', n_sim=n_sim, cv_num=cv_num, feature_file=feature_file, 
                                            hidden_dim=hidden_dim, num_mil=num_mil,save_dir=save_dir)
            outcome_mil_list.append(metric_mil)
            outcome_mil_pred.append(score_mil)
            pd.DataFrame(outcome_mil_pred).T.to_csv(os.path.join(save_dir, 'ep10_mil_'+str(num_mil)+'dim_'+str(hidden_dim)+'ct_'+str(ct_num)+'_pred.csv'),index=0)
            pd.DataFrame(outcome_mil_list).to_csv(os.path.join(save_dir, 'ep10_mil_'+str(num_mil)+'dim_'+str(hidden_dim)+'ct_'+str(ct_num)+'_metric.csv'),index=0)

        if if_deep:
            metric_res,score_res = cv_train_valid(datflag_file=datflag_file, data_file=data_file, model_stage=model_stage, 
                                            test_id=test_id, valid_id=valid_id,node_size=node_size, ct_num=ct_num,
                                            mode = 'res', n_sim=n_sim,cv_num=cv_num, feature_file=feature_file, 
                                            hidden_dim=hidden_dim, num_mil=num_mil,save_dir=save_dir)
            outcome_res_list.append(metric_res)
            outcome_res_pred.append(score_res)
            pd.DataFrame(outcome_res_pred).T.to_csv(os.path.join(save_dir,'res_'+str(ct_num)+'_pred.csv'),index=0)
            pd.DataFrame(outcome_res_list).to_csv(os.path.join(save_dir, 'res_'+str(ct_num)+'.csv'),index=0)

end = time.perf_counter()
print('total time:',(end-start)/60)  









