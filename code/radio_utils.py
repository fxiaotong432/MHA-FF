from functools import partial
from sklearn.metrics import accuracy_score,roc_curve,recall_score,roc_auc_score
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
import time
import radiomics
import logging
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from scipy.ndimage.interpolation import zoom
import sklearn
from scipy.stats import sem, t
def get_par_index(datflag_file, data_file, stage, train_id=False, test_id=[], Null_id=[]):
    datflag_0 = pd.read_csv(datflag_file)
    datflag_0['subset'] = None
    datflag_0.loc[datflag_0.apply(lambda x: x['ID'] in test_id, axis=1),'subset'] = 'test'
    if train_id:
        datflag_0.loc[datflag_0.apply(lambda x: x['ID'] in train_id, axis=1),'subset'] = 'train'
    else:
        datflag_0.loc[datflag_0['subset'].isnull(),'subset'] = 'train'
    datflag_0.loc[datflag_0.apply(lambda x: x['ID'] in Null_id, axis=1),'subset'] = None
    if stage=='task1':
        datflag = pd.DataFrame(datflag_0)
        datflag.loc[:, 'label'] = 1
        datflag.loc[datflag['flag']=='良性','label'] = 0
    elif stage=='task2':
        datflag = pd.DataFrame(datflag_0.loc[datflag_0['flag']!='良性',])
        datflag.loc[:, 'label'] = 1
        datflag.loc[datflag['flag']=='浸润前病变','label'] = 0
    elif stage=='task3':
        datflag = pd.DataFrame(datflag_0.loc[datflag_0.apply(lambda x: x['flag'] in ['grade1','grade2','grade3'], axis=1),])
        # grade1 = 0，grade2 = 1, grade3 = 2
        datflag.loc[:, 'label'] = 0
        datflag.loc[datflag['flag']=='grade2','label'] = 1
        datflag.loc[datflag['flag']=='grade3','label'] = 2
    datflag_train = datflag.loc[datflag['subset']=='train',]
    train_index = list(datflag_train.index)
    #train_flag = list(datflag_train['label'])
    datflag_test = datflag.loc[datflag['subset']=='test',]
    test_index = list(datflag_test.index)
    return train_index, test_index

def get_par_lab(datflag_file, data_file, stage, train_id=False, test_id=[], Null_id=[], no_trans=False):
    datflag_0 = pd.read_csv(datflag_file)
    datflag_0['subset'] = None
    datflag_0.loc[datflag_0.apply(lambda x: x['ID'] in test_id, axis=1),'subset'] = 'test'
    if train_id:
        datflag_0.loc[datflag_0.apply(lambda x: x['ID'] in train_id, axis=1),'subset'] = 'train'
    else:
        datflag_0.loc[datflag_0['subset'].isnull(),'subset'] = 'train'
    datflag_0.loc[datflag_0.apply(lambda x: x['ID'] in Null_id, axis=1),'subset'] = None
    
    if stage=='task1':
        datflag = pd.DataFrame(datflag_0)
        datflag.loc[:, 'label'] = 1
        datflag.loc[datflag['flag']=='良性','label'] = 0
    elif stage=='task2':
        datflag = pd.DataFrame(datflag_0.loc[datflag_0['flag']!='良性',])
        datflag.loc[:, 'label'] = 1
        datflag.loc[datflag['flag']=='浸润前病变','label'] = 0
    elif stage=='task3':
        datflag = pd.DataFrame(datflag_0.loc[datflag_0.apply(lambda x: x['flag'] in ['grade1','grade2','grade3'], axis=1),])
        # grade1 = 0，grade2 = 1, grade3 = 2
        datflag.loc[:, 'label'] = 0
        datflag.loc[datflag['flag']=='grade2','label'] = 1
        datflag.loc[datflag['flag']=='grade3','label'] = 2
    datflag_train = datflag.loc[datflag['subset']=='train',]
    train_index = list(datflag_train.index)
    #train_flag = list(datflag_train['label'])
    datflag_test = datflag.loc[datflag['subset']=='test',]
    test_index = list(datflag_test.index)
    file_index_train = []
    file_index_test = []
    train_file_list=[]
    train_flag_list=[]
    agm_flist=glob.glob(data_file+'data/agm/*')  
    for f_name in agm_flist:
        c_name = f_name.split('/')[-1]
        c_index = int(c_name.split('_')[0])
        if no_trans:
            if c_index in train_index:
                if 'trans' in c_name:
                    if 'trans_0' in c_name:
                        train_file_list.append('agm/'+c_name)
                        train_flag_list.append(datflag_train.loc[c_index,'label'])
                        file_index_train.append(c_index)
                else:
                    train_file_list.append('agm/'+c_name)
                    train_flag_list.append(datflag_train.loc[c_index,'label'])
                    file_index_train.append(c_index)
        else:
            if c_index in train_index:
                train_file_list.append('agm/'+c_name)
                train_flag_list.append(datflag_train.loc[c_index,'label'])
                file_index_train.append(c_index)
    test_file_list=[]
    test_flag_list=[]
    org_flist=glob.glob(data_file+'data/org/*')  
    for f_name in org_flist:
        c_name = f_name.split('/')[-1]
        c_index = int(c_name.split('.')[0])
        if c_index in test_index:
            test_file_list.append('org/'+c_name)
            test_flag_list.append(datflag_test.loc[c_index,'label'])
            file_index_test.append(c_index)
    partition = {'train': train_file_list, 'test': test_file_list, 'train_index':file_index_train, 'test_index':file_index_test}
    labels = dict(zip((train_file_list+test_file_list),(train_flag_list+test_flag_list)))
    return partition, labels


def get_par_lab_valid(datflag_file, data_file, stage, train_id=False, test_id=[], valid_id=[], no_trans=False):
    datflag_0 = pd.read_csv(datflag_file)
    datflag_0['subset'] = None
    datflag_0.loc[datflag_0.apply(lambda x: x['ID'] in test_id, axis=1),'subset'] = 'test'
    datflag_0.loc[datflag_0.apply(lambda x: x['ID'] in valid_id, axis=1),'subset'] = 'valid'
    if train_id:
        datflag_0.loc[datflag_0.apply(lambda x: x['ID'] in train_id, axis=1),'subset'] = 'train'
    else:
        datflag_0.loc[datflag_0['subset'].isnull(),'subset'] = 'train'
    
    if stage=='task1':
        datflag = pd.DataFrame(datflag_0)
        datflag.loc[:, 'label'] = 1
        datflag.loc[datflag['flag']=='良性','label'] = 0
    elif stage=='task2':
        datflag = pd.DataFrame(datflag_0.loc[datflag_0['flag']!='良性',])
        datflag.loc[:, 'label'] = 1
        datflag.loc[datflag['flag']=='浸润前病变','label'] = 0
    elif stage=='task3':
        datflag = pd.DataFrame(datflag_0.loc[datflag_0.apply(lambda x: x['flag'] in ['grade1','grade2','grade3'], axis=1),])
        # grade1 = 0，grade2 = 1, grade3 = 2
        datflag.loc[:, 'label'] = 0
        datflag.loc[datflag['flag']=='grade2','label'] = 1
        datflag.loc[datflag['flag']=='grade3','label'] = 2
    datflag_train = datflag.loc[datflag['subset']=='train',]
    train_index = list(datflag_train.index)

    datflag_test = datflag.loc[datflag['subset']=='test',]
    test_index = list(datflag_test.index)
    
    datflag_valid = datflag.loc[datflag['subset']=='valid',]
    valid_index = list(datflag_valid.index)
    file_index_train = []
    file_index_test = []
    file_index_valid =[]
    train_file_list=[]
    train_flag_list=[]
    agm_flist=glob.glob(data_file+'data/agm/*')  
    for f_name in agm_flist:
        c_name = f_name.split('/')[-1]
        c_index = int(c_name.split('_')[0])
        if no_trans:
            if c_index in train_index:
                if 'trans' in c_name:
                    if 'trans_0' in c_name:
                        train_file_list.append('agm/'+c_name)
                        train_flag_list.append(datflag_train.loc[c_index,'label'])
                        file_index_train.append(c_index)
                else:
                    train_file_list.append('agm/'+c_name)
                    train_flag_list.append(datflag_train.loc[c_index,'label'])
                    file_index_train.append(c_index)
        else:
            if c_index in train_index:
                train_file_list.append('agm/'+c_name)
                train_flag_list.append(datflag_train.loc[c_index,'label'])
                file_index_train.append(c_index)
    test_file_list=[]
    test_flag_list=[]
    org_flist=glob.glob(data_file+'data/org/*')  
    for f_name in org_flist:
        c_name = f_name.split('/')[-1]
        c_index = int(c_name.split('.')[0])
        if c_index in test_index:
            test_file_list.append('org/'+c_name)
            test_flag_list.append(datflag_test.loc[c_index,'label'])
            file_index_test.append(c_index)
    valid_file_list=[]
    valid_flag_list=[]
    org_flist=glob.glob(data_file+'data/org/*')  
    for f_name in org_flist:
        c_name = f_name.split('/')[-1]
        c_index = int(c_name.split('.')[0])
        if c_index in valid_index:
            valid_file_list.append('org/'+c_name)
            valid_flag_list.append(datflag_valid.loc[c_index,'label'])
            file_index_valid.append(c_index)
    
    partition = {'train': train_file_list, 'test': test_file_list,'valid':valid_file_list,
                 'train_index':file_index_train, 'test_index':file_index_test,'valid_index':file_index_valid}
    labels = dict(zip((train_file_list+test_file_list+valid_file_list),(train_flag_list+test_flag_list+valid_flag_list)))
    return partition, labels


def get_par_index_valid(datflag_file, data_file, stage, train_id=False, test_id=[], valid_id=[]):
    datflag_0 = pd.read_csv(datflag_file)
    datflag_0['subset'] = None
    datflag_0.loc[datflag_0.apply(lambda x: x['ID'] in test_id, axis=1),'subset'] = 'test'
    datflag_0.loc[datflag_0.apply(lambda x: x['ID'] in valid_id, axis=1),'subset'] = 'valid'
    if train_id:
        datflag_0.loc[datflag_0.apply(lambda x: x['ID'] in train_id, axis=1),'subset'] = 'train'
    else:
        datflag_0.loc[datflag_0['subset'].isnull(),'subset'] = 'train'

    if stage=='task1':
        datflag = pd.DataFrame(datflag_0)
        datflag.loc[:, 'label'] = 1
        datflag.loc[datflag['flag']=='良性','label'] = 0
    elif stage=='task2':
        datflag = pd.DataFrame(datflag_0.loc[datflag_0['flag']!='良性',])
        datflag.loc[:, 'label'] = 1
        datflag.loc[datflag['flag']=='浸润前病变','label'] = 0
    elif stage=='task3':
        datflag = pd.DataFrame(datflag_0.loc[datflag_0.apply(lambda x: x['flag'] in ['grade1','grade2','grade3'], axis=1),])
        # grade1 = 0，grade2 = 1, grade3 = 2
        datflag.loc[:, 'label'] = 0
        datflag.loc[datflag['flag']=='grade2','label'] = 1
        datflag.loc[datflag['flag']=='grade3','label'] = 2
    datflag_train = datflag.loc[datflag['subset']=='train',]
    train_index = list(datflag_train.index)

    datflag_test = datflag.loc[datflag['subset']=='test',]
    test_index = list(datflag_test.index)
    
    datflag_valid = datflag.loc[datflag['subset']=='valid',]
    valid_index = list(datflag_valid.index)
    return train_index, test_index, valid_index


import torch
from torch import as_tensor
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, radiomics_file, img_dir, input_size=32):
        self.img_labels = annotations_file
        self.radiomics = radiomics_file
        self.img_dir = img_dir
        self.input_size = input_size

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,'data', self.img_labels.iloc[idx, 0])
        input_size =self.input_size
        image = np.load(img_path, mmap_mode='r' )
        volume = image.shape[0]
        image = image[round(volume/2-input_size/2):round(volume/2+input_size/2),
                              round(volume/2-input_size/2):round(volume/2+input_size/2),
                              round(volume/2-input_size/2):round(volume/2+input_size/2)]
        label = self.img_labels.iloc[idx, 1]
        node_index =  self.img_labels.iloc[idx, 2]
        radio_features = np.array(self.radiomics.loc[self.radiomics['index']==node_index,:].drop(['index'], axis=1))
        image = torch.tensor(image).unsqueeze(0)
        label = torch.tensor(label)
        radio_features = torch.tensor(radio_features)
            
        return image, radio_features, label, node_index

class CustomImageDataset_2d(Dataset):
    def __init__(self, annotations_file, radiomics_file, img_dir, input_size=32):
        self.img_labels = annotations_file
        self.radiomics = radiomics_file
        self.img_dir = img_dir
        self.input_size = input_size

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,'data', self.img_labels.iloc[idx, 0])
        input_size =self.input_size
        image = np.load(img_path, mmap_mode='r' )
        volume = image.shape[0]
        image = image[round(volume/2-2):round(volume/2+1),
                              round(volume/2-input_size/2):round(volume/2+input_size/2),
                              round(volume/2-input_size/2):round(volume/2+input_size/2)]
        label = self.img_labels.iloc[idx, 1]
        node_index =  self.img_labels.iloc[idx, 2]
        radio_features = np.array(self.radiomics.loc[self.radiomics['index']==node_index,:].drop(['index'], axis=1))
        image = torch.tensor(image)
        label = torch.tensor(label)
        radio_features = torch.tensor(radio_features)
           
            
        return image, radio_features, label, node_index
    

class CustomImageDataset_2d_1slice(Dataset):
    def __init__(self, annotations_file, radiomics_file, img_dir, input_size=32):
        self.img_labels = annotations_file
        self.radiomics = radiomics_file
        self.img_dir = img_dir
        self.input_size = input_size

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,'data', self.img_labels.iloc[idx, 0])
        input_size =self.input_size
        image = np.load(img_path, mmap_mode='r' )
        volume = image.shape[0]
        image = image[round(volume/2-1):round(volume/2),
                              round(volume/2-input_size/2):round(volume/2+input_size/2),
                              round(volume/2-input_size/2):round(volume/2+input_size/2)]
        label = self.img_labels.iloc[idx, 1]
        node_index =  self.img_labels.iloc[idx, 2]
        radio_features = np.array(self.radiomics.loc[self.radiomics['index']==node_index,:].drop(['index'], axis=1))
        image = torch.tensor(image)
        label = torch.tensor(label)
        radio_features = torch.tensor(radio_features)
           
            
        return image, radio_features, label, node_index
    
class CustomImageDataset_2d_nslice(Dataset):
    def __init__(self, annotations_file, radiomics_file, img_dir, input_size=32, n=3):
        self.img_labels = annotations_file
        self.radiomics = radiomics_file
        self.img_dir = img_dir
        self.input_size = input_size
        self.n = n
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,'data', self.img_labels.iloc[idx, 0])
        input_size =self.input_size
        image = np.load(img_path, mmap_mode='r' )
        volume = image.shape[0]
        n_range = int(self.n/2)
        img_list = []
        for i in range(round(volume/2-n_range-1),round(volume/2+n_range)):
            temp = image[i,
                       round(volume/2-input_size/2):round(volume/2+input_size/2),
                       round(volume/2-input_size/2):round(volume/2+input_size/2)]
            img_list.append(temp)
        image = np.stack(img_list,axis=0)
        label = self.img_labels.iloc[idx, 1]
        node_index =  self.img_labels.iloc[idx, 2]
        radio_features = np.array(self.radiomics.loc[self.radiomics['index']==node_index,:].drop(['index'], axis=1))
        image = torch.tensor(image).unsqueeze(-3)
        label = torch.tensor(label)
        radio_features = torch.tensor(radio_features)
            
        return image, radio_features, label, node_index


class CustomImageDataset_2d_nchannel(Dataset):
    def __init__(self, annotations_file, radiomics_file, img_dir, input_size=32, n=3):
        self.img_labels = annotations_file
        self.radiomics = radiomics_file
        self.img_dir = img_dir
        self.input_size = input_size
        self.n = n
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,'data', self.img_labels.iloc[idx, 0])
        input_size =self.input_size
        image = np.load(img_path, mmap_mode='r' )
        volume = image.shape[0]
        n_range = int(self.n/2)
        img_list = []
        for i in range(round(volume/2-n_range-1),round(volume/2+n_range)):
            temp = image[i,
                       round(volume/2-input_size/2):round(volume/2+input_size/2),
                       round(volume/2-input_size/2):round(volume/2+input_size/2)]
            img_list.append(temp)
        image = np.stack(img_list,axis=0)
        label = self.img_labels.iloc[idx, 1]
        node_index =  self.img_labels.iloc[idx, 2]
        radio_features = np.array(self.radiomics.loc[self.radiomics['index']==node_index,:].drop(['index'], axis=1))
        image = torch.tensor(image)
        label = torch.tensor(label)
        radio_features = torch.tensor(radio_features)
        
            
        return image, radio_features, label, node_index


class CustomImageDataset_2d_nchannel_resize(Dataset):
    def __init__(self, annotations_file, radiomics_file, img_dir, input_size=32, n=3):
        self.img_labels = annotations_file
        self.radiomics = radiomics_file
        self.img_dir = img_dir
        self.input_size = input_size
        self.n = n
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,'data', self.img_labels.iloc[idx, 0])
        input_size =self.input_size
        image = np.load(img_path, mmap_mode='r' )
        volume = image.shape[0]
        n_range = int(self.n/2)
        img_list = []
        for i in range(round(volume/2-n_range-1),round(volume/2+n_range)):
            temp = image[i,
                       round(volume/2-input_size/2):round(volume/2+input_size/2),
                       round(volume/2-input_size/2):round(volume/2+input_size/2)]
            # Calculate the zoom factor
            # Resize the image
            temp = zoom(temp, 7)
            img_list.append(temp)
            
        image = np.stack(img_list,axis=0)
        label = self.img_labels.iloc[idx, 1]
        node_index =  self.img_labels.iloc[idx, 2]
        radio_features = np.array(self.radiomics.loc[self.radiomics['index']==node_index,:].drop(['index'], axis=1))
        image = torch.tensor(image)
        label = torch.tensor(label)
        radio_features = torch.tensor(radio_features)
        
            
        return image, radio_features, label, node_index


def train_one_epoch(train_loader, model, optimizer, lr_scheduler, loss_function, device, epoch, iters_verbose=10):
    with torch.autograd.set_detect_anomaly(True):
        model.train()
        epoch_loss = 0.0
        step = 0
        num_correct = 0.0
        metric_count = 0
        for train_features, radio_features, train_labels, node_index in train_loader:
            train_labels= train_labels.to(device)
            # train_features = train_features.unsqueeze(1)
            train_features = train_features.to(device, dtype=torch.float)
            radio_features = radio_features.to(device, dtype=torch.float)
            step_start = time.time()
            step += 1
            optimizer.zero_grad()
            clf = model(train_features, radio_features)
            loss = loss_function(clf, train_labels)
            loss.backward()
            clip_gradient(model, 1)
            optimizer.step()
            epoch_loss += loss.item()
            value = torch.eq(torch.argmax(clf, dim=1), train_labels)
            metric_count += len(value)
            num_correct += value.sum().item()
            if step % iters_verbose == 0:
                print(f"epoch: {epoch}, step:{step}, train_loss: {loss.item():.4f}, "
                      f"lr: {lr_scheduler.get_last_lr()[0]:.6f}, "
                      f"step time: {(time.time() - step_start):.4f}")
        metric = num_correct / metric_count
        lr_scheduler.step()
        epoch_loss /= step
        print(f"epoch {epoch} average loss: {epoch_loss:.4f}, acc: {metric:.4f}")
        return epoch_loss, metric

def cal_AUC_each(Y_true, Y_score):
    
    n_bootstraps = 1000
    rng_seed = 72  # control reproducibility
    bootstrapped_scores = []
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(Y_true), len(Y_true))
        if len(np.unique(np.array(Y_true)[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        bstr_score = roc_auc_score(np.array(Y_true)[indices], np.array(Y_score)[indices])
        bootstrapped_scores.append(bstr_score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_std = sorted_scores.std()
    confidence_lower, confidence_upper = np.percentile(sorted_scores, (2.5, 97.5))
    z = (roc_auc_score(Y_true, Y_score)-0.5)/confidence_std
    return confidence_lower, confidence_upper, confidence_std

def evaluate(test_loader, model, loss_function, device):
    model.eval()
    num_correct = 0.0
    metric_count = 0
    epoch_loss = 0.0
    step = 0
    metric = {}
    true_labels = []
    pred_scores = []
    pred_labels = []
    with torch.no_grad():
        for test_features, radio_features, test_labels, node_index in test_loader:
            test_labels= test_labels.to(device)
            true_labels.extend(test_labels.to('cpu').tolist())
            test_features = test_features.to(device, dtype=torch.float)
            radio_features = radio_features.to(device, dtype=torch.float)
            
            clf = model(test_features,radio_features)
            pred_scores.extend(clf.to('cpu').tolist())
            loss = loss_function(clf, test_labels)
            epoch_loss += loss.item()
            step += 1
            pred_labels.extend(torch.argmax(clf, dim=1).to('cpu').tolist())
        epoch_loss /= step
        
        # acc
        metric['acc'] = sklearn.metrics.accuracy_score(true_labels, pred_labels)
        # auc
        metric['auc'] = sklearn.metrics.roc_auc_score(true_labels, np.array(pred_scores)[:,1])
        auc_l, auc_h, z = cal_AUC_each(true_labels, np.array(pred_scores)[:,1])
        metric['auc_l'] = auc_l
        metric['auc_h'] = auc_h
        metric['auc_std'] = z
        
        # 其他指标
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(true_labels, pred_labels).ravel()
        metric['sensitivity'] = tp / (tp+fn)
        metric['specificity'] = tn / (tn+fp)
        metric['precision'] = tp / (tp+fp)
        f1 = 2*metric['precision']*metric['sensitivity']/(metric['precision']+metric['sensitivity'])
        metric['f1']=f1
        metric['loss'] = float(epoch_loss)
        return metric, pred_scores
    
def evaluate_binary(test_loader, model, loss_function, device):
    pred_scores = []
    model.eval()
    num_correct = 0.0
    metric_count = 0
    epoch_loss = 0.0
    step = 0
    with torch.no_grad():
        for test_features, radio_features, test_labels, node_index in test_loader:
            test_labels= test_labels.to(device)
            # test_features = test_features.unsqueeze(1)
            test_features = test_features.to(device, dtype=torch.float)
            radio_features = radio_features.to(device, dtype=torch.float)
            
            clf = model(test_features,radio_features)
            
            pred_scores.extend(clf.to('cpu').tolist())
            loss = loss_function(clf, test_labels)
            epoch_loss += loss.item()
            step += 1
            value = torch.eq(torch.argmax(clf, dim=1), test_labels)
            metric_count += len(value)
            num_correct += value.sum().item()
        metric = num_correct / metric_count
        epoch_loss /= step
        return float(epoch_loss), float(metric),pred_scores
def evaluate_binary_t(test_loader, model, loss_function, device):
    pred_scores = []
    true_labels = []
    model.eval()
    num_correct = 0.0
    metric_count = 0
    epoch_loss = 0.0
    step = 0
    with torch.no_grad():
        for test_features, radio_features, test_labels, node_index in test_loader:
            
            test_labels= test_labels.to(device)
            # test_features = test_features.unsqueeze(1)
            test_features = test_features.to(device, dtype=torch.float)
            radio_features = radio_features.to(device, dtype=torch.float)
            
            clf = model(test_features,radio_features)
            
            pred_scores.extend(clf.to('cpu').tolist())
            true_labels.extend(test_labels.to('cpu').tolist())
            loss = loss_function(clf, test_labels)
            epoch_loss += loss.item()
            step += 1
            value = torch.eq(torch.argmax(clf, dim=1), test_labels)
            metric_count += len(value)
            num_correct += value.sum().item()
        metric = num_correct / metric_count
        epoch_loss /= step
        return float(epoch_loss), float(metric),pred_scores,true_labels


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)  
    if torch.cuda.is_available():  
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.deterministic = True 