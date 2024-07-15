import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import cv2
import os
import SimpleITK as sitk
from PIL import Image
import imageio
from scipy.ndimage import zoom
from scipy import ndimage as ndi
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import disk, dilation, binary_erosion, binary_closing
from skimage.filters import roberts, sobel
import cv2
from matplotlib.image import imsave
import copy
from radiomics import featureextractor,imageoperations
import time
from time import sleep
import logging
# set level for all classes
logger = logging.getLogger("radiomics")
logger.setLevel(logging.ERROR)
def get_par_index(datflag_file, data_file, stage, train_id=False, test_id=[], Null_id=[]):
    print('Current task stage: '+str(stage))
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

def get_segmented_lungs(im):
    
    binary = im < -400                      
    cleared = clear_border(binary)          
    label_image = label(cleared)            
    
    areas = [r.area for r in regionprops(label_image)]  
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    
    selem = disk(2)                         
    binary = binary_erosion(binary, selem)
    selem = disk(10)                        
    binary = binary_closing(binary, selem)
    edges = roberts(binary)                 
    binary = ndi.binary_fill_holes(edges)
    binary = binary.astype(int)
    #get_high_vals = binary == 0             
    #im[get_high_vals] = 0
    #print('lung segmentation complete.')
    return im, binary
def get_Nodemask(im,x,y,diam,spacing, c_diam = 18):

    binary=np.zeros(im.shape) # 512*512
    #c_diam = (nn+1)*10
    im_range = im.shape[0]
    v_xmin = np.max([0,int(x-c_diam)])
    v_xmax = np.min([im_range-1,int(x+c_diam)])
    v_ymin = np.max([0,int(y-c_diam)]) 
    v_ymax = np.min([im_range-1,int(y+c_diam)])

    binary[v_ymin:v_ymax,v_xmin:v_xmax]=1
    return im, binary

def readDCM_Img(FilePath):
    reader = sitk.ImageSeriesReader()
    dcm_names = reader.GetGDCMSeriesFileNames(FilePath)
    reader.SetFileNames(dcm_names)
    image = reader.Execute()
    return image

def Extract_Features(image,mask,params_path):
    paramsFile = os.path.abspath(params_path)
    extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)
    result = extractor.execute(image, mask)
    general_info = {'diagnostics_Configuration_EnabledImageTypes','diagnostics_Configuration_Settings',
                    'diagnostics_Image-interpolated_Maximum','diagnostics_Image-interpolated_Mean',
                    'diagnostics_Image-interpolated_Minimum','diagnostics_Image-interpolated_Size',
                    'diagnostics_Image-interpolated_Spacing','diagnostics_Image-original_Hash',
                    'diagnostics_Image-original_Maximum','diagnostics_Image-original_Mean',
                    'diagnostics_Image-original_Minimum','diagnostics_Image-original_Size',
                    'diagnostics_Image-original_Spacing','diagnostics_Mask-interpolated_BoundingBox',
                    'diagnostics_Mask-interpolated_CenterOfMass','diagnostics_Mask-interpolated_CenterOfMassIndex',
                    'diagnostics_Mask-interpolated_Maximum','diagnostics_Mask-interpolated_Mean',
                    'diagnostics_Mask-interpolated_Minimum','diagnostics_Mask-interpolated_Size',
                    'diagnostics_Mask-interpolated_Spacing','diagnostics_Mask-interpolated_VolumeNum',
                    'diagnostics_Mask-interpolated_VoxelNum','diagnostics_Mask-original_BoundingBox',
                    'diagnostics_Mask-original_CenterOfMass','diagnostics_Mask-original_CenterOfMassIndex',
                    'diagnostics_Mask-original_Hash','diagnostics_Mask-original_Size',
                    'diagnostics_Mask-original_Spacing','diagnostics_Mask-original_VolumeNum',
                    'diagnostics_Mask-original_VoxelNum','diagnostics_Versions_Numpy',
                    'diagnostics_Versions_PyRadiomics','diagnostics_Versions_PyWavelet',
                    'diagnostics_Versions_Python','diagnostics_Versions_SimpleITK',
                    'diagnostics_Image-original_Dimensionality'}
    features = dict((key, value) for key, value in result.items() if key not in general_info)
    feature_info = dict((key, value) for key, value in result.items() if key in general_info)
    return features,feature_info


def resample(imgs, spacing, new_spacing=[1,1,1]):  
    new_shape = []  
    for i in range(3): 
        new_zyx = np.round(imgs.shape[i]*spacing[i]/new_spacing[i])  
        new_shape.append(new_zyx)  
    resize_factor = []  
    for i in range(3):  
        resize_zyx = new_shape[i]/imgs.shape[i]  
        resize_factor.append(resize_zyx) 
    imgs = zoom(imgs, resize_factor, mode = 'nearest')   
    return imgs
datflag_file = 'datflag641-plus.csv'
data_file = '/course75/RealData/CT/'
model_stage ='task2'
test_id = list(range(285,293))
test_id.extend(list(range(419,439)))
test_id.extend(list(range(1,84)))
test_id.extend(list(range(439,473)))
test_id.extend([259,294])
print('当前test长度',len(test_id))
# 
train_index, test_index = get_par_index(datflag_file=datflag_file, data_file=data_file, stage = model_stage, test_id=test_id)
datflag_0 = pd.read_csv(datflag_file)
train_id = datflag_0.loc[train_index,'ID'].unique()
datflag_train=datflag_0.loc[train_index,:]
datflag_test=datflag_0.loc[test_index,:]
datflag_cstage=datflag_0.loc[train_index+test_index,:]

CT_path = "/course75/RealData/CT/"
def extract_one_node(ii):
    current_feature=[]
    i = datflag_cstage['ID'].unique()[ii]
    file_path = CT_path+ str(i)
    reader = sitk.ImageSeriesReader()
    dcm_series = reader.GetGDCMSeriesFileNames(file_path)
    reader.SetFileNames(dcm_series)
    img = reader.Execute()
    origin = img.GetOrigin()
    spacing = img.GetSpacing()
    direction = img.GetDirection()
    img_array = sitk.GetArrayFromImage(img)
    img_array = np.flip(img_array, axis=[0])
    num_z, height, width = img_array.shape    
    print('CT大小：',img_array.shape)
    datmini = datflag_cstage[datflag_cstage['ID'] == i]
    for j in datmini.index: 
        img = img_array.copy()
        zmin = int(datmini['Z'][j].split('-')[0])
        zmax = int(datmini['Z'][j].split('-')[1])
        z_mean = int((zmax+zmin)/2)
        x = datmini['X'][j]
        y = datmini['Y'][j]
        diam = datmini['Size.cm.'][j]
        
        
        imgsmall = img[np.max([(z_mean-view_size),0]):np.min([z_mean+view_size,img.shape[0]])] 
        NMasks = []
        for i in range(imgsmall.shape[0]):
            NMasks.append(get_Nodemask(imgsmall[i],x,y,diam,spacing, c_diam=view_size)[1])
        NMasks=np.array(NMasks)
        print(imgsmall.shape, NMasks.shape)
        s_img = sitk.GetImageFromArray(imgsmall)
        s_img.SetOrigin(origin)
        s_img.SetSpacing(spacing)
        s_img.SetDirection(direction)
        s_Nmask = sitk.GetImageFromArray(NMasks)
        s_Nmask.SetOrigin(origin)
        s_Nmask.SetSpacing(spacing)
        s_Nmask.SetDirection(direction)
        N_3D_feature, N_3D_feature_info=Extract_Features(s_img, s_Nmask, 'params.yaml')  
        N_3D_feature['index']=j
        N_3D_feature['flag']=datmini['flag'][j]
        current_feature.append(N_3D_feature)
    return current_feature

for view_size in [8,16,24]:
    temp_result=[]
    N_Feature_list=[]
    from multiprocessing import Pool
    pool = Pool(processes=12)
    for ii in range(len(datflag_cstage['ID'].unique())):
        temp_result.append(pool.apply_async(extract_one_node,(ii, )))
    pool.close()
    pool.join()
    for ii in range(len(temp_result)):
        N_Feature_list.extend(temp_result[ii].get())
        
    xx= pd.DataFrame(N_Feature_list)
    for column in xx.columns:
        try:
            xx[column] = xx[column].apply(lambda x: x.real)
        except:
            print('END')
    xx.fillna('0').to_csv(model_stage+'_'+'Node_Features_'+str(view_size*2)+'.csv',index=False,sep=',')