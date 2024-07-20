# MHA-FF
This repository contains the code for **MHA-FF: A Multi-Head Attention Model for Adaptive Fusion of Heterogeneous Features in Lung Nodule Classification**.
## Introduction
In medical scenarios, image classification using heterogeneous feature poses a significant challenge. This is especially pertinent in lung cancer diagnosis, where accurately distinguishing between pre-invasive lung adenocarcinoma (Pre-IAs) and invasive adenocarcinoma (IAs), as well as classifying the subtypes of IAs, is crucial. 

To enhance accurate prediction, we utilized a combination of deep learning and radiomics features, each extracted via distince methodologies from computed tomography images. Specifically, we integrated these heterogeneous features using an adaptive fusion module that can learn attention-based discriminative features. 

The effectiveness of our proposed method is demonstrated using real-world data collected from a multi-center cohort.

## Model
Figure 1 shows the flowchart of the proposed MHAFF. 


![Model.png](https://github.com/fxiaotong432/MHA-FF/blob/main/Model.png)

The input is an array of CT images with regard to a specific patient, with nodule centers manually labelled, in advance. The input is first passed through two modules (A and B), in parallel. Module A involves knowledge-driven radiomics feature extraction, while Module B involves data-driven deep feature extraction. Next, both the radiomics and deep features are fed into a multi-head attentional block (i.e., module C) for feature fusion. The final probabilistic prediction of LUAD subtype (i.e., HDA, MDA, and PDA) is obtained by mean pooling plus softmax activation.  
## Requirements

The code is written in Python and requires the following packages: 

* Python 3.8.12 
* TensorFlow 2.11.0
* Torch 1.9.1
* Monai 1.1.0
* SimpleITK 2.2.0
* Matplotlib 3.4.3 
* Numpy 1.21.2 
* Pandas 1.3.4 
* Sklearn 1.1.3
* Scipy 1.7.1
