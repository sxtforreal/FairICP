## FairICP: Identifying Biases and Increasing Transparency at the Point of Decision in Post-Implementation Clinical Decision Support using Inductive Conformal Prediction
##### By Xiaotan Sun, Makiya Nakashima, Christopher Nguyen, Po-Hao Chen, W.H. Wilson Tang, Deborah Kwon, David Chen

## Introduction
In this repository we release the code to implement FairICP.

## Dataset
Since FairICP is a post-process/post-implementation framework, we directly used the prediction results from others' previous works. No model re-training is involved in our work. 
1. ICM/NICM Cardiac MRI Data (Cleveland Clinic): This is a private dataset.
2. CheXpert (Stanford Machine Learning Group): B. Glocker et al pre-trained a disease detection model on CheXpert Dataset <[Algorithmic encoding of protected characteristics in chest X-ray disease detection models](https://www.thelancet.com/journals/ebiom/article/PIIS2352-3964(23)00032-4/fulltext)>. Data processing can be found in **CheXpert Dataset** folder.
3. ISIC Challenge 2018 - Task 3 (The International Skin Imaging Collaboration):
