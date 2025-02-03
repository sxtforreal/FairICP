## FairICP: Identifying Biases and Increasing Transparency at the Point of Decision in Post-Implementation Clinical Decision Support using Inductive Conformal Prediction
##### By Xiaotan Sun, Makiya Nakashima, Christopher Nguyen, Po-Hao Chen, W.H. Wilson Tang, Deborah Kwon, David Chen

## Introduction
In this repository we release the code to implement FairICP.

## Dataset
Since FairICP is a post-process/post-implementation framework, we directly used the prediction results from others' previous works. No model re-training is involved in our work. 
1. ICM/NICM Cardiac MRI Data (Cleveland Clinic): This is a private dataset.
2. CheXpert (Stanford Machine Learning Group): B. Glocker et al pre-trained a disease detection model on CheXpert Dataset <[Algorithmic encoding of protected characteristics in chest X-ray disease detection models](https://www.thelancet.com/journals/ebiom/article/PIIS2352-3964(23)00032-4/fulltext)>. Data processing can be found in **CheXpert Dataset** folder.
3. ISIC Challenge 2018 - Task 3 (The International Skin Imaging Collaboration): T. Kalb et al pre-trained a disease detection model on ISIC Challenge 2018 - Task 3 Dataset <[Revisiting Skin Tone Fairness in Dermatological Lesion Classification](https://arxiv.org/abs/2308.09640v1)>. Data processing can be found in **ISIC 2018 Dataset** folder.

## Inference
### Create runs
Inferencing can be performed using the code in **main.py** and **func.py**.
After loading the processed prediction results, we randomly sample 100 times and save the results in the runs folders.
### Bias Mitigation
The **evaluation** function evaluates performances of all 5 frameworks given the runs data and specified ICP hyperparameters.
The **figure2**, **figure3**, and **test** functions create Figure 2, 3, and hypothesis tests in each dataset respectively.
### Decision Threshold Optimization
The **simulate_data** function generates the simulation data, the **visualize_simulation** function produces Supplementary Figure 1.
The **alpha_calsize** function evaluates the changes in metrics against the changes in both confidence levels and calibration sizes, the **plot_3d** function produces Figure 4.
The **adjustable_alpha** function evaluates the changes in metrics against the changes in confidence levels, the **plot_alpha** function produces Figure 5.
