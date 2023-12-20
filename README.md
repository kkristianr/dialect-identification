
## Part 2 of: "Discriminating similar languages and dialects using Transformers"

In this second submission for the "Applied Deep Learning" Course at the TU Wien, I present the insights, code and the results of my investigation and experimantion for discriminating similar laguanges and/or dialects using modern deep learning approachs, transformers to be more precise. 
The initial idea of the project was to "beat the classics" in multi (similar) languages/dialects classification. In previous open challenges in this field, traditional classifiers such as customized fine-tuned linear SVM, emsembles, and Naive Bayes using n-gram had the upperhand [1]. Only in the last years more generic but also fine-tuined as well challenge-suited transformers came into play to prove their use case in such challenges. 

To get an overall idea of the task and its challenges, I decided to implement the two following (strong) baselines, which are reported to work well in this task.
* SVM (linear kernel) 
* RNN based on bidirectional LSTM
  
> [!NOTE]<br>
> During the search for suitable Transformers, an underlying consideration led to small readjustments of the goals of the project.
> At the beginning I believed it would be feasible to approach different groups of languages (such as different versions of spanish, of portugese, german dialects and serbian/croatian/bosnian) all together, where the numbers of classes to predict is the total number of dialects/languages in the data set. In this case, the data set had four different "parent" languages and 12 target labels.
> Furthermore, the essential consideration was: "Is such a classification between different dialects and languages across the world even needed?". It was then clear, that the main interest is to be able to identify dialects or similar languages within a region (or within a "parent" language).   

As a result for this submission, both approaches would be examinated and compared. In the second part of the submission, we examine the discrimination of three similar south-slavic languages (bosnian, croatian, serbian). 

## Table of contents

- [1. Dataset and preprocessing](#1-Dataset-and-preprocessing)
- [2. Goals & Baselines](#2-Goals-&-Baselines)
- [3. Code](#3-Code)
- [4. Results](#4-Results)
  - [4.1 All dialects](#-all-dialects)
  - [4.2 Experiment on BKS](#-Experiment-on-BKS)
- [5. Discussion and Conclusion](#5-Discussion_and_Conclusion)

## 1. Dataset and preprocessing

* Target labels: 12 
* Train size: where 0.2 is used as validation test in the training
* Test size: used fully as "unseen" data to evaluate the fine-tuined model.
  
## 2. Goals & Baselines

## 3. Code


#### Dataset


```
├── data
│   ├── all
│   │   ├── train_prepro.csv
│   │   └── test_prepro.csv
│   ├── ..
│   └── data_collection.ipynb
├── scripts
│   ├── base_SVM.ipynb
│   ├── base_RNN.ipynb
│   ├── bert_multilingual.ipynb
│   ├── experiment
│   │   ├── base_SVM_bks.ipynb
│   │   ├── base_RNN_bks.ipynb
│   │   ├── bert_multilingual_bks.ipynb
│   │   └── bertic.ipynb
├── src (not really used since everything ran on Colab) 
│   ├── DataModule.py
│   ├── LanguagesDataSet.py
├── 
│   └── 

```




## 4. Results

#### All dialects

Model  | Epochs  | Accuracy | F1(weighted)
------------- | ------------- |------------- | -------------
base_linearSVM | - | 0.75 | 0.75
base_biLSTM |4 | 0.70 | 0.70
bert_multilingual | 2 | 0.80 | 0.80


#### Experiment on BKS  

Model  | Epochs  | Accuracy | F1(weighted)
------------- | -------------| ------------- | -------------
base_linearSVM | - | 0.81 | 0.81
base_biLSTM | 3 | 0.75 | 0.76
bert_multilingual |3 | 0.85 | 0.85
bertic | 3 | 0.72| 0.76




## 5. Discussion and conclusion 
Surprisingly the invested effort for the second submission "hacking" releaved itself to be close to the overall estimated effort. However, small differences occured and are visible in the following table. The pure development of an architecture for the transfomers to get it running was relatively fast, unfortunately only during the training and results examination many problems and poor performance come into sight. A lot of adjustments and re-runs to come close to our baseline were necessary. Patience was key, but my patience became quickly frustration and I switched to Google Colab+Pro to speed up my training times in order to try different paramenters and experiment a bit more. 


Task  | Estimated effort in hours  | Actual effort in hours 
------------- | ------------- | -------------
Data preprocessing | 10 | 12
Benchmark with traditional classifier & "traditional" RNN of some type | 12 | 12
Development of LLMs | 15 | 8
Training and fine-tuning | 30 | 40-50
Comparison of different LLMs models vs more traditional models & vs winner of last shared challenge | 5 | 4
Delivarable product as application | 25 | ... 
Report & final presentation | 8 | ...

### References 
[1] Marcos Zampieri, Shervin Malmasi,Nikola Ljubesic, Preslav Nakov, Ahmed Ali, Jörg Tiedemann, Yves Scherrer, Noemi Aepli, 2017, Findings of the VarDial Evaluation Campaign

