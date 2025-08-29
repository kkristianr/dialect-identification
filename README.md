
## Discriminating similar languages and dialects using Transformers

_Project for the "Applied Deep Learning" Course at the TU Wien. 3 ECTS/90hours including benchmark definition, data processing, model selection, training and finetuning, evaluation, discussion, presentation and deployment of demo application._ 

Link to presentation (_5 min_): https://www.youtube.com/watch?v=JFo7F-MQ7lM

I present the insights, code and the results of my investigation and experimantion for discriminating similar laguanges and/or dialects using modern deep learning approachs, transformers to be more precise. 
The initial idea of the project was to "beat the classics" in multi (similar) languages/dialects classification. In previous open challenges in this field, traditional classifiers such as customized fine-tuned linear SVM, emsembles, and Naive Bayes using n-gram had the upperhand [1]. Only in the last years more generic but also fine-tuined as well challenge-suited transformers came into play to prove their use case in such challenges[2]. 
  
> [!NOTE]<br>
> During the search for suitable Transformers, an underlying consideration led to small readjustments of the goals of the project.
> At the beginning I believed it would be feasible to approach different groups of languages (such as different versions of spanish, of portugese, german dialects and serbian/croatian/bosnian) all together, where the numbers of classes to predict is the total number of dialects/languages in the data set. In this case, the data set had four different "parent" languages and 12 target labels.
> Furthermore, the essential consideration was: "Is such a classification between different dialects and languages across the world even needed?". It was then clear, that the main interest is to be able to identify dialects or similar languages within a region (or within a "parent" language).   

As a result for this experiment, both approaches would be examinated and compared. In the second part of the experiment, we examine the classification of three similar south-slavic languages (bosnian, croatian, serbian). 

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
The goal was to come close the winners of the VarDial Challenge 2017, which means having a F1 weighted score around 90% for the full task. 

To get an overall idea of the task and its challenges, I decided to implement the two following (strong) baselines, which are reported to work well in this task.
* SVM (linear kernel) 
* RNN based on bidirectional LSTM
  
The results were still far from an ideal score. Not being an easy task, I accepted the results of my baselines and focused more on the deep learning approach. The results of the baselines are despicted in the "Results" table. 

## 3. Code
#### Structure
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
│   └── experiment
│       ├── base_SVM_bks.ipynb
│       ├── base_RNN_bks.ipynb
│       ├── bert_multilingual_bks.ipynb
│       └── bertic_bks.ipynb
│── src (not really used since everything ran on Colab) 
│    ├── DataModule.py
│    ├── LanguagesDataSet.py
│    ├── utlis.py
│    └── models (to store the fine-tuined models and vocabs from Colab)
│         ├── ..
└── requirements.txt
```

#### Data sets and preprocessing
The data is relatively simple, the one and only input feature is the content of the excerpt of the journalistic text. The target variable is the language/dialect (in our dataset: the abbreviation of the language/dialect).
The input text was roughly preprocessed, keeping special characters over the letters and not applying lower-case format for all tokens. It has to be said that the baseline results were slightly higher with more language-preprocessing such as removing all special letters and so on. Since we dealt with different languages and as result different uni-codes, the pre-processing steps were not that restricted. 

As result, we have a train_prepro.csv and a test_prepro.csv with the following properties
* Target labels: 12 
* Train size: 162190 - where 20% is used as validation test in the training
* Test size: 20116 - used fully as "unseen" data to evaluate the fine-tuined model.

#### Models & Evaluation
Given the nature of the data set, we opted for a transfomer language model, named [BERT multilingual base model (cased)](https://huggingface.co/bert-base-multilingual-cased), which was trained on top of 104 languages. 
For our sub-task on discriminating between bosnian, croatian and serbian, we experimented a bit with [BERTić](https://huggingface.co/classla/bcms-bertic) transformer language model as well. 
The models are implemented in seperate notebooks and were trained on Google Colab using GPU ressources. The training times for two/three epochs were inbetween 4-8 hours using between 15 and 30GBs of GPU. An upgrade to Google+Pro was at most point inevitable. 

## 4. Results
#### All dialects

Model  | Epochs  | Accuracy | F1(weighted)
------------- | ------------- |------------- | -------------
base_linearSVM | - | 0.75 | 0.75
base_biLSTM |4 | 0.70 | 0.70
BERT_Multilingual | 2 | 0.80 | 0.80

We were able to beat our baselines, however we missed our initial goal of 0.9 regarding the F1 score. It's difficult to interpret these results not having a strong LLM/bert baseline, however I am sure that with some constrastive learning, hyperparameter optimization and eventually more epochs, it's possible to achieve a higher score. 

#### Experiment on BKS  

Model  | Epochs  | Accuracy | F1(weighted)
------------- | -------------| ------------- | -------------
base_linearSVM | - | 0.81 | 0.81
base_biLSTM | 3 | 0.75 | 0.76
BERT_Multilingual |3 | 0.85 | 0.85
BERTić | 3 | 0.85 | 0.86

In this experiment, the goal was to use a better trained transformer for a similar group of languages. [BERTić](https://huggingface.co/classla/bcms-bertic) was the choice for this sub-task. For our task, BERTić outperformed the BERT multilingual base model. Overall, we could beat our baselines with the use of transformer language models. 


## 5. Discussion and conclusion 
Surprisingly the invested effort for the "hacking" part releaved itself to be close to the overall estimated effort. However, small differences occured and are visible in the following table. The pure development of an architecture for the transfomers to get it running was relatively fast, unfortunately only during the training and results examination many problems and poor performance come into sight. A lot of adjustments and re-runs to come close to our baseline were necessary. Patience was key, but my patience became quickly frustration and I switched to Google Colab+Pro to speed up my training times in order to try different paramenters and experiment a bit more. 


Task  | Estimated effort in hours  | Actual effort in hours 
------------- | ------------- | -------------
Data preprocessing | 10 | 12
Benchmark with traditional classifier & "traditional" RNN of some type | 12 | 12
Development of LLMs | 15 | 8
Training and fine-tuning | 30 | 40-50
Comparison of different LLMs models vs more traditional models & vs winner of last shared challenge | 5 | 4
Delivarable product as application | 25 | 20 
Report & final presentation | 8 | 5

### References 
[1] Marcos Zampieri, Shervin Malmasi,Nikola Ljubesic, Preslav Nakov, Ahmed Ali, Jörg Tiedemann, Yves Scherrer, Noemi Aepli, 2017, Findings of the VarDial Evaluation Campaign

[2] Aepli et al., VarDial 2022, [Findings of the VarDial Evaluation Campaign 2022](https://aclanthology.org/2022.vardial-1.1) 

