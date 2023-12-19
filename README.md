
## Part 2 of: "Discriminating similar languages and dialects using Transformers"

In this second submission for the "Applied Deep Learning" Course at the TU Wien, I present the insights, code and the results of my investigation and experimantion for discriminating similar laguanges and/or dialects using modern deep learning approachs, transformers to be more precise. 
The initial idea of the project is to "beat the classics" in multi (similar) languages/dialects classification. In previous open challenges in this field, traditional classifiers such as Linear SVM and Naive Bayese using n-gram had the upperhand. Only in the last years more generic but also fine-tuined and challenge-suited transformers came into play to prove their use case in such challenges. 

To get an overall idea of the task and its challenges, I decided to implement the following two strong baselines, which are reported to work well in this task.
* SVM (linear kernel) 
* RNN based on a doubled-stacked bidirectional LSTM layer

During the search for suitable Transformers, an underlying consideration led to small readjustments of the goals of the project. At the beginning I believed it would be feasible to approach different groups of languages (such as different versions of spanish, of portugese, german dialects and serbian/xroatian/bosnian) all together, 
where the numbers of classes to predict is the total number of dialects/languages in the data set. In this case, the data set had four different "parent" languages and 12 target labels. Furthermore, the essential consideration was: "Is such a classification between different dialects and languages across the world even needed?". It was then clear, that the main interest is to be able to identify dialects or similar languages within a region (or within a "parent" language).   

As a result for this submission, both approaches would be examinated and compared. In the second part of the submission, we examine the discrimination of three similar south-slavic languages (bosnian, croatian, serbian). 




## Result for all dialects in the data set 

* Target labels: 12 
* Train size: where 0.2 is used as validation test in the training
* Test size: used fully as "unseen" data to evaluate the fine-tuined model. 




Model  | Epochs  | Accuracy | F1(weighted)
------------- | ------------- |------------- | -------------
base_linearSVM | 30 | 0.77 | 0.77
base_biLSTM | | |
bert_multilingual | | |


## Results for the experiment: croatian-bosnian-serbian 

Model  | Epochs  | Accuracy | F1(weighted)
------------- | -------------| ------------- | -------------
base_linearSVM | 30 | 0.73 | 0.77
base_biLSTM | 10 | |
bert_multilingual |3 | 0.85 | 0.85
bertic |3 | 0.72| 0.76




● the error metric you specified
● the target of that error metric that you want to achieve
● the actually achieved value of that metric
● the amount of time you spent on each task, according to your own work breakdown
structure.

## Resourcen report 
Surprisingly the invested effort for the second submission "hacking" releaved itself to be close to the estimated effort. 


Task  | Effort in hours
------------- | -------------
Data preprocessing | 10
Benchmark with traditional classifier & "tradinational" RNN of some type | 12
Development of LLMs | 15
Training and fine-tuning | 30
Comparison of different LLMs models vs more traditional models & vs winner of last shared challenge | 5
Delivarable product as application | 25
Report & final presentation | 8

### References 
