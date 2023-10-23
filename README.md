# Discriminating similar languages and dialects using deep learning

### Goal of the project
The primary objective is to address the "Evaluation Campaign on Natural Language Processing (NLP) for Similar Languages, Varieties, and Dialects" using state-of-the-art language models and attempt to surpass the best performances, which were primarily achieved by traditional methods. At the time of the challenge (2017), deep learning models were significantly outperformed by traditional classifiers. In this project, I aim to develop and deploy newer language models and compare their performance (LLMs) to "more traditional" classifiers and deep learning models (LSTM, RNN, ...).

Therefore, the project type is crearly a "beat the classics". 

### Relevant scientific papers
* Findings of the VarDial Evaluation Campaign 2017
  * https://aclanthology.org/W17-1201.pdf
    * The paper presents the results of the fourth edition of the shared challenge: "VarDial Evaluation Campaign on Natural Language Processing (NLP) for Similar Languages, Varieties and Dialects", which took part in year 2017. Four tasks about Discriminating between Similar Languages (DSL), Dialect Identification (GDI), and similar were addressed by 19 teams. 
  * https://aclanthology.org/W17-1220.pdf
    * This paper addresses specifically the German Dialect Identification (GDI) task of the challenge. It presents three different solutions and their results.
  * http://nlp.ffzg.hr/data/publications/nljubesi/ljubesic07-language.pdf
    * This papes addresses the differences between bosnian, croatian, and serbian. It provides some knownledge and first NLP steps regarding these three similar languages.
  * In both cases, the results show that traditional classifiers outperformed deep learning approaches. 
* Automatic Arabic Dialect Classification Using Deep Learning Models
  * https://www.sciencedirect.com/science/article/pii/S1877050918321938
  * The authors use variations of CNN, RNN, LSTM networks to classify arabic dialects. 


### Datasets 
Three groups of similar languages/dialects will be used: 
* Southslavic languages (bosnian, croatian, and serbian): 
  * http://ttg.uni-saarland.de/resources/DSLCC/
 
* Argentine Spanish & Peninsular Spanish
 * http://ttg.uni-saarland.de/resources/DSLCC/

* German (swiss) dialects: 
  * https://www.spur.uzh.ch/en/departments/research/textgroup/ArchiMob.html

The first two datasets are part of the DSL Corpus Collection (DSLCC), which is a multilingual collection of short excerpts of journalistic texts. The data is very simple, the one and only input feature is the content of the excerpt of the journalistic text. The target variable is the language (in our dataset: the abbreviation of the language). 

The third dataset (swiss dialects) is slighty more limited containing 18 interviews. Neverthless, the autors of the dataset claim to provide around 145.000 tokens for training & testing after some preprocessing steps. 

### Work-breakdown 

Task  | Effort in hours
------------- | -------------
Data preprocessing | 10
Benchmark with traditional classifier & "tradinational" RNN of some type | 12
Development of LLMs | 15
Training and fine-tuning | 30
Comparison of different LLMs models vs more traditional models & vs winner of last shared challenge | 5
Delivarable product as application | 25
Report & final presentation | 8
