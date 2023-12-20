
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from google.colab import drive
import numpy as np
import torch
import random
import pandas as pd




class LanguagesDataSet(Dataset):
    def __init__(self, only_bks=False, myDrive=True):
        if myDrive == True:
            drive.mount('/content/drive')
            self.train = pd.read_csv('/content/drive/MyDrive/ds_data/train_prepro.csv')
            self.test = pd.read_csv('/content/drive/MyDrive/ds_data/test_prepro.csv')
        else:
            self.train = pd.read_csv('data/train_prepro.csv')
            self.test = pd.read_csv('data/test_prepro.csv')

        if only_bks == True:
            self.train = self.train[self.train['GROUP'] == 'bks']
            self.test = self.test[self.test['GROUP'] == 'bks']
        
        self.vocab_size = build_vocab_from_iterator(self.__iter__(), specials=["<unk>"]).get_vocab_size()
        self.number_labels = len(self.train['GROUP'].unique())

    def __len__(self):
        return len(self.train)
    
    def __getitem__(self, idx):
        return self.train.iloc[idx]['TEXT'], self.train.iloc[idx]['GROUP']
    
    def __iter__(self):
        for i in range(len(self.train)):
            yield self.train.iloc[i]['TEXT'].split()

    def get_vocab_size(self):
        return self.vocab_size
    
    def get_test(self):
        return self.test
    
    def get_train(self):
        return self.train
    



