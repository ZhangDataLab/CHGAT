import json
import dill
import torch
import numpy as np
import time

from tqdm import tqdm
from transformers import BertTokenizer

def save_pkl(path,obj):
    with open(path, 'wb') as f:
        pickle.dump(obj,f)
        
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)  

class NameGenderDataset(torch.utils.data.Dataset):
    def __init__(self,file,config):
        self.file = file
        self.bert = BertTokenizer.from_pretrained(config['chinese_bert_path'], do_lower_case=True)
        self.pad_index = self.bert.convert_tokens_to_ids(self.bert.pad_token)
        self.name_max_length =4
        self.pronunciation_max_lenth =12
        name_list = self.file['name'].tolist()
        pinyin_list = self.file['pinyin'].tolist()
        label_list = self.file['label'].tolist()
        self.origin_name = name_list
        self.name_list = []
        self.pinyin_list =[]
        for name, pinyin in zip(name_list, pinyin_list):
            name_bert = self.bert.encode(name, max_length=self.name_max_length, pad_to_max_length=True, truncation=True)
            try:
                py_bert = self.bert.encode(pinyin, max_length=self.pronunciation_max_lenth, pad_to_max_length=True, truncation=True)
            except:
                py_bert = list(np.zeros(self.pronunciation_max_lenth))
                py_bert = [int(x) for x in py_bert]
            self.name_list.append(name_bert)
            self.pinyin_list.append(py_bert)
        self.name_list=  np.array(self.name_list)
        self.name_list = torch.from_numpy(self.name_list)
        self.pinyin_list = np.array(self.pinyin_list)
        self.pinyin_list = torch.from_numpy(self.pinyin_list)
        self.label_list = torch.from_numpy(np.array(label_list))

    def  __getitem__(self,index):
        return self.name_list[index],self.pinyin_list[index],self.label_list[index],self.origin_name[index]
    
    def __len__(self):
        return len(self.file)

        
def collate_fn(data):
    name_feat_list = [x[0] for x in data]
    pinyin_feat_list = [x[1] for x in data]
    label_list = [x[2] for x in data]
    name_list = [x[3] for x in data]
    name_feat_list = torch.stack(name_feat_list)
    pinyin_feat_list = torch.stack(pinyin_feat_list)
    label_list = torch.tensor(label_list)
    return {'name_feature':name_feat_list,'pinyin_feature':pinyin_feat_list,'label_list':label_list,'name':name_list}


