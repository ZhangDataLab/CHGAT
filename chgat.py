import json
import os
import torch
import pandas as pd
import numpy as np
import pandas as pd
import warnings
import random
import logging
import argparse
import shutil
import copy

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report,f1_score
from torch.optim import AdamW
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from utils import *
from transformers import BertModel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from chgat_model import classifier

warnings.filterwarnings('ignore')

random_seed = 42
def seed_everything(seed=random_seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()

import argparse
parser = argparse.ArgumentParser(description='Training model')
parser.add_argument('--task', type=str, default='chgat')
args = parser.parse_args()
TASK = args.task

local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ['WORLD_SIZE'])
print(world_size)
rank = int(os.environ['RANK'])
print("Current local_rank: %d" %  local_rank)
if local_rank != -1:
    dist_backend = 'nccl'
    dist.init_process_group(backend=dist_backend, world_size=world_size, rank=rank)
torch.cuda.set_device(local_rank) 


def load_data(config):
    train_data = pd.read_csv(config['train_data_dir'])
    val_data = pd.read_csv(config['val_data_dir'])
    test_data = pd.read_csv(config['test_data_dir'])
    type_list = {'train':[NameGenderDataset(train_data,config), config['train_batch_size']],\
                    'val':[NameGenderDataset(val_data,config), config['dev_batch_size']],\
                 'test':[NameGenderDataset(test_data,config), config['test_batch_size']],}

    dataList = []
    for key,(data, batch_size) in type_list.items():
        sampler = torch.utils.data.distributed.DistributedSampler(data)
        if key == 'train':
            dataList.append(DataLoader(data,sampler=sampler,batch_size=int(batch_size/world_size),\
                                collate_fn=collate_fn,num_workers=world_size))
        else:
            dataList.append(DataLoader(data,sampler=sampler,batch_size=int(batch_size/world_size),\
                                collate_fn=collate_fn,num_workers=world_size))
    return dataList


def load_model(config,metapaths=None,n_gpu=0):
    model = classifier(2,config,metapaths=[['c-sem'],['c-pho'],['c-nonsp']],metapaths2=[['c-pinyin']])
    params = list(model.named_parameters())
    no_decay = ['bias,','LayerNorm','pooler']
    other = ['pos_emb','word_node','gat_','structure','word_attention']
    no_main = no_decay + other
    param_group = [
            {'params':[p for n,p in params if not any(nd in n for nd in no_main)],'weight_decay':1e-2,'lr':5e-5},
            {'params':[p for n,p in params if not any(nd in n for nd in other) and any(nd in n for nd in no_decay) ],'weight_decay':0,'lr':1e-5},
            {'params':[p for n,p in params if any(nd in n for nd in other) and not any(nd in n for nd in no_decay) ],'weight_decay':5e-4,'lr':1e-3}]
    optimizer = AdamW(param_group,lr=1e-5,eps=1e-7)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    if config['device'] == 'cuda':
        n_gpu = torch.cuda.device_count()
        print("Cuda Device number is %d" % n_gpu)
        if n_gpu > 1:
            model = model.to(local_rank)
            model = DDP(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
    optimizer = nn.DataParallel(optimizer,device_ids=np.arange(n_gpu))
    return model, optimizer, n_gpu,scheduler


def train(model,config,train_dataloader,dev_dataloader,test_dataloader,optimizer,scheduler,n_gpu):
    cr_loss = nn.CrossEntropyLoss()
    nb_tr_steps = 0

    patiences = config['patiences']
    pateience_name = 'patience_' + str(patiences)

    model.train()
    writer_dir = os.path.join(config['writer_dir'],TASK)
    if os.path.exists(writer_dir) is not True:
        try:
            os.makedirs(writer_dir)
        except:
            pass
    writer = SummaryWriter(writer_dir)

    best_val_model_acc = None
    dev_best_acc = 0
    for idx in range(int(config['num_train_epochs'])):
        step = 0
        print("process:{0}, epoch:{1}, dataLoder Length:{2}".format(local_rank,idx,len(train_dataloader)))
        total_loss = 0
        for batch in train_dataloader:
            model.train()
            nb_tr_steps += 1
            name_feat_list = batch['name_feature'].to(local_rank)
            label_list = batch['label_list'].to(local_rank)
            name_list = batch['name']
            logits = model(name_feat_list,name_list,labels=label_list,device=local_rank)
            loss = cr_loss(logits,label_list.view(-1))
            if n_gpu > 1:
                loss = loss.mean()
            total_loss += loss.item() / len(batch)
            loss.backward()

            optimizer.module.step()
            optimizer.module.zero_grad()
            writer.add_scalar(tag='loss/train_loss',scalar_value=loss.item(),global_step=nb_tr_steps)
            step+=1
        writer.add_scalar(tag='loss/train_total_loss',scalar_value=total_loss,global_step=idx)

        tmp_dev_loss, tmp_dev_acc, tmp_dev_f1 = eval_checkpoint(config, model,dev_dataloader,cr_loss)
        if tmp_dev_acc > dev_best_acc:
            dev_best_acc = tmp_dev_acc
            best_val_model_acc = copy.deepcopy(model.module ) if hasattr(model, "module") else copy.deepcopy(model)          
        scheduler.step()

        save_file_name = os.path.join(config['output_dir'], "model/")
        if os.path.exists(save_file_name) is not True:
            try:
                os.makedirs(save_file_name)
            except:
                pass

        if dist.get_rank() == 0:
            output_model_file = save_file_name + TASK + pateience_name+'_acc_best.bin'
            torch.save(best_val_model_acc.state_dict(), output_model_file)
            

@torch.no_grad()
def eval_checkpoint(config, model_object, eval_dataloader,cr_loss,mode=False):
    model_object.eval()
    eval_loss = 0
    eval_accuracy = []
    eval_f1 = []
    logits_all = []
    labels_all = []
    eval_steps = 0
    index_start = 0
    print("eval dataloader lengeth : %d" % len(eval_dataloader))
    for v,batch in enumerate(eval_dataloader):
        if v%50 == 0:
            print("current eval index: %d" % v)

        index_start+=1
        name_feat_list = batch['name_feature'].to(local_rank)
        label_list = batch['label_list'].to(local_rank)
        name_list = batch['name']
        
        logits =model_object(name_feat_list,name_list,labels=label_list,device=local_rank)
        loss = cr_loss(logits,label_list.view(-1))
        logits = torch.argmax(logits, dim=1)
        
        
        label_ids = label_list.clone().detach().cpu().numpy()
        logits = logits.clone().detach().cpu().numpy()
        eval_loss += loss.item()

        logits_all+=logits.tolist()
        labels_all+=label_ids.tolist()
        eval_steps += 1

    average_loss = round(eval_loss / eval_steps, 4)
    eval_accuracy = float(accuracy_score(y_true=labels_all, y_pred=logits_all))
    eval_f1 = float(f1_score(y_true=labels_all, y_pred=logits_all))
    rep = classification_report(labels_all,logits_all,digits=6)
    if mode==False:
        print('Validation : ',rep)
    return average_loss, eval_accuracy, eval_f1 



def main():
    model, optimizer, n_gpu, scheduler = load_model(config)
    train_dataloader,dev_dataloader,test_dataloader = load_data(config)
    train(model,config,train_dataloader,dev_dataloader,test_dataloader,optimizer,scheduler,n_gpu)


if __name__ == '__main__':
    time_begin = time.time()
    config = json.load(open('./config/config.json'))
    if os.path.exists(os.path.join(config['writer_dir'],TASK)) is not True:
        try:
            os.makedirs(os.path.join(config['writer_dir'],TASK))
        except:
            pass

    for f in os.listdir(os.path.join(config['writer_dir'],TASK)):
        try:
            os.remove(os.path.join(config['writer_dir'],TASK,f))
        except:
            pass
    main()
    time_end = time.time()
    time_use = int(time_end - time_begin)
    hour = time_use // 3600
    minute = (time_use - hour*3600) // 60
    seconds = time_use - hour*3600 - minute*60
    print("Running hours: {0}, minutes:{1}, seconds{2} ".format(hour,minute,seconds))