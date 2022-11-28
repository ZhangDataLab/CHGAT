import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from torch.nn import CrossEntropyLoss
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler
from dgl.nn.pytorch.glob import MaxPooling, AvgPooling, SumPooling
from dgl.nn.pytorch import GATConv
from utils import *
from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler
from random import random

all_word_graph_data_dic = load_pkl('./data/word2pinyin2real_word2formation2sem2pho_dict.pkl')
all_pinyin_dict = load_pkl('./data/pinyin_index_dict.pkl')
word_node_tmp = load_pkl('./data/chgat_formation_graph.pkl')
vocab_dict = load_pkl('./data/word_and_component_vocab_list.pkl')
vocab_list = vocab_dict + list(all_word_graph_data_dic.keys())
vocab_dict = {w:i for i,w in enumerate(vocab_list)}

'''
Reference : https://github.com/dmlc/dgl/blob/master/examples/pytorch/han/model.py
'''
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)

        return (beta * z).sum(1)


class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, feat_drop, attn_drop,
                 alpha=0.2, residual=True, agg_modes='flatten'):
        super(GATLayer, self).__init__() 
        self.gat_conv = GATConv(in_feats=in_feats, out_feats=out_feats, num_heads=num_heads,
                                feat_drop=feat_drop, attn_drop=attn_drop,
                                negative_slope=alpha, residual=residual)
        assert agg_modes in ['flatten', 'mean']
        self.agg_modes = agg_modes

    def forward(self, bg, feats):
        feats = self.gat_conv(bg, feats)
        if self.agg_modes == 'flatten':
            feats = feats.flatten(1)
        else:
            feats = feats.mean(1)
        return feats


class classifier(nn.Module):
    def __init__(self,num_labels,config,metapaths=None,metapaths2=None):
        super(classifier,self).__init__()
        self.num_labels = num_labels
        
        self.pos_emb = nn.Embedding(43,768)
        self.word_node_emb = nn.Embedding(len(vocab_dict),768)
        self.pinyin_emb = nn.Embedding(len(all_pinyin_dict),768)
        in_feat = 768
        out_feat = 32
        
        self.meta_paths = list(tuple(meta_path) for meta_path in metapaths)
        self.meta_paths2 = list(tuple(meta_path) for meta_path in metapaths2)
        
        #----------------------------------------------#
        self.gat_layers = nn.ModuleList()  
        self.gat_linear_layers = nn.ModuleList()
        for i in range(len(metapaths)):
            self.gat_layers.append(GATConv(in_feat, out_feat, 2,0.2, 0.2, activation=F.elu, allow_zero_in_degree=True))
            self.gat_linear_layers.append(nn.Linear(out_feat*2,768))
        self.structure = SemanticAttention(768)
        #----------------------------------------------#
        
        #----------------------------------------------#
        self.gat_layers2 = nn.ModuleList()  
        self.gat_linear_layers2 = nn.ModuleList()
        for i in range(len(metapaths2)):
            self.gat_layers2.append(GATConv(in_feat, out_feat, 2,0.2, 0.2, activation=F.elu, allow_zero_in_degree=True))
            self.gat_linear_layers2.append(nn.Linear(out_feat*2,768))
        self.word_attention = SemanticAttention(768)
        #----------------------------------------------#
        
        self.linear1 = nn.Linear(768,768)
        self.relu = nn.LeakyReLU(0.1)
        self.cat2encoder = nn.Linear(768*2,768)
        self.bert_model = BertModel.from_pretrained(config['chinese_bert_path'])
        self.bert_config = BertConfig(config['chinese_bert_path'])
        self.bert_encoder = BertEncoder(self.bert_config)
        self.pooler = BertPooler(self.bert_config)
        self.dropout = nn.Dropout(0.4)
        self.mlp = nn.Linear(768, num_labels)
        self.softmax = nn.Softmax(dim=1)


    def forward(self,name_embed_tensor,name_list,labels=None,device=None):
        all_name_features =  []
        all_bert_name_features = self.bert_model(name_embed_tensor)[0]
        for name in name_list:
            one_name_features_list = []
            for na in name:
                one_features = []
                if na in word_node_tmp:
                    one_dict = word_node_tmp[na]
                    one_word_list = one_dict['node_word_list']  
                    one_pinyin_list = one_dict['node_pinyin_list'] 
                    one_formation_graph = one_dict['graph']
                    if len(one_word_list)==1:
                        one_word = self.word_node_emb(torch.tensor(vocab_dict[one_word_list[0][0]]).to(device)).to(device)
                        one_pos = self.pos_emb(torch.tensor(one_word_list[0][1]).to(device)).to(device)
                        one_feat = one_word+one_pos
                        one_features.append(one_feat)
                        one_features = torch.stack(one_features).to(device)
                    else:
                        for line in one_word_list:
                            one_word = self.word_node_emb(torch.tensor(vocab_dict[line[0]]).to(device)).to(device)
                            one_pos = self.pos_emb(torch.tensor(line[1]).to(device)).to(device)
                            one_feat = one_word+one_pos.to(device)
                            one_features.append(one_feat)
                            
                        for line in one_pinyin_list:
                            one_word = self.pinyin_emb(torch.tensor(all_pinyin_dict[line[0]]).to(device)).to(device)
                            one_features.append(one_word)
                        one_features = torch.stack(one_features).to(device)
                        
                        cached_graph = one_formation_graph
                        cached_coalesced_graph = {}
                        for meta_path in self.meta_paths:
                            try:
                                cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(one_formation_graph, meta_path)
                            except:
                                continue
                        for meta_path in self.meta_paths2:
                            try:
                                cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(one_formation_graph, meta_path)
                            except:
                                continue     
                        # --------------------------------------------------------------- #
                        structure_embedding_list = []
                        for i,meta_path in enumerate(self.meta_paths):
                            if meta_path not in cached_coalesced_graph:
                                continue
                            new_g = cached_coalesced_graph[meta_path]
                            new_g = dgl.to_bidirected(new_g)
                            new_g = dgl.remove_self_loop(new_g)
                            new_g = dgl.add_self_loop(new_g)
                            new_g = new_g.to(device)
                            output_sub_graph_feat = self.gat_layers[i](new_g, one_features).to(device)
                            output_sub_graph_feat = output_sub_graph_feat.flatten(1)
                            output_sub_graph_feat =self.relu(self.gat_linear_layers[i](output_sub_graph_feat))
                            structure_embedding_list.append(output_sub_graph_feat)
                        if len(structure_embedding_list)!=0:
                            structure_embedding_list = torch.stack(structure_embedding_list,dim=1).to(device)
                            structure_embedding_list = self.structure(structure_embedding_list)
                        # ---------------------------------------------------------------- # 
                        
                        
                        # ---------------------------------------------------------------- #
                        pinyin_embedding_list = []
                        for i,meta_path in enumerate(self.meta_paths2):
                            if meta_path not in cached_coalesced_graph:
                                continue
                            new_g = cached_coalesced_graph[meta_path]
                            new_g = dgl.to_bidirected(new_g)
                            new_g = dgl.remove_self_loop(new_g)
                            new_g = dgl.add_self_loop(new_g)
                            new_g = new_g.to(device)
                            output_sub_graph_feat = self.gat_layers2[i](new_g, one_features).to(device)
                            output_sub_graph_feat = output_sub_graph_feat.flatten(1)
                            output_sub_graph_feat = self.gat_linear_layers2[i](output_sub_graph_feat)
                            pinyin_embedding_list.append(output_sub_graph_feat)
                        # ---------------------------------------------------------------- #
                        if len(structure_embedding_list)!=0:
                            pinyin_embedding_list.append(structure_embedding_list)
                        pinyin_embedding_list = torch.stack(pinyin_embedding_list,dim=1).to(device)
                        output_word_feat = self.word_attention(pinyin_embedding_list)
                        one_features = output_word_feat[0].unsqueeze(0)
                else:
                    one_word = self.word_node_emb(torch.tensor(vocab_dict[na]).to(device))
                    one_pos = self.pos_emb(42)
                    one_feat = one_word+one_pos
                    one_features.append(one_feat)
                    one_features = torch.stack(one_features)
                one_name_features_list.append(one_features)
            one_name_features_list = torch.stack(one_name_features_list)
            name_graph_features = torch.zeros((6,768))
            for i in range(len(one_name_features_list)):
                name_graph_features[i,:] = one_name_features_list[i]
            all_name_features.append(name_graph_features)
        all_name_features = torch.stack(all_name_features).to(device)
        all_name_features = self.relu(self.linear1(all_name_features))
        # ----------------------------------------------------------------------------- # 
        all_name_features = torch.cat([all_bert_name_features,all_name_features],dim=2)
        all_name_features = all_name_features.to(device)

        all_name_features=  self.relu(self.cat2encoder(all_name_features))
        output_name_features = self.bert_encoder(all_name_features)[0]
        output_name_features = self.pooler(output_name_features)
        output_name_features = self.mlp(output_name_features)
        # ----------------------------------------------------------------------------- # 
        return output_name_features