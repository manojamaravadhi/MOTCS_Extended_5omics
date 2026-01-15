""" Transformer-based 模型组件 """
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from sklearn import preprocessing

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


# 基因Transformer模型主类
class GeneTransformer(nn.Module):
    """
    基因表达分类的Transformer模型
    参数:
        n_genes: 基因数量（输入特征维度）
        #n_classes: 分类类别数（癌症亚型数量）
        d_model: 嵌入维度（默认128）
        nhead: 注意力头数（默认8）
        dim_feedforward: 前馈网络隐藏层维度（默认512）
        num_layers: Transformer编码器层数（默认6）
        dropout: dropout概率（默认0.1）
    """
    def __init__(self, n_genes,  dropout, d_model, dim_feedforward, nhead=8,
                 num_layers=2):
        super(GeneTransformer, self).__init__()
        
        # 基因嵌入层：将每个基因表达值映射到高维空间
        self.gene_embedding = nn.Linear(1, d_model)  # 每个基因值作为单独特征
        
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout)  # 添加这个参数)  # 单层定义
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers) 
        
        # 递归初始化所有参数
        self.apply(self._init_weights)
        
        # 保存模型参数
        self.d_model = d_model
        self.n_genes = n_genes
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        
            
    def forward(self, src):
        """
        前向传播
        参数:
            src: 输入基因表达数据，形状(batch_size, n_genes)
        返回:
            分类概率，形状(batch_size, n_classes)
        """
        # 1. 基因嵌入
        # 调整形状：(batch_size, n_genes, 1)
        src = src.unsqueeze(-1)
        # 嵌入到高维空间：(batch_size, n_genes, d_model)
        src = self.gene_embedding(src)
        
        # 2. 调整形状以适应Transformer：(n_genes, batch_size, d_model)
        src = src.permute(1, 0, 2)
        
        # 4. Transformer编码
        output = self.transformer_encoder(src)
        
        # 5. 展平输出：(batch_size, n_genes * d_model)
        output = output.permute(1, 0, 2).reshape(-1, self.n_genes * self.d_model)
        
        # 6. 分类层
        #output = self.classifier(output)
        
        # 返回log softmax概率（适用于NLLLoss）
        #return F.log_softmax(output, dim=-1)
        return output

class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=[512, 256, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = in_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=True))
            layers.append(nn.BatchNorm1d(hidden_dim))  # 添加BatchNorm
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, out_dim, bias=True))
        
        self.clf = nn.Sequential(*layers)
        self.clf.apply(xavier_init)

    def forward(self, x):
        return self.clf(x)



# 替换为你提供的标准VCDN类
class VCDN(nn.Module):
    def __init__(self, num_view, num_cls, hvcdn_dim):
        super().__init__()
        self.num_cls = num_cls
        self.model = nn.Sequential(
            nn.Linear(pow(num_cls, num_view), hvcdn_dim),
            nn.LeakyReLU(0.25),
            nn.Linear(hvcdn_dim, num_cls)
        )
        self.model.apply(xavier_init)
        
    def forward(self, in_list):
        num_view = len(in_list)
        # 1. logits转概率（sigmoid激活）
        for i in range(num_view):
            in_list[i] = torch.sigmoid(in_list[i])
        # 2. 计算前2个视图的笛卡尔积
        x = torch.reshape(
            torch.matmul(in_list[0].unsqueeze(-1), in_list[1].unsqueeze(1)),
            (-1, pow(self.num_cls, 2), 1)
        )
        # 3. 迭代计算后续视图的笛卡尔积（适配3视图）
        for i in range(2, num_view):
            x = torch.reshape(
                torch.matmul(x, in_list[i].unsqueeze(1)),
                (-1, pow(self.num_cls, i+1), 1)
            )
        # 4. 重塑为二维特征（适配全连接输入）
        vcdn_feat = torch.reshape(x, (-1, pow(self.num_cls, num_view)))
        # 5. 全连接网络输出logits
        output = self.model(vcdn_feat)
        return output        
        
class CancerTypeClassifier(nn.Module):
    def __init__(self, n_genes,n_genes_dg, n_classes, dropout, d_model,
                 dim_feedforward, nhead=8,  num_layers=2):
        """
        使用 GeneTransformer + MLP 的癌症分类器
        
        参数:
            n_genes: 基因数量（输入特征维度）
            n_classes: 分类类别数（癌症亚型数量）
            dropout: dropout概率
            d_model: 嵌入维度（默认128）
            nhead: 注意力头数（默认8）
            dim_feedforward: 前馈网络隐藏层维度（默认512）
            num_layers: Transformer编码器层数（默认6）
        """
        super(CancerTypeClassifier, self).__init__()
        
        # 基因Transformer模块
        self.gene_transformer = GeneTransformer(
            n_genes=n_genes,
            dropout=dropout,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers
        )
        
        # 基因Transformer模块  驱动基因
        self.gene_transformer_dg = GeneTransformer(
            n_genes=n_genes_dg,
            dropout=dropout,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers
        )
        
        # 使用简单的MLP分类器 (输入维度 = n_genes * d_model)
        #self.mlp = MLP(nfeat=(n_genes + n_genes_dg) * d_model, output_dim=n_classes)
        
    def forward(self, x, x_dg):
        """
        前向传播
        
        参数:
            x: 输入张量，形状 (batch_size, n_genes)
            x_dg: 输入驱动基因特征的张量，形状（batch_size, n_genes)
            
        返回:
            分类的对数概率，形状 (batch_size, n_classes)
        """
        # 1. 通过Transformer获取特征
        features = self.gene_transformer(x)  # 输出形状: (batch_size, n_genes * d_model)
        features_dg = self.gene_transformer_dg(x_dg)
        total_features = torch.cat((features, features_dg), dim=1)
        
        
        return total_features
    
    
    
    
    
    
    
    
    
    
def init_model_dict1(num_view, num_class, dim_fea_list,dim_dg_list, d_model,
                   dim_feedforward, dropout,nhead, dim_hc=None):
    model_dict = {}
    for i in range(num_view):
        model_dict[f"E{i+1}"] = CancerTypeClassifier(
            n_genes = dim_fea_list[i],
            n_genes_dg = dim_dg_list[i],
            n_classes = num_class,
            d_model = d_model,  
            dim_feedforward = dim_feedforward,    
            dropout = dropout,
            nhead = nhead
        )
        model_dict[f"C{i+1}"] = Classifier_1((dim_fea_list[i]+dim_dg_list[i])*d_model, num_class)
    
    if num_view >= 2:
        dim_hc = pow(num_class, num_view)
        model_dict["C"] = VCDN(num_view, num_class, dim_hc)
    
    return model_dict        

def init_optim(num_view, model_dict, lr_e=1e-4, lr_c=1e-4):
    optim_dict = {}
    for i in range(num_view):
        optim_dict[f"C{i+1}"] = torch.optim.Adam(
            list(model_dict[f"E{i+1}"].parameters()) + 
            list(model_dict[f"C{i+1}"].parameters()),
            lr=lr_e
        )
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    return optim_dict

class MultiOmicsListDataset(Dataset):
    def __init__(self, lasso_features_list, dg_features_list, labels):
        self.lasso_features_list = lasso_features_list  # 多视图特征列表
        self.dg_features_list = dg_features_list
        self.labels = labels
        self.num_samples = len(labels)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 返回：视图1特征, 视图2特征, 视图3特征, 标签, 样本索引
        lasso_feats = [feat[idx] for feat in self.lasso_features_list]
        dg_feats = [feat[idx] for feat in self.dg_features_list]
        label = self.labels[idx]
        return lasso_feats, dg_feats, label, idx  # 新增返回 idx
    
def prepare_trte_data(data_fea_folder,data_dg_folder, view_list):
    labels_tr = np.loadtxt(os.path.join(data_fea_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_fea_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_fea_tr_list = []
    data_fea_te_list = []
    num_view = len(view_list)
    for i in view_list:
        data_fea_tr_list.append(np.loadtxt(os.path.join(data_fea_folder, str(i)+"_tr.csv"), delimiter=','))
        data_fea_te_list.append(np.loadtxt(os.path.join(data_fea_folder, str(i)+"_te.csv"), delimiter=','))
    num_fea_tr = data_fea_tr_list[0].shape[0]
    num_fea_te = data_fea_te_list[0].shape[0]
    data_dg_tr_list = []
    data_dg_te_list = []
    for i in view_list:
        data_dg_tr_list.append(np.loadtxt(os.path.join(data_dg_folder, str(i)+"_tr.csv"), delimiter=','))
        data_dg_te_list.append(np.loadtxt(os.path.join(data_dg_folder, str(i)+"_te.csv"), delimiter=','))
    num_dg_tr = data_dg_tr_list[0].shape[0]
    num_dg_te = data_dg_te_list[0].shape[0]
    
    
    #将多个视图（views）的训练数据和测试数据在垂直方向（行方向）上进行合并，然后将每个视图合并后的数据存储在一个列表中
    data_fea_mat_list = []
    for i in range(num_view):
        data_fea_mat_list.append(np.concatenate((data_fea_tr_list[i], data_fea_te_list[i]), axis=0))
    #将多个视图（views）的训练数据和测试数据在垂直方向（行方向）上进行合并，然后将每个视图合并后的数据存储在一个列表中
    data_dg_mat_list = []
    for i in range(num_view):
        data_dg_mat_list.append(np.concatenate((data_dg_tr_list[i], data_dg_te_list[i]), axis=0))
    data_fea_tensor_list = []
    for i in range(len(data_fea_mat_list)):
        data_fea_tensor_list.append(torch.FloatTensor(data_fea_mat_list[i]))
    
    data_dg_tensor_list = []
    for i in range(len(data_dg_mat_list)):
        data_dg_tensor_list.append(torch.FloatTensor(data_dg_mat_list[i]))
    num_tr = data_fea_tr_list[0].shape[0]
    num_te = data_fea_te_list[0].shape[0]
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))
    data_fea_train_list = []
    data_fea_all_list = []
    for i in range(len(data_fea_tensor_list)):
        data_fea_train_list.append(data_fea_tensor_list[i][idx_dict["tr"]].clone())
        data_fea_all_list.append(torch.cat((data_fea_tensor_list[i][idx_dict["tr"]].clone(),
                                       data_fea_tensor_list[i][idx_dict["te"]].clone()),0))
    data_dg_train_list = []
    data_dg_all_list = []
    for i in range(len(data_dg_tensor_list)):
        data_dg_train_list.append(data_dg_tensor_list[i][idx_dict["tr"]].clone())
        data_dg_all_list.append(torch.cat((data_dg_tensor_list[i][idx_dict["tr"]].clone(),
                                       data_dg_tensor_list[i][idx_dict["te"]].clone()),0))
    labels = np.concatenate((labels_tr, labels_te))
    
    return (data_fea_train_list, data_fea_all_list, data_dg_train_list, data_dg_all_list, idx_dict, labels)

