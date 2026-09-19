"""
MOTCS Extended: Generalized Multi-Omics Model Architecture
Supports arbitrary N-omics views (e.g., 3-omics, 4-omics, 5-omics).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset


def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class GeneTransformer(nn.Module):
    """
    Gene / Omics Transformer Feature Encoder
    Projects each feature value to d_model, performs Multi-Head Self-Attention,
    and returns flattened deep feature representations.
    Omits positional encoding (features are permutation-invariant).
    """
    def __init__(self, n_genes, dropout=0.1, d_model=128, dim_feedforward=256, nhead=4, num_layers=2):
        super(GeneTransformer, self).__init__()
        self.d_model = d_model
        self.n_genes = n_genes

        # Feature embedding: linear projection from 1D scalar to d_model
        self.gene_embedding = nn.Linear(1, d_model)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, src):
        # src shape: (batch_size, n_genes)
        src = src.unsqueeze(-1)                          # (batch_size, n_genes, 1)
        src = self.gene_embedding(src)                  # (batch_size, n_genes, d_model)
        src = src.permute(1, 0, 2)                      # (n_genes, batch_size, d_model)
        output = self.transformer_encoder(src)          # (n_genes, batch_size, d_model)
        output = output.permute(1, 0, 2).reshape(-1, self.n_genes * self.d_model) # (batch_size, n_genes * d_model)
        return output


class Classifier_1(nn.Module):
    """
    Per-Omics Multi-Layer Perceptron (MLP) Classifier
    Maps concatenated (driver + non-driver) embeddings to subtype logits.
    """
    def __init__(self, in_dim, out_dim, hidden_dims=[512, 256, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = in_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=True))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, out_dim, bias=True))
        self.clf = nn.Sequential(*layers)
        self.clf.apply(xavier_init)

    def forward(self, x):
        return self.clf(x)


class GeneralizedVCDN(nn.Module):
    """
    Generalized View Correlation Discovery Network (VCDN)
    Computes cross-view outer product tensor across any number of views (num_view >= 2),
    then fuses high-order interactions via fully-connected layers.
    """
    def __init__(self, num_view, num_cls, hvcdn_dim=None):
        super().__init__()
        self.num_view = num_view
        self.num_cls = num_cls
        in_dim = pow(num_cls, num_view)
        if hvcdn_dim is None:
            hvcdn_dim = min(in_dim, 512)

        self.model = nn.Sequential(
            nn.Linear(in_dim, hvcdn_dim),
            nn.LeakyReLU(0.25),
            nn.Dropout(0.1),
            nn.Linear(hvcdn_dim, num_cls)
        )
        self.model.apply(xavier_init)

    def forward(self, in_list):
        num_view = len(in_list)
        # Convert logits to class probabilities (using sigmoid as in original MOTCS)
        probs = [torch.sigmoid(logits) for logits in in_list]

        # Iteratively compute the cross-view outer product tensor
        # Start with view 0 and view 1
        x = torch.bmm(probs[0].unsqueeze(2), probs[1].unsqueeze(1)) # (B, c, c)
        x = x.view(-1, pow(self.num_cls, 2), 1)                     # (B, c^2, 1)

        # Accumulate remaining views (view 2, view 3, view 4, ...)
        for i in range(2, num_view):
            x = torch.bmm(x, probs[i].unsqueeze(1))                 # (B, c^i, c)
            x = x.view(-1, pow(self.num_cls, i + 1), 1)             # (B, c^(i+1), 1)

        vcdn_feat = x.view(-1, pow(self.num_cls, num_view))         # (B, c^num_view)
        return self.model(vcdn_feat)


class CancerTypeClassifier(nn.Module):
    """
    Dual-branch encoder: combines a GeneTransformer for non-driver features
    and a GeneTransformer for driver gene/feature features.
    """
    def __init__(self, n_genes, n_genes_dg, n_classes, dropout, d_model,
                 dim_feedforward, nhead=4, num_layers=2):
        super(CancerTypeClassifier, self).__init__()
        self.gene_transformer = GeneTransformer(
            n_genes=n_genes,
            dropout=dropout,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers
        )
        self.gene_transformer_dg = GeneTransformer(
            n_genes=n_genes_dg,
            dropout=dropout,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers
        )

    def forward(self, x, x_dg):
        features = self.gene_transformer(x)
        features_dg = self.gene_transformer_dg(x_dg)
        return torch.cat((features, features_dg), dim=1)


def init_model_dict_5omics(view_list, num_class, dim_fea_list, dim_dg_list,
                           d_model=128, dim_feedforward=256, dropout=0.1, nhead=4):
    """
    Initializes model dictionary for any subset of views.
    """
    model_dict = {}
    num_view = len(view_list)

    for i, v in enumerate(view_list):
        model_dict[f"E{v}"] = CancerTypeClassifier(
            n_genes=dim_fea_list[i],
            n_genes_dg=dim_dg_list[i],
            n_classes=num_class,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            nhead=nhead
        )
        model_dict[f"C{v}"] = Classifier_1(
            in_dim=(dim_fea_list[i] + dim_dg_list[i]) * d_model,
            out_dim=num_class,
            dropout=dropout
        )

    if num_view >= 2:
        model_dict["C"] = GeneralizedVCDN(num_view, num_class)

    return model_dict


def init_optim_5omics(view_list, model_dict, lr_e=1e-4, lr_c=1e-4):
    optim_dict = {}
    for v in view_list:
        optim_dict[f"C{v}"] = torch.optim.Adam(
            list(model_dict[f"E{v}"].parameters()) +
            list(model_dict[f"C{v}"].parameters()),
            lr=lr_e
        )
    if len(view_list) >= 2 and "C" in model_dict:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    return optim_dict


class MultiOmicsListDataset(Dataset):
    def __init__(self, lasso_features_list, dg_features_list, labels):
        self.lasso_features_list = lasso_features_list
        self.dg_features_list = dg_features_list
        self.labels = labels
        self.num_samples = len(labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        lasso_feats = [feat[idx] for feat in self.lasso_features_list]
        dg_feats = [feat[idx] for feat in self.dg_features_list]
        label = self.labels[idx]
        return lasso_feats, dg_feats, label, idx


def prepare_trte_data_5omics(data_fea_folder, data_dg_folder, view_list):
    """
    Dynamically loads and partitions arbitrary N-omics train/test data.
    """
    labels_tr = np.loadtxt(os.path.join(data_fea_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_fea_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)

    data_fea_tr_list = []
    data_fea_te_list = []
    data_dg_tr_list = []
    data_dg_te_list = []

    for v in view_list:
        data_fea_tr_list.append(np.loadtxt(os.path.join(data_fea_folder, f"{v}_tr.csv"), delimiter=','))
        data_fea_te_list.append(np.loadtxt(os.path.join(data_fea_folder, f"{v}_te.csv"), delimiter=','))
        data_dg_tr_list.append(np.loadtxt(os.path.join(data_dg_folder, f"{v}_tr.csv"), delimiter=','))
        data_dg_te_list.append(np.loadtxt(os.path.join(data_dg_folder, f"{v}_te.csv"), delimiter=','))

    num_tr = data_fea_tr_list[0].shape[0]
    num_te = data_fea_te_list[0].shape[0]
    idx_dict = {"tr": list(range(num_tr)), "te": list(range(num_tr, num_tr + num_te))}

    data_fea_all_list = []
    data_dg_all_list = []
    data_fea_train_list = []
    data_dg_train_list = []

    for i in range(len(view_list)):
        full_fea = np.concatenate((data_fea_tr_list[i], data_fea_te_list[i]), axis=0)
        full_dg = np.concatenate((data_dg_tr_list[i], data_dg_te_list[i]), axis=0)

        tensor_fea = torch.FloatTensor(full_fea)
        tensor_dg = torch.FloatTensor(full_dg)

        data_fea_all_list.append(tensor_fea)
        data_dg_all_list.append(tensor_dg)
        data_fea_train_list.append(tensor_fea[idx_dict["tr"]].clone())
        data_dg_train_list.append(tensor_dg[idx_dict["tr"]].clone())

    labels = np.concatenate((labels_tr, labels_te))
    return (data_fea_train_list, data_fea_all_list, data_dg_train_list, data_dg_all_list, idx_dict, labels)
