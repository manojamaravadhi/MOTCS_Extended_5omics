"""
MOTCS Extended: 5-Omics Training & Testing Pipeline
Supports arbitrary subset of views (e.g. --views 1 2 3 4 5)
Auto-detects Apple Silicon MPS, CUDA, or CPU.
"""

import os
import sys
import copy
import random
import datetime
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Import extended architecture and utilities
from model_Tran_5omics import (
    init_model_dict_5omics,
    init_optim_5omics,
    prepare_trte_data_5omics,
    MultiOmicsListDataset,
)
from utils import cal_sample_weight


def parse_args():
    parser = argparse.ArgumentParser(description='MOTCS Extended: 5-Omics Cancer Subtype Classification')
    parser.add_argument('-c', '--cancertype', '--cancer', type=str, default='KIPAN', help='Cancer cohort name (default: KIPAN)')
    parser.add_argument('--views', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                        help='Views to include (e.g. 1 2 3 4 5). 1:mRNA, 2:miRNA, 3:DNAmeth, 4:Prot, 5:Metab')
    parser.add_argument('-f', '--fold', type=int, default=1, help='Fold number (default: 1)')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of joint training epochs (default: 50)')
    parser.add_argument('-pe', '--pretrain_epochs', type=int, default=30, help='Number of pretrain epochs (default: 30)')
    parser.add_argument('-he', '--head', type=int, default=4, help='Attention heads (default: 4)')
    parser.add_argument('-dp', '--dropout', type=float, default=0.1, help='Dropout rate (default: 0.1)')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('-lr_e', '--learningrate_e', type=float, default=0.0001, help='Encoder learning rate')
    parser.add_argument('-lr_c', '--learningrate_c', type=float, default=0.0001, help='VCDN / classifier learning rate')
    parser.add_argument('-df', '--dim_feedforward', type=int, default=256, help='FFN hidden dim (default: 256)')
    parser.add_argument('-dm', '--d_model', type=int, default=64, help='Transformer d_model (default: 64)')
    parser.add_argument('-seed', '--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, mps, cuda, or cpu')
    parser.add_argument('--data_dir', type=str, default='', help='Custom data directory path')
    parser.add_argument('--save_dir', type=str, default='', help='Custom results directory path')
    return parser.parse_args()


def select_device(req_device='auto'):
    if req_device != 'auto':
        return torch.device(req_device)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


def train_epoch(model_dict, optim_dict, train_loader, sample_weight_tr, view_list, device, use_vcdn=True):
    for m in model_dict.values():
        m.train()

    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    loss_dict = {f"C{v}": 0.0 for v in view_list}
    if use_vcdn and len(view_list) >= 2:
        loss_dict["C"] = 0.0

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_lasso, batch_dg, targets, batch_indices in train_loader:
        batch_lasso = [x.to(device) for x in batch_lasso]
        batch_dg = [x.to(device) for x in batch_dg]
        targets = targets.to(device)
        weights = sample_weight_tr[batch_indices].to(device)

        ci_list = []

        # 1. Train per-view encoders and MLP heads
        for idx, v in enumerate(view_list):
            optim_dict[f"C{v}"].zero_grad()
            ei = model_dict[f"E{v}"](batch_lasso[idx], batch_dg[idx])
            ci = model_dict[f"C{v}"](ei)
            ci_list.append(ci.detach() if use_vcdn else ci)

            loss_i = torch.mean(criterion(ci, targets) * weights)
            loss_i.backward()
            optim_dict[f"C{v}"].step()

            loss_dict[f"C{v}"] += loss_i.item()

            if not use_vcdn:
                _, pred = ci.max(1)
                correct += pred.eq(targets).sum().item()
                total += targets.size(0)

        # 2. Train VCDN multi-omics fusion
        if use_vcdn and len(view_list) >= 2:
            optim_dict["C"].zero_grad()
            vcdn_out = model_dict["C"](ci_list)
            vcdn_loss = torch.mean(criterion(vcdn_out, targets) * weights)
            vcdn_loss.backward()
            optim_dict["C"].step()

            loss_dict["C"] += vcdn_loss.item()
            _, pred = vcdn_out.max(1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)

    avg_loss = sum(loss_dict.values()) / max(len(train_loader), 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc, loss_dict


def test_eval(model_dict, test_loader, view_list, device):
    for m in model_dict.values():
        m.eval()

    all_targets, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch_lasso, batch_dg, targets, _ in test_loader:
            batch_lasso = [x.to(device) for x in batch_lasso]
            batch_dg = [x.to(device) for x in batch_dg]
            targets = targets.to(device)

            ci_list = []
            for idx, v in enumerate(view_list):
                ei = model_dict[f"E{v}"](batch_lasso[idx], batch_dg[idx])
                ci = model_dict[f"C{v}"](ei)
                ci_list.append(ci)

            if len(view_list) >= 2 and "C" in model_dict:
                vcdn_out = model_dict["C"](ci_list)
                probs = torch.softmax(vcdn_out, dim=1)
                _, preds = torch.max(vcdn_out, 1)
            else:
                # Single-omics mode
                probs = torch.softmax(ci_list[0], dim=1)
                _, preds = torch.max(ci_list[0], 1)

            all_targets.append(targets.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    prec_w = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_w = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    return acc, f1_w, f1_m, prec_w, rec_w, y_pred, y_prob


def main(args):
    # Set random seeds for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = select_device(args.device)
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Using compute device: {device}")

    cancer = args.cancertype
    kf_num = args.fold
    view_list = sorted(args.views)
    num_view = len(view_list)
    print(f"Running MOTCS for {cancer} with {num_view} views: {view_list}")

    # Determine data paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    if args.data_dir:
        data_base = args.data_dir
    else:
        data_base = os.path.join(project_root, '1.data', cancer, 'MultiOmics')

    data_fea_folder = os.path.join(data_base, 'WODG', f"{kf_num}.KF")
    data_dg_folder = os.path.join(data_base, f"{kf_num}.KF", "没进行svcl1的dg")

    if not os.path.exists(data_fea_folder):
        raise FileNotFoundError(f"Missing non-driver data folder: {data_fea_folder}")
    if not os.path.exists(data_dg_folder):
        raise FileNotFoundError(f"Missing driver data folder: {data_dg_folder}")

    # Load data
    (data_fea_tr_list, data_fea_all_list,
     data_dg_tr_list, data_dg_all_list,
     trte_idx, labels_trte) = prepare_trte_data_5omics(data_fea_folder, data_dg_folder, view_list)

    num_class = len(np.unique(labels_trte))
    dim_fea_list = [x.shape[1] for x in data_fea_tr_list]
    dim_dg_list = [x.shape[1] for x in data_dg_tr_list]

    print(f"Classes: {num_class} | Train samples: {len(trte_idx['tr'])} | Test samples: {len(trte_idx['te'])}")
    for i, v in enumerate(view_list):
        print(f"  View {v}: Non-driver features={dim_fea_list[i]}, Driver features={dim_dg_list[i]}")

    # Datasets and Loaders
    train_dataset = MultiOmicsListDataset(
        lasso_features_list=data_fea_tr_list,
        dg_features_list=data_dg_tr_list,
        labels=labels_trte[trte_idx['tr']]
    )
    test_dataset = MultiOmicsListDataset(
        lasso_features_list=[data_fea_all_list[i][trte_idx['te']] for i in range(num_view)],
        dg_features_list=[data_dg_all_list[i][trte_idx['te']] for i in range(num_view)],
        labels=labels_trte[trte_idx['te']]
    )

    sample_weight_tr = torch.FloatTensor(cal_sample_weight(labels_trte[trte_idx["tr"]], num_class))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Output directory
    if args.save_dir:
        results_cwd = args.save_dir
    else:
        views_tag = f"views_{'_'.join(map(str, view_list))}"
        results_cwd = os.path.join(project_root, '3.results', cancer, f"{kf_num}.KF", views_tag)
    os.makedirs(results_cwd, exist_ok=True)

    # Initialize model and optimizers
    model_dict = init_model_dict_5omics(
        view_list=view_list,
        num_class=num_class,
        dim_fea_list=dim_fea_list,
        dim_dg_list=dim_dg_list,
        d_model=args.d_model,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        nhead=args.head
    )
    for m in model_dict.values():
        m.to(device)

    # --- STAGE 1: Pretraining encoders and per-view classifiers ---
    if args.pretrain_epochs > 0:
        print(f"\n[Stage 1] Pretraining encoders for {args.pretrain_epochs} epochs...")
        optim_pretrain = init_optim_5omics(view_list, model_dict, lr_e=args.learningrate_e, lr_c=args.learningrate_c)
        for ep in range(1, args.pretrain_epochs + 1):
            loss, acc, _ = train_epoch(model_dict, optim_pretrain, train_loader, sample_weight_tr,
                                       view_list, device, use_vcdn=False)
            if ep % 10 == 0 or ep == args.pretrain_epochs:
                print(f"  Pretrain Epoch {ep:3d}/{args.pretrain_epochs} - Loss: {loss:.4f}, Acc: {acc:.2f}%")

    # --- STAGE 2: Joint Training (Encoders + VCDN) ---
    print(f"\n[Stage 2] Joint Training with VCDN for {args.epochs} epochs...")
    optim_joint = init_optim_5omics(view_list, model_dict, lr_e=args.learningrate_e, lr_c=args.learningrate_c)

    best_acc = 0.0
    best_f1_w = 0.0
    best_f1_m = 0.0
    best_epoch = 0
    best_preds = None
    best_probs = None
    best_model_weights = None

    for ep in range(1, args.epochs + 1):
        loss, train_acc, _ = train_epoch(model_dict, optim_joint, train_loader, sample_weight_tr,
                                         view_list, device, use_vcdn=(num_view >= 2))
        test_acc, test_f1_w, test_f1_m, test_prec, test_rec, preds, probs = test_eval(
            model_dict, test_loader, view_list, device
        )

        if test_acc > best_acc or (test_acc == best_acc and test_f1_w > best_f1_w):
            best_acc = test_acc
            best_f1_w = test_f1_w
            best_f1_m = test_f1_m
            best_epoch = ep
            best_preds = preds
            best_probs = probs
            best_model_weights = {k: copy.deepcopy(v.state_dict()) for k, v in model_dict.items()}

        if ep % 10 == 0 or ep == args.epochs:
            print(f"  Epoch {ep:3d}/{args.epochs} - Loss: {loss:.4f} | Test Acc: {test_acc:.4f}, F1_w: {test_f1_w:.4f}, F1_m: {test_f1_m:.4f}")

    # Save best checkpoint
    best_model_path = os.path.join(results_cwd, "best_model_5omics.pth")
    if best_model_weights is not None:
        torch.save({
            'model_state_dict': best_model_weights,
            'epoch': best_epoch,
            'best_acc': best_acc,
            'best_f1_weighted': best_f1_w,
            'best_f1_macro': best_f1_m,
            'views': view_list,
            'classes': num_class,
            'd_model': args.d_model,
            'dim_feedforward': args.dim_feedforward,
            'head': args.head,
            'dropout': args.dropout
        }, best_model_path)
        for mod, sd in best_model_weights.items():
            torch.save(sd, os.path.join(results_cwd, f"{mod}.pth"))

    if best_preds is not None:
        np.savetxt(os.path.join(results_cwd, 'best_pred.csv'), best_preds, fmt='%d', delimiter=',')
        np.savetxt(os.path.join(results_cwd, 'best_probs.csv'), best_probs, fmt='%.8f', delimiter=',')

    print("\n" + "=" * 50)
    print(f"Best Results (Epoch {best_epoch}):")
    print(f"  ACC:         {best_acc:.4f}")
    print(f"  F1_Weighted: {best_f1_w:.4f}")
    print(f"  F1_Macro:    {best_f1_m:.4f}")
    print(f"Model saved to: {results_cwd}")
    print("=" * 50)
    return {
        'acc': best_acc,
        'f1_weighted': best_f1_w,
        'f1_macro': best_f1_m,
        'epoch': best_epoch,
        'views': view_list
    }


if __name__ == '__main__':
    args = parse_args()
    main(args)
