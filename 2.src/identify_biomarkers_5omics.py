"""
MOTCS Extended: Biomarker Identification Engine (Section 2.5 of paper)
Calculates feature importance by zeroing out individual features one-by-one,
measuring the drop in F1-score (delta F1), and extracting the Top-30 biomarkers
for all 5 omics levels (mRNA, miRNA, DNA methylation, Proteomics, Metabolomics).
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from model_Tran_5omics import (
    init_model_dict_5omics,
    prepare_trte_data_5omics,
    MultiOmicsListDataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Biomarker Recognition via Feature Perturbation')
    parser.add_argument('-c', '--cancertype', type=str, default='KIPAN')
    parser.add_argument('--views', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('-f', '--fold', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=30, help='Number of top biomarkers to extract per view')
    parser.add_argument('--model_dir', type=str, default='', help='Directory with trained model checkpoints')
    parser.add_argument('--device', type=str, default='auto')
    return parser.parse_args()


def select_device(req_device='auto'):
    if req_device != 'auto':
        return torch.device(req_device)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


def eval_model(model_dict, test_loader, view_list, device):
    for m in model_dict.values():
        m.eval()
    all_targets, all_preds = [], []
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
                _, preds = torch.max(vcdn_out, 1)
            else:
                _, preds = torch.max(ci_list[0], 1)

            all_targets.append(targets.cpu())
            all_preds.append(preds.cpu())

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    return f1_score(y_true, y_pred, average='weighted', zero_division=0)


def load_feature_names(folder, view_num):
    fname_file = os.path.join(folder, f"{view_num}_featname.csv")
    if os.path.exists(fname_file):
        df = pd.read_csv(fname_file, header=None)
        return df.iloc[:, 0].astype(str).tolist()
    return []


def main(args):
    device = select_device(args.device)
    cancer = args.cancertype
    kf_num = args.fold
    view_list = sorted(args.views)
    num_view = len(view_list)
    top_k = args.top_k

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_base = os.path.join(project_root, '1.data', cancer, 'MultiOmics')
    data_fea_folder = os.path.join(data_base, 'WODG', f"{kf_num}.KF")
    data_dg_folder = os.path.join(data_base, f"{kf_num}.KF", "没进行svcl1的dg")

    if args.model_dir:
        results_dir = args.model_dir
    else:
        views_tag = f"views_{'_'.join(map(str, view_list))}"
        results_dir = os.path.join(project_root, '3.results', cancer, f"{kf_num}.KF", views_tag)

    checkpoint_path = os.path.join(results_dir, "best_model_5omics.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Trained checkpoint not found at {checkpoint_path}. Train the model first.")

    # Load data
    (data_fea_tr_list, data_fea_all_list,
     data_dg_tr_list, data_dg_all_list,
     trte_idx, labels_trte) = prepare_trte_data_5omics(data_fea_folder, data_dg_folder, view_list)

    num_class = len(np.unique(labels_trte))
    dim_fea_list = [x.shape[1] for x in data_fea_tr_list]
    dim_dg_list = [x.shape[1] for x in data_dg_tr_list]

    checkpoint = torch.load(checkpoint_path, map_location=device)
    d_model = checkpoint.get('d_model', 64)
    dim_feedforward = checkpoint.get('dim_feedforward', 256)
    nhead = checkpoint.get('head', 4)
    dropout = checkpoint.get('dropout', 0.1)

    # Initialize model
    model_dict = init_model_dict_5omics(
        view_list=view_list,
        num_class=num_class,
        dim_fea_list=dim_fea_list,
        dim_dg_list=dim_dg_list,
        d_model=d_model,
        dim_feedforward=dim_feedforward,
        nhead=nhead,
        dropout=dropout
    )
    for k, v in model_dict.items():
        v.load_state_dict(checkpoint['model_state_dict'][k])
        v.to(device)

    # Base test data
    te_idx = trte_idx["te"]
    labels_te = labels_trte[te_idx]

    base_test_dataset = MultiOmicsListDataset(
        lasso_features_list=[data_fea_all_list[i][te_idx].clone() for i in range(num_view)],
        dg_features_list=[data_dg_all_list[i][te_idx].clone() for i in range(num_view)],
        labels=labels_te
    )
    base_loader = torch.utils.data.DataLoader(base_test_dataset, batch_size=128, shuffle=False)
    baseline_f1 = eval_model(model_dict, base_loader, view_list, device)
    print(f"=== Baseline F1 (Weighted) on Test Set: {baseline_f1:.4f} ===\n")

    view_names = {
        1: "mRNA expression",
        2: "miRNA expression",
        3: "DNA methylation",
        4: "Proteomics",
        5: "Metabolomics"
    }

    all_biomarkers = {}

    for idx, v in enumerate(view_list):
        v_name = view_names.get(v, f"View {v}")
        print(f"--- Computing Feature Importance for View {v}: {v_name} ---")

        ndv_names = load_feature_names(data_fea_folder, v)
        dg_names = load_feature_names(data_dg_folder, v)

        n_ndv = data_fea_all_list[idx].shape[1]
        n_dg = data_dg_all_list[idx].shape[1]

        if not ndv_names:
            ndv_names = [f"V{v}_NDV_{i}" for i in range(n_ndv)]
        if not dg_names:
            dg_names = [f"V{v}_DG_{i}" for i in range(n_dg)]

        feature_scores = []

        # 1. Perturb non-driver features
        for f_idx in range(n_ndv):
            perturbed_lasso = [data_fea_all_list[i][te_idx].clone() for i in range(num_view)]
            perturbed_lasso[idx][:, f_idx] = 0.0  # Zero-out feature

            dataset = MultiOmicsListDataset(
                lasso_features_list=perturbed_lasso,
                dg_features_list=[data_dg_all_list[i][te_idx].clone() for i in range(num_view)],
                labels=labels_te
            )
            loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
            score = eval_model(model_dict, loader, view_list, device)
            delta_f1 = baseline_f1 - score
            feature_scores.append({
                'feature': ndv_names[f_idx] if f_idx < len(ndv_names) else f"NDV_{f_idx}",
                'type': 'non-driver',
                'delta_f1': delta_f1
            })

        # 2. Perturb driver features
        for f_idx in range(n_dg):
            perturbed_dg = [data_dg_all_list[i][te_idx].clone() for i in range(num_view)]
            perturbed_dg[idx][:, f_idx] = 0.0  # Zero-out driver feature

            dataset = MultiOmicsListDataset(
                lasso_features_list=[data_fea_all_list[i][te_idx].clone() for i in range(num_view)],
                dg_features_list=perturbed_dg,
                labels=labels_te
            )
            loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
            score = eval_model(model_dict, loader, view_list, device)
            delta_f1 = baseline_f1 - score
            feature_scores.append({
                'feature': dg_names[f_idx] if f_idx < len(dg_names) else f"DG_{f_idx}",
                'type': 'driver',
                'delta_f1': delta_f1
            })

        # Sort descending by delta_f1
        df_scores = pd.DataFrame(feature_scores).sort_values(by='delta_f1', ascending=False)
        top_df = df_scores.head(top_k)
        all_biomarkers[v] = top_df

        # Save view biomarkers to CSV
        out_csv = os.path.join(results_dir, f"biomarkers_view_{v}_{v_name.replace(' ', '_')}.csv")
        top_df.to_csv(out_csv, index=False)
        print(f"Saved Top-{top_k} biomarkers to: {out_csv}")
        print(top_df[['feature', 'type', 'delta_f1']].head(10).to_string(index=False))
        print("\n" + "-" * 50 + "\n")

    print(f"Biomarker recognition completed for all {num_view} omics views!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
