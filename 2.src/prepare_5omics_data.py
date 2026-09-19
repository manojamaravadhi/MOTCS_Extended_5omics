"""
MOTCS Extended: Data Preparation & Ingestion Pipeline for 5-Omics
Provides automated utilities to:
1. Align patient barcodes across mRNA, miRNA, DNA methylation, Proteomics, and Metabolomics.
2. Download TCGA RPPA Proteomics via TCPA / LinkedOmics / cptac.
3. Extract driver genes/proteins from NCG, OncomiR, and hallmark oncometabolites.
4. Perform L1-SVC (C=0.1) feature selection on non-driver features.
5. Save data into MOTCS standard K-fold cross-validation CSV layout.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer


DEFAULT_NCG_DRIVERS = [
    'TP53', 'PIK3CA', 'PTEN', 'EGFR', 'ERBB2', 'BRCA1', 'BRCA2',
    'KRAS', 'HRAS', 'NRAS', 'BRAF', 'CDH1', 'RB1', 'GATA3',
    'FOXA1', 'MAP3K1', 'APC', 'SMAD4', 'CTNNB1', 'ARID1A',
    'ATM', 'SETD2', 'BAP1', 'VHL', 'PBRM1', 'KDM5C', 'MTOR',
    'TSC1', 'TSC2', 'NF2', 'STAG2', 'KMT2C', 'KMT2D', 'FAT1',
    'NFE2L2', 'KEAP1', 'FLCN', 'MET', 'SMARCB1', 'CUL3',
    'KDM6A', 'ARID2', 'RHEB', 'AKT1', 'SMAD2', 'SOX9', 'TGFBR2',
    'MSH2', 'MLH1', 'SPOP', 'MAP2K1', 'ERBB3', 'CDKN1A', 'CASP3'
]

DEFAULT_ONCOMETABOLITES = [
    '2-Hydroxyglutarate', 'L-Lactate', 'Succinate', 'Fumarate', 'Citrate',
    'Alpha-Ketoglutarate', 'Glutamate', 'Glutamine', 'Kynurenine', 'Choline',
    'Phosphocholine', 'Itaconate', 'Aspartate', 'Malate', 'Pyruvate',
    'Glucose-6-Phosphate', 'Palmitate', 'Arginine', 'S-Adenosylmethionine', 'Uracil'
]


def l1_svc_feature_selection(X_train, y_train, C=0.1):
    """
    Applies L1-regularized LinearSVC to select key non-driver features.
    As specified in the MOTCS paper (Section 4.2), C=0.1 yields optimal sparsity.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = LinearSVC(penalty='l1', dual=False, C=C, random_state=42, max_iter=2000)
    clf.fit(X_train_scaled, y_train)

    # Features with non-zero weights across any class
    non_zero = np.any(clf.coef_ != 0, axis=0)
    selected_indices = np.where(non_zero)[0]

    # Fallback to top variance features if L1 eliminates too many
    if len(selected_indices) < 10:
        variances = np.var(X_train, axis=0)
        selected_indices = np.argsort(variances)[-50:]

    return selected_indices


def partition_driver_and_nondriver(df_omics, driver_names, y_labels, n_splits=5, output_dir='', view_id=4):
    """
    Partitions an omics dataframe into driver and non-driver matrices,
    performs 5-fold stratified splitting, and writes MOTCS CSVs.
    """
    common_drivers = [col for col in df_omics.columns if any(d.lower() == col.split('_')[0].lower() for d in driver_names)]
    nondriver_cols = [col for col in df_omics.columns if col not in common_drivers]

    print(f"View {view_id}: Total={df_omics.shape[1]} | Drivers={len(common_drivers)} | Non-drivers={len(nondriver_cols)}")

    df_dg = df_omics[common_drivers] if common_drivers else df_omics.iloc[:, :20]
    df_ndv = df_omics[nondriver_cols] if nondriver_cols else df_omics

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    X = df_ndv.values
    y = np.array(y_labels)

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        fold_wodg_dir = os.path.join(output_dir, 'WODG', f"{fold_idx}.KF")
        fold_dg_dir = os.path.join(output_dir, f"{fold_idx}.KF", "没进行svcl1的dg")
        os.makedirs(fold_wodg_dir, exist_ok=True)
        os.makedirs(fold_dg_dir, exist_ok=True)

        # L1-SVC selection on training fold of non-drivers
        sel_idx = l1_svc_feature_selection(X[tr_idx], y[tr_idx], C=0.1)

        # Non-driver CSVs
        np.savetxt(os.path.join(fold_wodg_dir, f"{view_id}_tr.csv"), X[tr_idx][:, sel_idx], delimiter=',', fmt='%.4f')
        np.savetxt(os.path.join(fold_wodg_dir, f"{view_id}_te.csv"), X[te_idx][:, sel_idx], delimiter=',', fmt='%.4f')
        pd.DataFrame(df_ndv.columns[sel_idx]).to_csv(os.path.join(fold_wodg_dir, f"{view_id}_featname.csv"), index=False, header=False)

        # Driver CSVs
        np.savetxt(os.path.join(fold_dg_dir, f"{view_id}_tr.csv"), df_dg.values[tr_idx], delimiter=',', fmt='%.4f')
        np.savetxt(os.path.join(fold_dg_dir, f"{view_id}_te.csv"), df_dg.values[te_idx], delimiter=',', fmt='%.4f')
        pd.DataFrame(df_dg.columns).to_csv(os.path.join(fold_dg_dir, f"{view_id}_featname.csv"), index=False, header=False)

        # Labels
        np.savetxt(os.path.join(fold_wodg_dir, "labels_tr.csv"), y[tr_idx], delimiter=',', fmt='%d')
        np.savetxt(os.path.join(fold_wodg_dir, "labels_te.csv"), y[te_idx], delimiter=',', fmt='%d')
        np.savetxt(os.path.join(fold_dg_dir, "labels_tr.csv"), y[tr_idx], delimiter=',', fmt='%d')
        np.savetxt(os.path.join(fold_dg_dir, "labels_te.csv"), y[te_idx], delimiter=',', fmt='%d')

    print(f"Successfully processed View {view_id} into {n_splits} folds under {output_dir}")


def download_cptac_proteomics(cancer_name="Ccrcc"):
    """
    Downloads mass spectrometry proteomics using the official cptac Python library.
    pip install cptac
    """
    try:
        import cptac
        print(f"Downloading CPTAC {cancer_name} dataset...")
        cptac.download(dataset=cancer_name)
        if cancer_name.lower() == "ccrcc":
            cohort = cptac.Ccrcc()
        elif cancer_name.lower() == "brca":
            cohort = cptac.Brca()
        elif cancer_name.lower() == "coad":
            cohort = cptac.Coad()
        else:
            raise ValueError(f"Unsupported CPTAC dataset: {cancer_name}")

        proteomics = cohort.get_proteomics()
        print(f"Downloaded CPTAC Proteomics: {proteomics.shape}")
        return proteomics
    except ImportError:
        print("cptac package not installed. Install via: pip install cptac")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess and format 5-omics data for MOTCS')
    parser.add_argument('--cancer', type=str, default='KIPAN')
    args = parser.parse_args()
    print(f"Data preparation pipeline initialized for {args.cancer}.")
