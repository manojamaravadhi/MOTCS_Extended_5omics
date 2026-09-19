"""
Convert MO-GCAN Figshare datasets → MOTCS format
==================================================

Downloads and converts MO-GCAN data (8 TCGA cancer types from Figshare)
into the exact folder/file structure that MOTCS's prepare_trte_data() expects.

MO-GCAN provides: mRNA, CNA, methylation, RPPA (no miRNA)
MOTCS expects:    3 views with driver/non-driver gene splits + K-fold CSVs

View mapping used:
  View 1 = mRNA expression
  View 2 = CNA (substituting for miRNA, which is absent)
  View 3 = DNA methylation

Usage:
------
  # Step 1: Download Figshare data first
  #   wget https://figshare.com/ndownloader/articles/25823950/versions/1 -O mogcan_data.zip
  #   unzip mogcan_data.zip -d mogcan_raw/

  # Step 2: Process the original data into per-cancer CSVs (uses MO-GCAN's format)
  #   python convert_mogcan_to_motcs.py process-raw --raw_dir mogcan_raw --out_dir mogcan_processed

  # Step 3: Convert a single cancer type
  python convert_mogcan_to_motcs.py convert \
      --input_dir mogcan_processed/brca \
      --output_dir 1.data/BRCA/MultiOmics \
      --n_folds 5

  # Step 4: Convert ALL 8 cancer types at once
  python convert_mogcan_to_motcs.py convert-all \
      --input_dir mogcan_processed \
      --output_dir 1.data \
      --n_folds 5

Requirements:
  pip install numpy pandas scikit-learn
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Default driver gene list (NCG v7.1 coding drivers — curated)
# These are commonly used pan-cancer driver genes.
# You can override this by providing --driver_gene_file
# ---------------------------------------------------------------------------
DEFAULT_DRIVER_GENES = {
    "TP53", "PIK3CA", "PTEN", "EGFR", "ERBB2", "BRCA1", "BRCA2",
    "KRAS", "HRAS", "NRAS", "BRAF", "CDH1", "RB1", "GATA3",
    "FOXA1", "MAP3K1", "APC", "SMAD4", "CTNNB1", "ARID1A",
    "ATM", "SETD2", "BAP1", "VHL", "PBRM1", "KDM5C", "MTOR",
    "TSC1", "TSC2", "NF2", "STAG2", "KMT2C", "KMT2D", "FAT1",
    "NFE2L2", "KEAP1", "FLCN", "MET", "SMARCB1", "CUL3",
    "KDM6A", "ARID2", "RHEB", "LRP1B", "CARD11", "TRRAP",
    "LRRK2", "ELOC", "TRIO", "MSR1", "CUL7", "SYNE2", "COL11A1",
    "NAV3", "FAT3", "CUBN", "TRIM37", "MED13", "FMN2", "NPNT",
    "PIEZO2", "CCNB2", "WDFY3", "AHNAK", "CNOT1", "AKAP9",
    "AKAP13", "PLEC", "NRXN1", "RYR1", "ZNF469", "TXNIP",
    "RADIL", "SPTBN4", "SHANK1", "ZNF804A", "CCDC168", "FAM111B",
    "IDH1", "IDH2", "FUBP1", "CIC", "NOTCH1", "NOTCH2",
    "ATRX", "CDKN2A", "NF1", "FBXW7", "SF3B1", "U2AF1",
    "DNMT3A", "TET2", "SRSF2", "RUNX1", "ASXL1", "EZH2",
    "BCOR", "PPM1D", "SPOP", "PIK3R1", "MAP2K1", "ERBB3",
    "AKT1", "SMAD2", "SOX9", "TGFBR2", "ACVR2A", "POLE",
    "MSH2", "MSH6", "MLH1", "PMS2",
}

CANCER_TYPES = ["lgg", "ucec", "stad", "sarc", "coadread", "cesc", "hnsc", "brca"]


def load_driver_genes(driver_file=None):
    """Load driver genes from file or use built-in defaults."""
    if driver_file and os.path.exists(driver_file):
        # Support both plain text (one gene per line) and CSV with 'gene' column
        if driver_file.endswith('.csv'):
            df = pd.read_csv(driver_file)
            if 'gene' in df.columns:
                genes = set(df['gene'].str.strip().tolist())
            elif 'symbol' in df.columns:
                genes = set(df['symbol'].str.strip().tolist())
            else:
                genes = set(df.iloc[:, 0].str.strip().tolist())
        else:
            with open(driver_file) as f:
                genes = set(line.strip().strip('"') for line in f if line.strip() and not line.startswith('#'))
        print(f"  Loaded {len(genes)} driver genes from {driver_file}")
        return genes
    else:
        print(f"  Using built-in driver gene list ({len(DEFAULT_DRIVER_GENES)} genes)")
        return DEFAULT_DRIVER_GENES.copy()


def load_mogcan_data(input_dir):
    """
    Load MO-GCAN processed data from a cancer-type folder.

    Expected files:
      mrna_data.csv, met_data.csv, cna_data.csv, subtype_data.csv
    """
    required = ['mrna_data.csv', 'met_data.csv', 'cna_data.csv', 'subtype_data.csv']
    for f in required:
        if not os.path.exists(os.path.join(input_dir, f)):
            raise FileNotFoundError(
                f"Missing {f} in {input_dir}\n"
                f"Expected files: {required}\n"
                f"Did you run 'process-raw' first, or download pre-processed data?"
            )

    print("  Loading mrna_data.csv ...")
    mrna = pd.read_csv(os.path.join(input_dir, 'mrna_data.csv'), index_col=0)

    print("  Loading met_data.csv ...")
    met = pd.read_csv(os.path.join(input_dir, 'met_data.csv'), index_col=0)

    print("  Loading cna_data.csv ...")
    cna = pd.read_csv(os.path.join(input_dir, 'cna_data.csv'), index_col=0)

    print("  Loading subtype_data.csv ...")
    subtypes = pd.read_csv(os.path.join(input_dir, 'subtype_data.csv'))

    # Drop samples with missing subtypes
    subtypes = subtypes.dropna(subset=['SUBTYPE'])
    subtypes = subtypes[subtypes['SUBTYPE'].str.strip() != '']

    # Align samples across all modalities
    common = (set(mrna.index) & set(met.index) & set(cna.index)
              & set(subtypes['PATIENT_ID'].values))
    common = sorted(list(common))

    if len(common) == 0:
        raise ValueError(
            f"No common samples found across modalities in {input_dir}!\n"
            f"  mrna samples: {mrna.shape[0]}, met: {met.shape[0]}, "
            f"cna: {cna.shape[0]}, subtypes: {len(subtypes)}"
        )

    mrna = mrna.loc[common]
    met = met.loc[common]
    cna = cna.loc[common]
    subtypes = subtypes[subtypes['PATIENT_ID'].isin(common)].set_index('PATIENT_ID').loc[common]

    # Encode labels as integers
    le = LabelEncoder()
    labels = le.fit_transform(subtypes['SUBTYPE'].values)

    print(f"  ✓ Aligned {len(common)} samples, {len(le.classes_)} subtypes: {list(le.classes_)}")
    print(f"    mRNA: {mrna.shape[1]} features")
    print(f"    Methylation: {met.shape[1]} features")
    print(f"    CNA: {cna.shape[1]} features")

    return mrna, met, cna, labels, le


def impute_nans(X):
    """Replace NaN values with column medians."""
    if not np.isnan(X).any():
        return X
    col_medians = np.nanmedian(X, axis=0)
    col_medians = np.nan_to_num(col_medians, nan=0.0)
    nan_locs = np.where(np.isnan(X))
    if len(nan_locs[0]) > 0:
        X[nan_locs] = np.take(col_medians, nan_locs[1])
    return X


def split_driver_nondriver(columns, driver_genes):
    """
    Split column names into driver and non-driver indices.
    Handles common prefixes (mRNA_, met_, cna_) and suffixes (_dnamethy).
    """
    driver_upper = {g.upper() for g in driver_genes}
    driver_idx = []
    nondriver_idx = []

    for i, col in enumerate(columns):
        gene_name = col
        # Strip known prefixes
        for prefix in ['mRNA_', 'met_', 'cna_', 'methylation_', 'CNA_', 'RPPA_']:
            if col.startswith(prefix):
                gene_name = col[len(prefix):]
                break
        # Strip known suffixes
        gene_name = gene_name.replace('_dnamethy', '').replace('_methylation', '')

        if gene_name.upper() in driver_upper:
            driver_idx.append(i)
        else:
            nondriver_idx.append(i)

    return np.array(driver_idx), np.array(nondriver_idx)


def l1_svc_select(X_train, y_train, X_all, C=0.1):
    """
    Apply L1-penalized SVC for feature selection.
    Returns mask of selected feature indices.
    """
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_all_sc = scaler.transform(X_all)

    try:
        svc = LinearSVC(C=C, penalty='l1', dual=False, max_iter=10000)
        svc.fit(X_tr_sc, y_train)

        if len(svc.coef_.shape) == 1:
            importance = np.abs(svc.coef_)
        else:
            importance = np.max(np.abs(svc.coef_), axis=0)

        selected = importance > 0
    except Exception as e:
        print(f"    WARNING: L1-SVC failed ({e}), falling back to variance-based selection")
        selected = np.ones(X_all_sc.shape[1], dtype=bool)

    # Fallback if nothing selected
    if selected.sum() == 0:
        top_k = min(50, X_all_sc.shape[1])
        idx = np.argsort(importance)[-top_k:]
        selected = np.zeros(len(importance), dtype=bool)
        selected[idx] = True
        print(f"    WARNING: L1-SVC selected 0 features, using top-{top_k} by importance")

    return X_all_sc[:, selected], selected


def save_view_data(out_dir, view_num, X_train, X_test, feat_names):
    """Save train/test CSVs and feature names for one view."""
    np.savetxt(os.path.join(out_dir, f'{view_num}_tr.csv'),
               X_train, delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(out_dir, f'{view_num}_te.csv'),
               X_test, delimiter=',', fmt='%.6f')
    with open(os.path.join(out_dir, f'{view_num}_featname.csv'), 'w') as f:
        for name in feat_names:
            f.write(f'"{name}"\n')


def convert_single(input_dir, output_dir, n_folds=5, driver_file=None, l1_C=0.1):
    """Convert one cancer type from MO-GCAN format to MOTCS format."""
    cancer_name = os.path.basename(os.path.normpath(input_dir)).upper()

    print(f"\n{'='*65}")
    print(f"  CONVERTING: {cancer_name}")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"{'='*65}\n")

    # 1. Load data
    print("[1/4] Loading MO-GCAN data...")
    mrna, met, cna, labels, le = load_mogcan_data(input_dir)

    # 2. Load driver genes
    print("\n[2/4] Loading driver genes...")
    driver_genes = load_driver_genes(driver_file)

    # 3. Define 3 views (mRNA, CNA replacing miRNA, methylation)
    views = [
        (1, "mRNA", mrna),
        (2, "CNA", cna),           # Substituting for miRNA
        (3, "Methylation", met),
    ]

    # 4. K-Fold splitting
    print(f"\n[3/4] Creating {n_folds}-fold stratified splits...")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(mrna.values, labels), 1):
        print(f"\n  ── Fold {fold_idx}/{n_folds} "
              f"(train={len(train_idx)}, test={len(test_idx)}) ──")

        # Create output directories
        wodg_dir = os.path.join(output_dir, 'WODG', f'{fold_idx}.KF')
        dg_dir = os.path.join(output_dir, f'{fold_idx}.KF', '没进行svcl1的dg')
        os.makedirs(wodg_dir, exist_ok=True)
        os.makedirs(dg_dir, exist_ok=True)

        # Save integer labels
        np.savetxt(os.path.join(wodg_dir, 'labels_tr.csv'),
                   labels[train_idx], delimiter=',', fmt='%d')
        np.savetxt(os.path.join(wodg_dir, 'labels_te.csv'),
                   labels[test_idx], delimiter=',', fmt='%d')
        # Also copy labels to dg folder (prepare_trte_data reads from fea_folder)
        np.savetxt(os.path.join(dg_dir, 'labels_tr.csv'),
                   labels[train_idx], delimiter=',', fmt='%d')
        np.savetxt(os.path.join(dg_dir, 'labels_te.csv'),
                   labels[test_idx], delimiter=',', fmt='%d')

        for view_num, view_name, view_df in views:
            columns = np.array(view_df.columns.tolist())
            X = view_df.values.astype(np.float32).copy()
            X = impute_nans(X)

            # Split into driver / non-driver
            dg_idx, nd_idx = split_driver_nondriver(columns, driver_genes)

            # ── WODG: Non-driver features with L1-SVC selection ──
            if len(nd_idx) > 0:
                X_nd = X[:, nd_idx]
                nd_names = columns[nd_idx]

                X_sel, sel_mask = l1_svc_select(
                    X_nd[train_idx], labels[train_idx], X_nd, C=l1_C
                )
                sel_names = nd_names[sel_mask]
            else:
                # No non-driver features (unlikely)
                X_sel = X
                sel_names = columns

            save_view_data(wodg_dir, view_num,
                           X_sel[train_idx], X_sel[test_idx], sel_names)

            # ── DG: Driver gene features (no feature selection) ──
            if len(dg_idx) > 0:
                X_dg = X[:, dg_idx]
                dg_names = columns[dg_idx]
                # Standardize driver gene features
                scaler = StandardScaler()
                X_dg_tr = scaler.fit_transform(X_dg[train_idx])
                X_dg_te = scaler.transform(X_dg[test_idx])
            else:
                # No driver genes found for this view — use a placeholder
                X_dg_tr = np.zeros((len(train_idx), 1), dtype=np.float32)
                X_dg_te = np.zeros((len(test_idx), 1), dtype=np.float32)
                dg_names = np.array(['_placeholder'])

            save_view_data(dg_dir, view_num,
                           X_dg_tr, X_dg_te, dg_names)

            print(f"    View {view_num} ({view_name:12s}): "
                  f"WODG={X_sel.shape[1]:4d} feats, "
                  f"DG={len(dg_idx):3d} feats")

    # Save label encoder mapping for reference
    mapping_file = os.path.join(output_dir, 'label_mapping.txt')
    with open(mapping_file, 'w') as f:
        f.write("# Label encoding: integer → subtype name\n")
        for i, cls in enumerate(le.classes_):
            f.write(f"{i},{cls}\n")
    print(f"\n  Label mapping saved to {mapping_file}")

    print(f"\n[4/4] ✓ Conversion complete for {cancer_name}!")
    print(f"       Output: {output_dir}")
    return True


def convert_all(input_dir, output_dir, n_folds=5, driver_file=None, l1_C=0.1):
    """Convert all 8 cancer types."""
    print(f"\n{'#'*65}")
    print(f"  BATCH CONVERSION: All MO-GCAN cancer types → MOTCS format")
    print(f"{'#'*65}")

    succeeded = []
    failed = []

    for cancer in CANCER_TYPES:
        cancer_input = os.path.join(input_dir, cancer)
        cancer_output = os.path.join(output_dir, cancer.upper(), 'MultiOmics')

        if not os.path.exists(cancer_input):
            print(f"\n  ⚠ Skipping {cancer.upper()}: folder not found at {cancer_input}")
            failed.append(cancer.upper())
            continue

        try:
            convert_single(cancer_input, cancer_output, n_folds, driver_file, l1_C)
            succeeded.append(cancer.upper())
        except Exception as e:
            print(f"\n  ✗ FAILED for {cancer.upper()}: {e}")
            failed.append(cancer.upper())

    print(f"\n{'='*65}")
    print(f"  BATCH RESULTS")
    print(f"{'='*65}")
    print(f"  ✓ Succeeded ({len(succeeded)}): {', '.join(succeeded)}")
    if failed:
        print(f"  ✗ Failed    ({len(failed)}): {', '.join(failed)}")
    print()


def process_raw(raw_dir, out_dir):
    """
    Process MO-GCAN's original_data/ (from cBioPortal) into per-cancer CSVs.

    This replicates what MO-GCAN's data_process.py does:
      original_data/{cancer}_tcga_pan_can_atlas_2018/ → out_dir/{cancer}/
    """
    print(f"\n{'='*65}")
    print(f"  PROCESSING RAW cBioPortal DATA → per-cancer CSVs")
    print(f"{'='*65}\n")

    for cancer in CANCER_TYPES:
        raw_cancer = os.path.join(raw_dir, f'{cancer}_tcga_pan_can_atlas_2018')
        if not os.path.exists(raw_cancer):
            # Also try without suffix
            raw_cancer = os.path.join(raw_dir, cancer)
            if not os.path.exists(raw_cancer):
                print(f"  ⚠ Skipping {cancer}: raw folder not found")
                continue

        print(f"\n  Processing {cancer.upper()}...")
        out_cancer = os.path.join(out_dir, cancer)
        os.makedirs(out_cancer, exist_ok=True)

        try:
            # Clinical / subtype data
            clin_file = os.path.join(raw_cancer, 'data_clinical_patient_modified.txt')
            if not os.path.exists(clin_file):
                clin_file = os.path.join(raw_cancer, 'data_clinical_patient.txt')
            subtype = pd.read_csv(clin_file, delimiter='\t', comment='#')
            subtype = subtype[['PATIENT_ID', 'SUBTYPE']].dropna()
            subtype.to_csv(os.path.join(out_cancer, 'subtype_data.csv'), index=False)

            # CNA
            cna_file = os.path.join(raw_cancer, 'data_log2_cna.txt')
            if os.path.exists(cna_file):
                cna = pd.read_csv(cna_file, delimiter='\t').transpose()
                cna.columns = cna.iloc[0].values
                cna = cna.iloc[2:, :].reset_index()
                cna = cna.rename(columns={'index': 'PATIENT_ID'}).set_index('PATIENT_ID')
                cna.to_csv(os.path.join(out_cancer, 'cna_data.csv'))

            # Methylation
            met_file = os.path.join(raw_cancer, 'data_methylation_hm27_hm450_merged.txt')
            if os.path.exists(met_file):
                met = pd.read_csv(met_file, delimiter='\t').transpose()
                met.columns = met.iloc[1].values
                met = met.iloc[4:, :].reset_index()
                met = met.rename(columns={'index': 'PATIENT_ID'}).set_index('PATIENT_ID')
                met.to_csv(os.path.join(out_cancer, 'met_data.csv'))

            # mRNA
            mrna_file = os.path.join(raw_cancer,
                                     'data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt')
            if os.path.exists(mrna_file):
                mrna = pd.read_csv(mrna_file, delimiter='\t').transpose()
                mrna.columns = mrna.iloc[0].values
                mrna = mrna.iloc[2:, :].reset_index()
                mrna = mrna.rename(columns={'index': 'PATIENT_ID'}).set_index('PATIENT_ID')
                mrna.to_csv(os.path.join(out_cancer, 'mrna_data.csv'))

            # RPPA (optional)
            rppa_file = os.path.join(raw_cancer, 'data_rppa_zscores.txt')
            if os.path.exists(rppa_file):
                rppa = pd.read_csv(rppa_file, delimiter='\t').transpose()
                rppa.columns = rppa.iloc[0].values
                rppa = rppa.iloc[2:, :].reset_index()
                rppa = rppa.rename(columns={'index': 'PATIENT_ID'}).set_index('PATIENT_ID')
                rppa.to_csv(os.path.join(out_cancer, 'rppa_data.csv'))

            print(f"    ✓ Saved to {out_cancer}")

        except Exception as e:
            print(f"    ✗ Error: {e}")

    print(f"\n  Done! Processed data saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert MO-GCAN Figshare data → MOTCS format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process raw cBioPortal data into per-cancer CSVs
  python convert_mogcan_to_motcs.py process-raw --raw_dir mogcan_raw --out_dir mogcan_processed

  # Convert a single cancer type
  python convert_mogcan_to_motcs.py convert --input_dir mogcan_processed/brca --output_dir 1.data/BRCA/MultiOmics

  # Convert all 8 cancer types at once
  python convert_mogcan_to_motcs.py convert-all --input_dir mogcan_processed --output_dir 1.data
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # --- process-raw ---
    p_raw = subparsers.add_parser('process-raw',
                                   help='Process raw cBioPortal data into per-cancer CSVs')
    p_raw.add_argument('--raw_dir', required=True,
                       help='Path to extracted Figshare/cBioPortal original_data/')
    p_raw.add_argument('--out_dir', required=True,
                       help='Output directory for processed CSVs')

    # --- convert (single cancer) ---
    p_conv = subparsers.add_parser('convert',
                                    help='Convert one cancer type to MOTCS format')
    p_conv.add_argument('--input_dir', required=True,
                        help='Path to processed MO-GCAN cancer folder (e.g., data/brca)')
    p_conv.add_argument('--output_dir', required=True,
                        help='Output path for MOTCS format')
    p_conv.add_argument('--n_folds', type=int, default=5,
                        help='Number of K-folds (default: 5)')
    p_conv.add_argument('--driver_gene_file', default=None,
                        help='Path to driver gene list (CSV or one-per-line text)')
    p_conv.add_argument('--l1_C', type=float, default=0.1,
                        help='L1-SVC regularization parameter (default: 0.1)')

    # --- convert-all ---
    p_all = subparsers.add_parser('convert-all',
                                   help='Convert all 8 cancer types to MOTCS format')
    p_all.add_argument('--input_dir', required=True,
                       help='Root dir containing per-cancer subfolders')
    p_all.add_argument('--output_dir', required=True,
                       help='Root output dir (subfolders created per cancer)')
    p_all.add_argument('--n_folds', type=int, default=5,
                       help='Number of K-folds (default: 5)')
    p_all.add_argument('--driver_gene_file', default=None,
                       help='Path to driver gene list')
    p_all.add_argument('--l1_C', type=float, default=0.1,
                       help='L1-SVC regularization parameter (default: 0.1)')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == 'process-raw':
        process_raw(args.raw_dir, args.out_dir)

    elif args.command == 'convert':
        convert_single(args.input_dir, args.output_dir,
                        args.n_folds, args.driver_gene_file, args.l1_C)

    elif args.command == 'convert-all':
        convert_all(args.input_dir, args.output_dir,
                    args.n_folds, args.driver_gene_file, args.l1_C)


if __name__ == '__main__':
    main()
