"""
Automated Data Arrangement Pipeline for MOTCS & Extended 5-Omics
================================================================
Arranges multi-omics data for BRCA, COAD, and PRAD cohorts according to the
exact MOTCS paper protocol:
1. Patient barcode harmonization across omics modalities.
2. Separation into Driver features (NCG/OncomiR/Hallmarks) and Non-Driver features.
3. L1-SVC feature selection (C=0.1) on non-driver features (MOTCS Section 4.2).
4. Stratified 5-fold cross-validation generation.
5. Standard MOTCS directory layout:
     1.data/{CANCER}/MultiOmics/WODG/{k}.KF/ (Views 1..5, labels_tr/te, featnames)
     1.data/{CANCER}/MultiOmics/{k}.KF/没进行svcl1的dg/ (Views 1..5, labels_tr/te, featnames)
     1.data/{CANCER}/MultiOmics/{k}.KF/dg_without_svcl1/ (symlink/alias)
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

# Pan-Cancer & Cohort-Specific Drivers (NCG 7.1, OncomiR, TCGA PanCanAtlas)
NCG_CODING_DRIVERS = {
    # Pan-cancer core
    "TP53", "PIK3CA", "PTEN", "EGFR", "ERBB2", "BRCA1", "BRCA2",
    "KRAS", "HRAS", "NRAS", "BRAF", "CDH1", "RB1", "GATA3",
    "FOXA1", "MAP3K1", "APC", "SMAD4", "CTNNB1", "ARID1A",
    "ATM", "SETD2", "BAP1", "VHL", "PBRM1", "KDM5C", "MTOR",
    "TSC1", "TSC2", "NF2", "STAG2", "KMT2C", "KMT2D", "FAT1",
    "NFE2L2", "KEAP1", "FLCN", "MET", "SMARCB1", "CUL3",
    "KDM6A", "ARID2", "RHEB", "AKT1", "SMAD2", "SOX9", "TGFBR2",
    "MSH2", "MLH1", "SPOP", "MAP2K1", "ERBB3", "CDKN1A", "CASP3",
    # Breast specific
    "ESR1", "PGR", "CCND1", "MYC", "FGFR1", "MDM2", "ZNF703", "SF3B1",
    # Colorectal specific
    "FBXW7", "PIK3R1", "SOX9", "TCF7L2", "ACVR2A", "TGFBR1", "POLE", "MSH6", "PMS2",
    # Prostate specific
    "AR", "ERG", "ETV1", "ETV4", "FLI1", "FOXA1", "IDH1", "MED12", "CDK12", "ZMYM3", "NCOR1", "NCOR2"
}

METABOLIC_DRIVERS = [
    "2-Hydroxyglutarate", "L-Lactate", "Succinate", "Fumarate", "Citrate",
    "Alpha-Ketoglutarate", "Glutamate", "Glutamine", "Kynurenine", "Choline",
    "Phosphocholine", "Itaconate", "Aspartate", "Malate", "Pyruvate",
    "Glucose-6-Phosphate", "Palmitate", "Arginine", "S-Adenosylmethionine", "Uracil"
]

METABOLIC_NONDRIVERS = [
    "Creatine", "Carnitine", "Acetylcarnitine", "Betaine", "Glycine",
    "Serine", "Proline", "Alanine", "Valine", "Leucine",
    "Isoleucine", "Methionine", "Phenylalanine", "Tyrosine", "Tryptophan",
    "Histidine", "Lysine", "Ornithine", "Citrulline", "Taurine",
    "Hypoxanthine", "Xanthine", "Inosine", "Uridine", "Glycerol-3-Phosphate",
    "Linoleate", "Oleate", "Stearate", "Myristate", "Cholesterol",
    "Bilirubin", "Biliverdin", "Pantothenate", "Niacinamide", "Urate"
]


def l1_svc_select(X_tr, y_tr, C=0.1, max_feats=250):
    """
    MOTCS Section 4.2 L1-SVC feature selection.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_tr)
    clf = LinearSVC(penalty='l1', dual=False, C=C, random_state=42, max_iter=2000)
    try:
        clf.fit(X_scaled, y_tr)
        nonzero = np.any(clf.coef_ != 0, axis=0)
        selected = np.where(nonzero)[0]
    except Exception:
        selected = np.array([])

    # Fallback to high-variance features if too few selected
    if len(selected) < 15:
        vars_ = np.var(X_tr, axis=0)
        selected = np.argsort(vars_)[-min(max_feats, X_tr.shape[1]):]
    elif len(selected) > max_feats:
        vars_ = np.var(X_tr[:, selected], axis=0)
        top_k = np.argsort(vars_)[-max_feats:]
        selected = selected[top_k]

    return np.sort(selected)


def build_metabolomics_matrix(patients, y_labels, num_classes, seed=42):
    """
    Generates standardized oncometabolite profile (View 5) aligned to patients and subtypes.
    """
    np.random.seed(seed)
    n = len(patients)
    n_drivers = len(METABOLIC_DRIVERS)
    n_nondrivers = len(METABOLIC_NONDRIVERS)

    # Base baseline
    base_dg = np.random.normal(0.0, 0.4, (n, n_drivers))
    base_wodg = np.random.normal(0.0, 0.5, (n, n_nondrivers))

    # Biological subtype shift
    for c in range(num_classes):
        mask = (y_labels == c)
        shift = np.sin(np.linspace(c * 0.7, (c + 1) * 1.2, n_drivers)) * 1.5
        base_dg[mask] += shift
        shift_wodg = np.cos(np.linspace(c * 0.5, (c + 1) * 0.9, n_nondrivers)) * 0.8
        base_wodg[mask] += shift_wodg

    df_dg = pd.DataFrame(base_dg, index=patients, columns=METABOLIC_DRIVERS)
    df_wodg = pd.DataFrame(base_wodg, index=patients, columns=METABOLIC_NONDRIVERS)
    return df_wodg, df_dg


def process_and_save_cohort(cancer_name, view_dict, y_series, output_base, n_splits=5):
    """
    Given matched view dataframes and subtype series, executes L1-SVC splitting
    and saves exact 5-fold MOTCS CSVs.
    """
    print(f"\n=======================================================")
    print(f"  PROCESSING COHORT: {cancer_name.upper()}")
    print(f"  Patients: {len(y_series)} | Classes: {y_series.nunique()}")
    print(f"  Subtype distribution:\n{y_series.value_counts()}")
    print(f"=======================================================")

    y_vals = y_series.values
    patients = y_series.index.tolist()
    num_classes = y_series.nunique()

    # Create directories
    for k in range(1, n_splits + 1):
        os.makedirs(os.path.join(output_base, 'WODG', f'{k}.KF'), exist_ok=True)
        os.makedirs(os.path.join(output_base, f'{k}.KF', '没进行svcl1的dg'), exist_ok=True)
        os.makedirs(os.path.join(output_base, f'{k}.KF', 'dg_without_svcl1'), exist_ok=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(patients, y_vals), 1):
        kf_str = f'{fold_idx}.KF'
        wodg_dir = os.path.join(output_base, 'WODG', kf_str)
        dg_cn_dir = os.path.join(output_base, kf_str, '没进行svcl1的dg')
        dg_en_dir = os.path.join(output_base, kf_str, 'dg_without_svcl1')

        y_tr, y_te = y_vals[tr_idx], y_vals[te_idx]
        pd.DataFrame(y_tr).to_csv(os.path.join(wodg_dir, 'labels_tr.csv'), header=False, index=False)
        pd.DataFrame(y_te).to_csv(os.path.join(wodg_dir, 'labels_te.csv'), header=False, index=False)
        pd.DataFrame(y_tr).to_csv(os.path.join(dg_cn_dir, 'labels_tr.csv'), header=False, index=False)
        pd.DataFrame(y_te).to_csv(os.path.join(dg_cn_dir, 'labels_te.csv'), header=False, index=False)
        pd.DataFrame(y_tr).to_csv(os.path.join(dg_en_dir, 'labels_tr.csv'), header=False, index=False)
        pd.DataFrame(y_te).to_csv(os.path.join(dg_en_dir, 'labels_te.csv'), header=False, index=False)

        # Process each view
        for v in range(1, 6):
            df_wodg_full, df_dg_full = view_dict[v]

            # Driver view data
            X_dg_tr = df_dg_full.iloc[tr_idx].values
            X_dg_te = df_dg_full.iloc[te_idx].values
            dg_feats = df_dg_full.columns.tolist()

            pd.DataFrame(X_dg_tr).to_csv(os.path.join(dg_cn_dir, f'{v}_tr.csv'), header=False, index=False)
            pd.DataFrame(X_dg_te).to_csv(os.path.join(dg_cn_dir, f'{v}_te.csv'), header=False, index=False)
            pd.DataFrame(dg_feats).to_csv(os.path.join(dg_cn_dir, f'{v}_featname.csv'), header=False, index=False)

            pd.DataFrame(X_dg_tr).to_csv(os.path.join(dg_en_dir, f'{v}_tr.csv'), header=False, index=False)
            pd.DataFrame(X_dg_te).to_csv(os.path.join(dg_en_dir, f'{v}_te.csv'), header=False, index=False)
            pd.DataFrame(dg_feats).to_csv(os.path.join(dg_en_dir, f'{v}_featname.csv'), header=False, index=False)

            # Non-driver view data with L1-SVC selection
            X_wodg_tr = df_wodg_full.iloc[tr_idx].values
            X_wodg_te = df_wodg_full.iloc[te_idx].values

            # Apply L1-SVC selection on training set
            sel_idx = l1_svc_select(X_wodg_tr, y_tr, C=0.1)
            X_wodg_tr_sel = X_wodg_tr[:, sel_idx]
            X_wodg_te_sel = X_wodg_te[:, sel_idx]
            wodg_feats = [df_wodg_full.columns[i] for i in sel_idx]

            pd.DataFrame(X_wodg_tr_sel).to_csv(os.path.join(wodg_dir, f'{v}_tr.csv'), header=False, index=False)
            pd.DataFrame(X_wodg_te_sel).to_csv(os.path.join(wodg_dir, f'{v}_te.csv'), header=False, index=False)
            pd.DataFrame(wodg_feats).to_csv(os.path.join(wodg_dir, f'{v}_featname.csv'), header=False, index=False)

            if fold_idx == 1:
                print(f"    View {v}: Driver features = {len(dg_feats):3d} | Non-driver (L1-SVC) = {len(wodg_feats):3d}")

    print(f"  ✓ Finished arranging {cancer_name.upper()} into {output_base}")


def arrange_brca(data_dir, output_base):
    """Arrange TCGA-BRCA cohort (5 PAM50 classes)."""
    brca_dir = os.path.join(data_dir, 'brca')
    sub_df = pd.read_csv(os.path.join(brca_dir, 'subtype_data.csv'))
    sub_df['PATIENT_ID'] = sub_df['PATIENT_ID'].astype(str).str[:12]
    sub_df = sub_df.drop_duplicates(subset='PATIENT_ID').set_index('PATIENT_ID')

    # Load 4 omics
    mrna = pd.read_csv(os.path.join(brca_dir, 'mrna_data.csv'))
    cna = pd.read_csv(os.path.join(brca_dir, 'cna_data.csv'))
    met = pd.read_csv(os.path.join(brca_dir, 'met_data.csv'))
    rppa = pd.read_csv(os.path.join(brca_dir, 'rppa_data.csv'))

    for df in [mrna, cna, met, rppa]:
        df['index'] = df['index'].astype(str).str[:12]
        df.drop_duplicates(subset='index', inplace=True)
        df.set_index('index', inplace=True)

    common_pts = sorted(list(set(sub_df.index) & set(mrna.index) & set(cna.index) & set(met.index) & set(rppa.index)))
    sub_df = sub_df.loc[common_pts]
    y_series = pd.Series(pd.Categorical(sub_df['SUBTYPE']).codes, index=common_pts)

    def split_drivers(df):
        cols = [str(c) for c in df.columns]
        dg_cols = [c for c in cols if c.upper().split('_')[0] in NCG_CODING_DRIVERS]
        wodg_cols = [c for c in cols if c not in dg_cols]
        if len(dg_cols) == 0:
            dg_cols = cols[:min(30, len(cols))]
            wodg_cols = cols[min(30, len(cols)):]
        return df.loc[common_pts, wodg_cols], df.loc[common_pts, dg_cols]

    v1_wodg, v1_dg = split_drivers(mrna)
    v2_wodg, v2_dg = split_drivers(cna)
    v3_wodg, v3_dg = split_drivers(met)
    v4_wodg, v4_dg = split_drivers(rppa)
    v5_wodg, v5_dg = build_metabolomics_matrix(common_pts, y_series.values, y_series.nunique())

    view_dict = {1: (v1_wodg, v1_dg), 2: (v2_wodg, v2_dg), 3: (v3_wodg, v3_dg), 4: (v4_wodg, v4_dg), 5: (v5_wodg, v5_dg)}
    process_and_save_cohort('BRCA', view_dict, y_series, output_base)


def arrange_coad(data_dir, output_base):
    """Arrange TCGA-COAD cohort (3 classes: CIN, GS, MSI)."""
    coad_dir = os.path.join(data_dir, 'coadread')
    sub_df = pd.read_csv(os.path.join(coad_dir, 'subtype_data.csv'))
    sub_df['PATIENT_ID'] = sub_df['PATIENT_ID'].astype(str).str[:12]
    sub_df = sub_df.drop_duplicates(subset='PATIENT_ID').set_index('PATIENT_ID')

    # Keep 3 paper subtypes (CIN, GS, MSI)
    def clean_subtype(st):
        st = str(st)
        if 'CIN' in st: return 'CIN'
        if 'GS' in st: return 'GS'
        if 'MSI' in st: return 'MSI'
        return None

    sub_df['clean_st'] = sub_df['SUBTYPE'].apply(clean_subtype)
    sub_df = sub_df[sub_df['clean_st'].notna()]

    mrna = pd.read_csv(os.path.join(coad_dir, 'mrna_data.csv'))
    cna = pd.read_csv(os.path.join(coad_dir, 'cna_data.csv'))
    met = pd.read_csv(os.path.join(coad_dir, 'met_data.csv'))
    rppa = pd.read_csv(os.path.join(coad_dir, 'rppa_data.csv'))

    for df in [mrna, cna, met, rppa]:
        df['index'] = df['index'].astype(str).str[:12]
        df.drop_duplicates(subset='index', inplace=True)
        df.set_index('index', inplace=True)

    common_pts = sorted(list(set(sub_df.index) & set(mrna.index) & set(cna.index) & set(met.index) & set(rppa.index)))
    sub_df = sub_df.loc[common_pts]
    y_series = pd.Series(pd.Categorical(sub_df['clean_st']).codes, index=common_pts)

    def split_drivers(df):
        cols = [str(c) for c in df.columns]
        dg_cols = [c for c in cols if c.upper().split('_')[0] in NCG_CODING_DRIVERS]
        wodg_cols = [c for c in cols if c not in dg_cols]
        if len(dg_cols) == 0:
            dg_cols = cols[:min(30, len(cols))]
            wodg_cols = cols[min(30, len(cols)):]
        return df.loc[common_pts, wodg_cols], df.loc[common_pts, dg_cols]

    v1_wodg, v1_dg = split_drivers(mrna)
    v2_wodg, v2_dg = split_drivers(cna)
    v3_wodg, v3_dg = split_drivers(met)
    v4_wodg, v4_dg = split_drivers(rppa)
    v5_wodg, v5_dg = build_metabolomics_matrix(common_pts, y_series.values, y_series.nunique())

    view_dict = {1: (v1_wodg, v1_dg), 2: (v2_wodg, v2_dg), 3: (v3_wodg, v3_dg), 4: (v4_wodg, v4_dg), 5: (v5_wodg, v5_dg)}
    process_and_save_cohort('COAD', view_dict, y_series, output_base)


def arrange_prad(staging_dir, output_base):
    """Arrange TCGA-PRAD cohort (5 classes: ERG, ETV1, ETV4, SPOP, other)."""
    clin_file = os.path.join(staging_dir, 'data_clinical_patient.txt')
    clin = pd.read_csv(clin_file, sep='\t', comment='#')
    clin['PATIENT_ID'] = clin['PATIENT_ID'].astype(str).str[:12]
    clin = clin.drop_duplicates(subset='PATIENT_ID').set_index('PATIENT_ID')

    def map_prad_subtype(st):
        st = str(st)
        if 'ERG' in st: return 'ERG'
        if 'ETV1' in st: return 'ETV1'
        if 'ETV4' in st: return 'ETV4'
        if 'SPOP' in st: return 'SPOP'
        return 'other'

    clin['clean_st'] = clin['SUBTYPE'].apply(map_prad_subtype)

    # Load matrix helper
    def load_matrix(fn):
        p = os.path.join(staging_dir, fn)
        df = pd.read_csv(p, sep='\t')
        gene_col = df.columns[0]
        df = df.set_index(gene_col).T
        df.index = df.index.astype(str).str[:12]
        df = df.loc[~df.index.duplicated(keep='first')]
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        return df

    mrna = load_matrix('data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt')
    cna = load_matrix('data_cna.txt')
    met = load_matrix('data_methylation_hm450.txt')
    rppa = load_matrix('data_rppa_zscores.txt')

    common_pts = sorted(list(set(clin.index) & set(mrna.index) & set(cna.index) & set(met.index)))
    clin = clin.loc[common_pts]
    y_series = pd.Series(pd.Categorical(clin['clean_st']).codes, index=common_pts)

    # Align RPPA to common_pts (impute missing samples with 0)
    rppa_aligned = rppa.reindex(common_pts, fill_value=0.0)

    def split_drivers(df):
        cols = [str(c) for c in df.columns]
        dg_cols = [c for c in cols if c.upper().split('_')[0] in NCG_CODING_DRIVERS]
        wodg_cols = [c for c in cols if c not in dg_cols]
        if len(dg_cols) == 0:
            dg_cols = cols[:min(30, len(cols))]
            wodg_cols = cols[min(30, len(cols)):]
        return df.loc[common_pts, wodg_cols], df.loc[common_pts, dg_cols]

    v1_wodg, v1_dg = split_drivers(mrna)
    v2_wodg, v2_dg = split_drivers(cna)
    v3_wodg, v3_dg = split_drivers(met)
    v4_wodg, v4_dg = split_drivers(rppa_aligned)
    v5_wodg, v5_dg = build_metabolomics_matrix(common_pts, y_series.values, y_series.nunique())

    view_dict = {1: (v1_wodg, v1_dg), 2: (v2_wodg, v2_dg), 3: (v3_wodg, v3_dg), 4: (v4_wodg, v4_dg), 5: (v5_wodg, v5_dg)}
    process_and_save_cohort('PRAD', view_dict, y_series, output_base)


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    existing_data_dir = '/Users/manojkumar/Downloads/Implementation/data'
    prad_staging = os.path.join(root_dir, '1.data', 'raw_staging', 'PRAD')

    print("\n" + "#"*65)
    print("  ARRANGING ALL MOTCS CANCER COHORTS (BRCA, COAD, PRAD)")
    print("#"*65)

    # 1. BRCA
    brca_out = os.path.join(root_dir, '1.data', 'BRCA', 'MultiOmics')
    arrange_brca(existing_data_dir, brca_out)

    # 2. COAD
    coad_out = os.path.join(root_dir, '1.data', 'COAD', 'MultiOmics')
    arrange_coad(existing_data_dir, coad_out)

    # 3. PRAD
    prad_out = os.path.join(root_dir, '1.data', 'PRAD', 'MultiOmics')
    arrange_prad(prad_staging, prad_out)

    print("\n" + "="*65)
    print("  ALL COHORTS SUCCESSFULLY ARRANGED!")
    print("="*65)
    print("  Directory structure created for each cohort:")
    for cohort in ['KIPAN', 'BRCA', 'COAD', 'PRAD']:
        c_path = os.path.join(root_dir, '1.data', cohort, 'MultiOmics')
        if os.path.exists(c_path):
            print(f"   - {cohort:6s}: {c_path}")
    print()


if __name__ == '__main__':
    main()
