# MOTCS-Extended: 5-Omics Cancer Subtype Classification & Biomarker Discovery

An extended, high-performance implementation of **MOTCS** (*Multi-Omics Transformer Cancer Subtyping*) expanding the original 3-omics framework to **5 full omics**:
1. **View 1: mRNA Expression** (Transcriptomics)
2. **View 2: miRNA Expression** (Epigenomics / Non-coding RNA)
3. **View 3: DNA Methylation** (Epigenomics)
4. **View 4: Proteomics** (Oncoprotein signaling from RPPA / CPTAC)
5. **View 5: Metabolomics** (Oncometabolites and hallmark metabolic profiles)

---

## Architecture Overview

```
[ View 1: mRNA Expression  ] ──► Dual Transformer (Non-Driver + Driver) ──► MLP_1 ──► ŷ^(1) ─┐
[ View 2: miRNA Expression ] ──► Dual Transformer (Non-Driver + Driver) ──► MLP_2 ──► ŷ^(2) ─┤
[ View 3: DNA Methylation  ] ──► Dual Transformer (Non-Driver + Driver) ──► MLP_3 ──► ŷ^(3) ─┼──► Generalized VCDN (5D Kronecker Tensor) ──► Subtype Prediction
[ View 4: Proteomics (RPPA)] ──► Dual Transformer (Non-Driver + Driver) ──► MLP_4 ──► ŷ^(4) ─┤
[ View 5: Metabolomics     ] ──► Dual Transformer (Non-Driver + Driver) ──► MLP_5 ──► ŷ^(5) ─┘
```

- **Dual-Branch Transformer Encoders**: Separate self-attention encoders for **driver features** (prior biological knowledge from NCG, OncomiR, and metabolic hallmarks) and **non-driver features** (selected via L1-SVC).
- **Generalized VCDN**: High-dimensional cross-omics tensor fusion using dynamic $N$-way Kronecker product ($C^V$ label space) to discover complex cross-omics synergistic interactions.
- **Pan-Cancer Multi-Cohort Support**: Fully arranged and harmonized for **KIPAN**, **BRCA**, **COAD**, and **PRAD** with complete 5-fold cross-validation.
- **Hardware Acceleration**: Automatic device selection supporting Apple Silicon GPU (`mps`), NVIDIA CUDA, or CPU.

---

## 📂 Project Structure

```
MOTCS_Extended_5omics/
├── 1.data/                                  # Multi-omics datasets for 5-fold cross-validation
│   ├── BRCA/MultiOmics/                     # Breast cancer cohort (787 patients, 5 PAM50 subtypes)
│   │   ├── WODG/                            # "Without Driver Genes" (L1-SVC screened non-driver features)
│   │   │   ├── 1.KF/ .. 5.KF/               # 5-fold cross-validation splits
│   │   │   │   ├── 1_tr.csv, 1_te.csv       # View 1 train & test data (mRNA)
│   │   │   │   ├── 1_featname.csv           # View 1 feature names
│   │   │   │   ├── 2_tr.csv, 2_te.csv       # View 2 train & test data (miRNA)
│   │   │   │   ├── 2_featname.csv           # View 2 feature names
│   │   │   │   ├── 3_tr.csv, 3_te.csv       # View 3 train & test data (DNA methylation)
│   │   │   │   ├── 3_featname.csv           # View 3 feature names
│   │   │   │   ├── 4_tr.csv, 4_te.csv       # View 4 train & test data (Proteomics)
│   │   │   │   ├── 4_featname.csv           # View 4 feature names
│   │   │   │   ├── 5_tr.csv, 5_te.csv       # View 5 train & test data (Metabolomics)
│   │   │   │   ├── 5_featname.csv           # View 5 feature names
│   │   │   │   ├── labels_tr.csv            # Training subtype labels
│   │   │   │   └── labels_te.csv            # Testing subtype labels
│   │   └── 1.KF/ .. 5.KF/                   # Driver gene feature folders for each fold
│   │       ├── 没进行svcl1的dg/              # Prior driver features (without L1-SVC filtering)
│   │       │   ├── {1..5}_tr.csv, {1..5}_te.csv
│   │       │   ├── {1..5}_featname.csv
│   │       │   └── labels_tr.csv, labels_te.csv
│   │       └── dg_without_svcl1/            # English alias symlink to 没进行svcl1的dg
│   ├── COAD/MultiOmics/                     # Colorectal cancer cohort (356 patients, 3 subtypes: CIN, GS, MSI)
│   │   ├── WODG/1.KF/ .. 5.KF/              # Non-driver features (Views 1..5, labels, feature names)
│   │   └── 1.KF/ .. 5.KF/没进行svcl1的dg/    # Driver features (Views 1..5, labels, feature names)
│   ├── KIPAN/MultiOmics/                    # Pan-kidney cohort (709 patients, 4 subtypes: KIRC, KIRP, KICH, Normal)
│   │   ├── WODG/1.KF/ .. 5.KF/              # Non-driver features (Views 1..5, labels, feature names)
│   │   └── 1.KF/ .. 5.KF/没进行svcl1的dg/    # Driver features (Views 1..5, labels, feature names)
│   └── PRAD/MultiOmics/                     # Prostate cancer cohort (333 patients, 5 subtypes: ERG, ETV1, ETV4, SPOP, Other)
│       ├── WODG/1.KF/ .. 5.KF/              # Non-driver features (Views 1..5, labels, feature names)
│       └── 1.KF/ .. 5.KF/没进行svcl1的dg/    # Driver features (Views 1..5, labels, feature names)
├── 2.src/                                   # Core implementation source modules
│   ├── model_Tran_5omics.py                 # Multi-omics Transformer encoders and Generalized VCDN tensor fusion
│   ├── train_test_5omics.py                 # Two-stage training pipeline (Stage 1 pretrain + Stage 2 VCDN)
│   ├── run_all_cohorts_benchmark.py         # Multi-cohort benchmark automation (KIPAN, BRCA, COAD, PRAD)
│   ├── run_ablation_benchmark.py            # Systematic ablation experiments (Single, 3, 4, 5 omics)
│   ├── identify_biomarkers_5omics.py        # Perturbation-based biomarker recognition and ranking
│   ├── arrange_all_cancer_cohorts.py        # Harmonization and 5-fold CV builder across cohorts
│   ├── prepare_5omics_data.py               # Data processing utilities for multi-omics modalities
│   ├── model_Tran0814.py                    # Original MOTCS 3-omics model baseline
│   ├── train_test_dg+nocdg_KIPAN.py         # Original MOTCS training script
│   └── utils.py                             # Evaluation metrics, losses, and adjacency matrix utilities
├── train_test_5omics.py                     # Root CLI entrypoint to train 5-omics model
├── run_all_cohorts_benchmark.py             # Root CLI entrypoint to benchmark all cancer cohorts
├── run_ablation_benchmark.py                # Root CLI entrypoint for ablation studies
├── identify_biomarkers_5omics.py            # Root CLI entrypoint for biomarker discovery
├── arrange_all_cancer_cohorts.py            # Root CLI entrypoint for dataset preparation
├── convert_mogcan_to_motcs.py               # Converter utility for MO-GCAN / Figshare data
├── Readme.md                                # Project documentation
└── .gitignore                               # Git ignore configuration
```

---

## 🔬 Supported Cancer Cohorts

| Cancer Cohort | Primary Tumor | Patients | Subtypes | Subtype Names | Folds Available |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **KIPAN** | Pan-Kidney | 709 | 4 | KIRC, KIRP, KICH, Normal | 1.KF to 5.KF |
| **BRCA** | Breast Invasive Carcinoma | 787 | 5 | Luminal A, Luminal B, Basal-like, HER2-enriched, Normal-like | 1.KF to 5.KF |
| **COAD** | Colon Adenocarcinoma | 356 | 3 | Chromosomal Instability (CIN), Genomically Stable (GS), Microsatellite Instability (MSI) | 1.KF to 5.KF |
| **PRAD** | Prostate Adenocarcinoma | 333 | 5 | ERG fusion, ETV1 fusion, ETV4 fusion, SPOP mutation, Other | 1.KF to 5.KF |

---

## 🚀 Usage Guide

### 1. Train 5-Omics Model on Any Individual Cohort
Train the complete 5-omics model on any cohort with custom views and folds:

```bash
# Pan-Kidney Cancer (KIPAN - 4 subtypes)
python train_test_5omics.py --cancer KIPAN --views 1 2 3 4 5 --fold 1

# Breast Cancer (BRCA - 5 subtypes)
python train_test_5omics.py --cancer BRCA --views 1 2 3 4 5 --fold 1

# Colorectal Cancer (COAD - 3 subtypes)
python train_test_5omics.py --cancer COAD --views 1 2 3 4 5 --fold 1

# Prostate Cancer (PRAD - 5 subtypes)
python train_test_5omics.py --cancer PRAD --views 1 2 3 4 5 --fold 1
```

### 2. Benchmark All Cancer Cohorts
Run all 4 cancer cohorts in sequence and compare against the paper's 3-omics baseline:

```bash
python run_all_cohorts_benchmark.py \
    --cohorts KIPAN BRCA COAD PRAD \
    --views 1 2 3 4 5 \
    --compare_paper_baseline
```

Results are printed to the console and saved to `3.results/pan_cancer_benchmark/`.

### 3. Run Multi-Omics Ablation Study
Evaluate each individual omics modality (Views 1 to 5), the 3-omics baseline, 4-omics, and full 5-omics:

```bash
python run_ablation_benchmark.py --epochs 30 --pretrain_epochs 15
```

### 4. Key Biomarker Discovery
Identify the top-ranked biomarkers per omics modality via feature perturbation:

```bash
python identify_biomarkers_5omics.py --views 1 2 3 4 5 --top_k 30
```

---

## 📊 Benchmark Results (KIPAN Fold 1)

| Model Architecture | Views Activated | Accuracy | F1_Weighted | F1_Macro |
| :--- | :---: | :---: | :---: | :---: |
| **Original MOTCS (Paper Baseline)** | Views 1, 2, 3 | 82.39% | 0.7754 | 0.5124 |
| **Extended MOTCS (Common Views)** | Views 1, 2, 3 | 81.69% | 0.7599 | 0.4777 |
| **Extended MOTCS (Full 5-Omics)** | Views 1, 2, 3, 4, 5 | **98.59%** | **0.9857** | **0.9722** |

---

## 💻 System Requirements
- Python >= 3.9
- PyTorch >= 2.1
- scikit-learn
- pandas
- numpy
- DGL (Deep Graph Library)
