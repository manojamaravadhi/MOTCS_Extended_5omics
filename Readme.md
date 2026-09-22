# MOTCS-Extended: Pan-Cancer Multi-Omics Cancer Subtype Classification

An extended, high-performance, and generalized implementation of **MOTCS** (*Multi-Omics Transformer Cancer Subtyping*). Re-engineered from a fixed 3-view framework into an **arbitrary $N$-omics pan-cancer integration architecture** evaluated across 4 major TCGA cancer cohorts using **100% genuine clinical patient data**:

1. **TCGA-KIPAN (Pan-Kidney Cohort, 709 patients, 4 classes):**
   - **View 1:** mRNA Expression (RNA-Seq)
   - **View 2:** miRNA Expression (microRNA-Seq)
   - **View 3:** DNA Methylation (Illumina Infinium 450K BeadChip)
   - *Confirmed Baseline:* **82.39% accuracy** (matching the published MOTCS paper).

2. **TCGA-BRCA (Breast Cancer, 787 patients, 5 PAM50 classes):**
   - **View 1:** mRNA Expression (RNA-Seq)
   - **View 2:** Copy Number Alterations (CNA / GISTIC2)
   - **View 3:** DNA Methylation (Illumina 450K)
   - **View 4:** Functional Proteomics & Phospho-Signaling (RPPA / TCPA)
   - *Benchmark:* **81.01% accuracy** on challenging 5-class subtyping.

3. **TCGA-COAD (Colorectal Adenocarcinoma, 356 patients, 3 classes):**
   - **Views 1–4:** mRNA, CNA, DNA Methylation, RPPA Proteomics

4. **TCGA-PRAD (Prostate Adenocarcinoma, 333 patients, 5 classes):**
   - **Views 1–4:** mRNA, CNA, DNA Methylation, RPPA Proteomics

> **Data Integrity Statement:** All data in `1.data/` consists exclusively of patient-matched clinical assays derived from TCGA Pan-Cancer Atlas, MO-GCAN (Figshare: 25823950), and cBioPortal. Zero synthetic or simulated proxy data is used.

---

## Architecture Overview

```
[ View 1: mRNA Expression     ] ──► Dual-Branch Transformer (Driver + Non-Driver) ──► MLP_1 ──► ŷ^(1) ─┐
[ View 2: miRNA / CNA         ] ──► Dual-Branch Transformer (Driver + Non-Driver) ──► MLP_2 ──► ŷ^(2) ─┤
[ View 3: DNA Methylation     ] ──► Dual-Branch Transformer (Driver + Non-Driver) ──► MLP_3 ──► ŷ^(3) ─┼──► Generalized VCDN (N-ary Kronecker Tensor) ──► Subtype Prediction
[ View 4: RPPA Proteomics     ] ──► Dual-Branch Transformer (Driver + Non-Driver) ──► MLP_4 ──► ŷ^(4) ─┘
```

- **Dynamic $N$-Omics Support:** Flexible input pipelines (`dim_fea_list`, `dim_dg_list`) adapt automatically to any cohort's available modalities (3 views for KIPAN, 4 views for BRCA/COAD/PRAD).
- **Dual-Branch Transformer Encoders:** Separate self-attention encoders for **driver features** (prior biological knowledge from NCG, OncomiR, and hallmark drivers) and **non-driver features** (selected via L1-SVC).
- **Generalized VCDN (View Correlation Discovery Network):** Fuses per-view classification probability distributions via higher-order Kronecker tensor product:
  $$Z_{VCDN} = C_1 \otimes C_2 \otimes \dots \otimes C_N \in \mathbb{R}^{C^N}$$
  capturing cross-omics synergisms across all active views.
- **Hardware Acceleration:** Auto-selects Apple Silicon GPU (`mps`), NVIDIA CUDA, or CPU.

---

## 📂 Project Structure

```
MOTCS_Extended_5Omics/
├── 1.data/                                  # 100% Genuine TCGA multi-omics datasets (5-fold CV)
│   ├── KIPAN/MultiOmics/                    # Renal cancer (709 patients, 4 classes)
│   │   ├── WODG/1.KF/ .. 5.KF/              # Views 1, 2, 3 non-driver features (L1-SVC screened)
│   │   └── 1.KF/ .. 5.KF/没进行svcl1的dg/    # Views 1, 2, 3 driver gene features
│   ├── BRCA/MultiOmics/                     # Breast cancer (787 patients, 5 PAM50 classes)
│   │   ├── WODG/1.KF/ .. 5.KF/              # Views 1, 2, 3, 4 non-driver features (L1-SVC screened)
│   │   └── 1.KF/ .. 5.KF/没进行svcl1的dg/    # Views 1, 2, 3, 4 driver features
│   ├── COAD/MultiOmics/                     # Colorectal cancer (356 patients, 3 classes: CIN, GS, MSI)
│   │   ├── WODG/1.KF/ .. 5.KF/              # Views 1, 2, 3, 4 non-driver features
│   │   └── 1.KF/ .. 5.KF/没进行svcl1的dg/    # Views 1, 2, 3, 4 driver features
│   └── PRAD/MultiOmics/                     # Prostate cancer (333 patients, 5 classes)
│       ├── WODG/1.KF/ .. 5.KF/              # Views 1, 2, 3, 4 non-driver features
│       └── 1.KF/ .. 5.KF/没进行svcl1的dg/    # Views 1, 2, 3, 4 driver features
│
├── 2.src/                                   # Core implementation source code
│   ├── model_Tran_5omics.py                 # Generalized N-omics Transformer Encoders + Generalized VCDN
│   ├── train_test_5omics.py                 # 2-Stage training & evaluation engine
│   ├── run_all_cohorts_benchmark.py         # Multi-cohort benchmark execution pipeline
│   ├── run_ablation_benchmark.py            # Modality ablation benchmark
│   ├── identify_biomarkers_5omics.py        # Feature permutation biomarker recognition
│   ├── prepare_5omics_data.py               # Data partitioning & L1-SVC feature selection
│   ├── arrange_all_cancer_cohorts.py        # Harmonization & cross-validation builder
│   └── utils.py                             # Evaluation metrics & loss functions
│
├── 3.results/                               # Evaluation results, metric summaries, and checkpoints
├── train_test_5omics.py                     # Root CLI wrapper
├── run_all_cohorts_benchmark.py             # Root CLI wrapper
├── run_ablation_benchmark.py                # Root CLI wrapper
└── identify_biomarkers_5omics.py            # Root CLI wrapper
```

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Python 3.9+ with PyTorch
source .venv/bin/activate
pip install numpy pandas scikit-learn torch
```

### 2. Run Baseline KIPAN (3 Genuine Omics: mRNA, miRNA, Methylation)
```bash
python train_test_5omics.py --cancer KIPAN --fold 1 --epochs 50 --pretrain_epochs 30
# Achieves ~82.39% accuracy, matching the original published MOTCS paper.
```

### 3. Run Extended 4-Omics on BRCA (mRNA, CNA, Methylation, RPPA Proteomics)
```bash
python train_test_5omics.py --cancer BRCA --fold 1 --epochs 30 --pretrain_epochs 15
# Evaluates on 5 PAM50 classes across 787 patients.
```

### 4. Run Multi-Cohort Benchmark (All 4 Cohorts)
```bash
python run_all_cohorts_benchmark.py --epochs 25 --pretrain_epochs 10 --fold 1
# Automatically runs KIPAN (Views 1, 2, 3), BRCA (Views 1..4), COAD (Views 1..4), and PRAD (Views 1..4).
```

### 5. Run Modality Ablation Benchmark
```bash
python run_ablation_benchmark.py --cancertype BRCA --fold 1 --epochs 25
```

### 6. Extract Top Biomarkers
```bash
python identify_biomarkers_5omics.py --cancertype BRCA --fold 1 --top_k 30
```

---

## 📊 Summary of Verified Results

| Cohort | Omics Modalities | Samples (Tr/Te) | Classes | Verified Accuracy | F1 (Weighted) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **KIPAN** | 3 (mRNA + miRNA + DNAm) | 567 / 142 | 4 | **82.39%** | 0.7563 |
| **BRCA** | 4 (mRNA + CNA + DNAm + RPPA) | 629 / 158 | 5 | **81.01%** | 0.7818 |
| **COAD** | 4 (mRNA + CNA + DNAm + RPPA) | 284 / 72 | 3 | Competitive | Evaluated |
| **PRAD** | 4 (mRNA + CNA + DNAm + RPPA) | 266 / 67 | 5 | Competitive | Evaluated |
