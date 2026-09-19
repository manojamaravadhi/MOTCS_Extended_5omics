"""
MOTCS Extended: Ablation Study & Multi-Omics Comparison Benchmark
Mirrors Figure 1 and Tables 1-4 of the MOTCS paper:
Compares:
  - Single-omics: mRNA, miRNA, DNA methylation, Proteomics, Metabolomics
  - 3-Omics MOTCS: [1, 2, 3] (Original MOTCS)
  - 4-Omics MOTCS: [1, 2, 3, 4] (+ Proteomics)
  - 5-Omics Extended MOTCS: [1, 2, 3, 4, 5] (Full Multi-Omics)
"""

import os
import argparse
import numpy as np
import pandas as pd
from train_test_5omics import main as run_train_test


def parse_args():
    parser = argparse.ArgumentParser(description='Run MOTCS Multi-Omics Ablation Benchmarks')
    parser.add_argument('-c', '--cancertype', type=str, default='KIPAN')
    parser.add_argument('-f', '--fold', type=int, default=1)
    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('-pe', '--pretrain_epochs', type=int, default=15)
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='auto')
    return parser.parse_args()


class ArgsWrapper:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def main():
    cli_args = parse_args()

    experiments = [
        # Single Omics
        {"name": "Single-Omics (mRNA)",             "views": [1]},
        {"name": "Single-Omics (miRNA)",            "views": [2]},
        {"name": "Single-Omics (DNA Methylation)",  "views": [3]},
        {"name": "Single-Omics (Proteomics)",       "views": [4]},
        {"name": "Single-Omics (Metabolomics)",     "views": [5]},
        # Multi-Omics Baselines & Extensions
        {"name": "Original MOTCS (3-Omics)",       "views": [1, 2, 3]},
        {"name": "MOTCS + Proteomics (4-Omics)",    "views": [1, 2, 3, 4]},
        {"name": "MOTCS Extended (Full 5-Omics)",   "views": [1, 2, 3, 4, 5]},
    ]

    results = []

    print("=" * 70)
    print("STARTING MOTCS 5-OMICS ABLATION BENCHMARK")
    print(f"Cancer: {cli_args.cancertype} | Fold: {cli_args.fold} | Epochs: {cli_args.epochs} (Pretrain: {cli_args.pretrain_epochs})")
    print("=" * 70)

    for exp in experiments:
        print(f"\n>>> Running Experiment: {exp['name']} (Views: {exp['views']}) <<<")
        run_args = ArgsWrapper(
            cancertype=cli_args.cancertype,
            views=exp["views"],
            fold=cli_args.fold,
            epochs=cli_args.epochs,
            pretrain_epochs=cli_args.pretrain_epochs,
            head=4,
            dropout=0.1,
            batch_size=cli_args.batch_size,
            learningrate_e=0.0001,
            learningrate_c=0.0001,
            dim_feedforward=256,
            d_model=64,
            seed=42,
            device=cli_args.device,
            data_dir='',
            save_dir=''
        )

        res = run_train_test(run_args)
        results.append({
            "Experiment": exp["name"],
            "Views": "+".join(map(str, exp["views"])),
            "Num_Views": len(exp["views"]),
            "Accuracy": res["acc"],
            "F1_Weighted": res["f1_weighted"],
            "F1_Macro": res["f1_macro"],
            "Best_Epoch": res["epoch"]
        })

    df_res = pd.DataFrame(results)

    # Save to CSV and Markdown
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    out_dir = os.path.join(project_root, '3.results', cli_args.cancertype)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "ablation_benchmark_results.csv")
    md_path = os.path.join(out_dir, "ablation_benchmark_results.md")

    df_res.to_csv(csv_path, index=False)

    # Convert DataFrame to Markdown table manually (no tabulate dependency)
    headers = list(df_res.columns)
    md_lines = [
        f"# MOTCS 5-Omics Ablation Benchmark Results\n\n**Cohort:** {cli_args.cancertype} (Fold {cli_args.fold})\n\n",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    for _, row in df_res.iterrows():
        row_str = " | ".join(
            f"{val:.4f}" if isinstance(val, (float, np.floating)) else str(val)
            for val in row
        )
        md_lines.append(f"| {row_str} |")
    md_content = "\n".join(md_lines) + "\n"

    with open(md_path, 'w') as f:
        f.write(md_content)

    print("\n" + "=" * 70)
    print("ALL ABLATION EXPERIMENTS COMPLETED!")
    print(f"Results saved to:\n  - {csv_path}\n  - {md_path}")
    print("=" * 70 + "\n")
    print(df_res.to_string(index=False))


if __name__ == '__main__':
    main()
