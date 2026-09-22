"""
Multi-Cohort Benchmark Runner for Extended MOTCS
================================================
Evaluates the 5-omics architecture across multiple cancer cohorts:
  - KIPAN (Renal Cancer)
  - BRCA  (Breast Cancer)
  - COAD  (Colorectal Cancer)
  - PRAD  (Prostate Cancer)
"""

import os
import re
import sys
import argparse
import subprocess
import pandas as pd

COHORTS_CONFIG = {
    "KIPAN": {"subtypes": 4, "name": "Renal Cancer (KIRC, KIRP, KICH, Normal)", "default_views": [1, 2, 3]},
    "BRCA":  {"subtypes": 5, "name": "Breast Cancer (LumA, LumB, Basal, Her2, Normal)", "default_views": [1, 2, 3, 4]},
    "COAD":  {"subtypes": 3, "name": "Colorectal Cancer (CIN, GS, MSI)", "default_views": [1, 2, 3, 4]},
    "PRAD":  {"subtypes": 5, "name": "Prostate Cancer (ERG, ETV1, ETV4, SPOP, Other)", "default_views": [1, 2, 3, 4]}
}


def run_command_capture(cmd, env):
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, env=env
    )
    acc, f1_w, f1_m = None, None, None
    for line in iter(process.stdout.readline, ""):
        sys.stdout.write(line)
        sys.stdout.flush()

        acc_match = re.search(r"ACC:\s+([0-9\.]+)", line)
        if acc_match:
            acc = float(acc_match.group(1))

        f1w_match = re.search(r"F1_Weighted:\s+([0-9\.]+)", line)
        if f1w_match:
            f1_w = float(f1w_match.group(1))

        f1m_match = re.search(r"F1_Macro:\s+([0-9\.]+)", line)
        if f1m_match:
            f1_m = float(f1m_match.group(1))

    process.stdout.close()
    process.wait()
    return acc, f1_w, f1_m


def main():
    parser = argparse.ArgumentParser(description="Multi-Cohort Extended MOTCS Runner")
    parser.add_argument("--cohorts", nargs="+", default=["KIPAN", "BRCA", "COAD", "PRAD"],
                        help="Cohorts to benchmark (default: KIPAN BRCA COAD PRAD)")
    parser.add_argument("--views", nargs="+", type=int, default=None,
                        help="Views to activate (default: cohort-specific real views, e.g. 1 2 3 for KIPAN, 1 2 3 4 for BRCA/COAD/PRAD)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of joint training epochs (default: 15)")
    parser.add_argument("--pretrain_epochs", type=int, default=10, help="Number of pretrain epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--fold", type=int, default=1, help="Cross-validation fold to evaluate (default: 1)")
    args = parser.parse_args()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(cur_dir)
    train_script = os.path.join(cur_dir, "train_test_5omics.py")

    env = os.environ.copy()
    env["PYTHONPATH"] = cur_dir + ":" + env.get("PYTHONPATH", "")

    results = []

    print("\n" + "=" * 70)
    print("  EXTENDED MOTCS MULTI-COHORT EXECUTION")
    print(f"  Cohorts: {args.cohorts}")
    print(f"  Views: {args.views}")
    print(f"  Parameters: Epochs={args.epochs}, Pretrain={args.pretrain_epochs}, BatchSize={args.batch_size}, Fold={args.fold}")
    print("=" * 70 + "\n")

    for cohort in args.cohorts:
        cohort = cohort.upper()
        if cohort not in COHORTS_CONFIG:
            print(f"Unknown cohort: {cohort}. Supported: {list(COHORTS_CONFIG.keys())}")
            continue

        cfg = COHORTS_CONFIG[cohort]
        subtypes = cfg["subtypes"]
        cohort_views = args.views if args.views is not None else cfg["default_views"]

        print("\n" + "#" * 70)
        print(f"  Running {cohort} ({cfg['name']}, {subtypes} Subtypes, Fold {args.fold})")
        print("#" * 70 + "\n")

        views_str = [str(v) for v in cohort_views]
        cmd = [
            sys.executable, train_script,
            "--cancer", cohort,
            "--views"
        ] + views_str + [
            "--fold", str(args.fold),
            "--epochs", str(args.epochs),
            "--pretrain_epochs", str(args.pretrain_epochs),
            "--batch_size", str(args.batch_size)
        ]
        acc, f1w, f1m = run_command_capture(cmd, env)

        res_entry = {
            "Cohort": cohort,
            "Subtypes": subtypes,
            "Views": "+".join(views_str),
            "Accuracy": f"{acc*100:.2f}%" if acc is not None else "N/A",
            "F1_Weighted": f"{f1w:.4f}" if f1w is not None else "N/A",
            "F1_Macro": f"{f1m:.4f}" if f1m is not None else "N/A"
        }
        results.append(res_entry)

    df_res = pd.DataFrame(results)
    out_dir = os.path.join(root_dir, "3.results", "extended_benchmark")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "extended_benchmark_results.csv")
    md_path = os.path.join(out_dir, "extended_benchmark_results.md")

    df_res.to_csv(csv_path, index=False)

    headers = list(df_res.columns)
    md_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join([":---"] * len(headers)) + " |"]
    for _, row in df_res.iterrows():
        md_lines.append("| " + " | ".join(str(val) for val in row.values) + " |")
    md_table = "\n".join(md_lines)

    with open(md_path, "w") as f:
        f.write("# Extended MOTCS Multi-Cohort Benchmark Results\n\n")
        f.write(f"**Execution Parameters**: Epochs={args.epochs}, Pretrain={args.pretrain_epochs}, BatchSize={args.batch_size}, Fold={args.fold}\n\n")
        f.write(md_table)
        f.write("\n")

    print("\n" + "=" * 70)
    print("  EXTENDED MOTCS BENCHMARK SUMMARY")
    print("=" * 70)
    print(df_res.to_string(index=False))
    print(f"\nSaved benchmark outputs to:\n  - CSV: {csv_path}\n  - Markdown: {md_path}\n")


if __name__ == "__main__":
    main()
