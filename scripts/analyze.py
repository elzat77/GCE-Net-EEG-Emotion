import os
import sys
import argparse
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.statistics import full_comparison_report, paired_ttest, wilcoxon_test, cohens_d, confidence_interval_95


def load_metrics(results_dirs):
    methods = {}
    for dir_path in results_dirs:
        metrics_file = os.path.join(dir_path, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                data = json.load(f)
            method_name = os.path.basename(dir_path)
            if "folds" in data:
                cold_accs = [data["folds"][str(f)]["cold_acc"]
                           for f in range(1, 16) if str(f) in data["folds"]]
                methods[method_name] = cold_accs
    return methods


def print_summary(methods):
    print("\n=== Experiment Summary ===")
    for name, scores in sorted(methods.items()):
        arr = np.array(scores)
        ci = confidence_interval_95(arr)
        print(f"  {name}: {arr.mean():.4f} ± {arr.std(ddof=1):.4f}, 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="Paths to results directories")
    parser.add_argument("--output", default=None, help="Output path for comparison CSV")
    args = parser.parse_args()

    methods = load_metrics(args.results)

    if len(methods) < 2:
        print("Need at least 2 result directories to compare. Found:", len(methods))
        return

    print_summary(methods)

    output_path = args.output or os.path.join(os.path.commonprefix(args.results), "comparison_report.csv")
    report = full_comparison_report(methods, save_path=output_path)
    print(f"\nComparison report saved to {output_path}")
    print(report)


if __name__ == "__main__":
    main()
