#!/usr/bin/env python3
"""
Train QSVM and save artifacts.
Usage:
  python scripts/train.py --data data/Brain_Tumor.csv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.model import QSVMModel


def main():
    parser = argparse.ArgumentParser(description="Train QSVM Brain Tumour Classifier")
    parser.add_argument("--data", required=True, help="Path to Brain Tumor CSV dataset")
    parser.add_argument("--output", default="artifacts", help="Output directory for artifacts")
    args = parser.parse_args()

    import os
    os.environ["MODEL_DIR"] = args.output

    model = QSVMModel()
    metrics = model.train(args.data)
    model.save()

    print("\n=== Training Results ===")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print(f"\nClassification Report:")
    for cls, vals in metrics["classification_report"].items():
        if isinstance(vals, dict):
            print(f"  Class {cls}: precision={vals['precision']:.3f}  recall={vals['recall']:.3f}  f1={vals['f1-score']:.3f}")

    acc = metrics["accuracy"]
    threshold = 0.70
    if acc < threshold:
        print(f"\n[FAIL] Accuracy {acc:.4f} below threshold {threshold}. Aborting.")
        sys.exit(1)
    print(f"\n[PASS] Accuracy {acc:.4f} >= {threshold}")


if __name__ == "__main__":
    main()
