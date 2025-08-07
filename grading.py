#!/usr/bin/env python3
"""
Grading script for ASVspoof 2019 Voice Anti-spoofing homework
Based on the requirements from README.md
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve


def compute_eer(bonafide_scores, spoof_scores):
    """
    Compute Equal Error Rate (EER)
    This function matches the one mentioned in README.md

    Args:
        bonafide_scores: scores for bonafide (genuine) samples
        spoof_scores: scores for spoof (fake) samples

    Returns:
        eer: Equal Error Rate as percentage
    """
    # Combine scores and labels
    scores = np.concatenate([bonafide_scores, spoof_scores])
    labels = np.concatenate(
        [np.ones(len(bonafide_scores)), np.zeros(len(spoof_scores))]
    )

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)

    # EER is the point where FPR = FNR = 1 - TPR
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        return eer * 100  # Convert to percentage
    except:
        # Fallback method if interpolation fails
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.absolute(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        return eer * 100


def load_protocol(protocol_path):
    """Load ASVspoof protocol file"""
    protocol_data = {}

    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                speaker_id, file_id, system_id, null_field, label = parts[:5]

                # Only process eval files
                if file_id.startswith("LA_E"):
                    protocol_data[file_id] = {
                        "label": 0 if label == "bonafide" else 1,
                        "speaker_id": speaker_id,
                        "system_id": system_id,
                    }

    return protocol_data


def grade_single_submission(csv_path, protocol_data):
    """Grade a single submission CSV file"""
    try:
        # Load predictions
        predictions_df = pd.read_csv(csv_path)

        # Validate CSV format
        required_columns = ["file_id", "score"]
        if not all(col in predictions_df.columns for col in required_columns):
            print(f"ERROR: CSV must have columns: {required_columns}")
            print(f"Found columns: {list(predictions_df.columns)}")
            return None, None

        # Match predictions with protocol
        bonafide_scores = []
        spoof_scores = []
        matched_files = 0

        for _, row in predictions_df.iterrows():
            file_id = row["file_id"]
            score = row["score"]

            if file_id in protocol_data:
                matched_files += 1
                label = protocol_data[file_id]["label"]

                if label == 0:  # bonafide
                    bonafide_scores.append(score)
                else:  # spoof
                    spoof_scores.append(score)

        if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
            print(f"ERROR: No bonafide or spoof samples found!")
            return None, None

        # Compute EER
        eer = compute_eer(np.array(bonafide_scores), np.array(spoof_scores))

        # Compute grade according to README.md
        if eer > 9.5:
            grade = 0
        elif 5.3 <= eer <= 9.5:
            # Linear interpolation: P=4 at EER=9.5, P=10 at EER=5.3
            grade = 4 + (9.5 - eer) * (10 - 4) / (9.5 - 5.3)
        else:  # eer < 5.3
            grade = 10

        print(f"Matched files: {matched_files}/{len(predictions_df)}")
        print(f"Bonafide samples: {len(bonafide_scores)}")
        print(f"Spoof samples: {len(spoof_scores)}")
        print(f"EER: {eer:.2f}%")
        print(f"Performance grade (P): {grade:.1f}/10")

        return eer, grade

    except Exception as e:
        print(f"ERROR processing {csv_path}: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Grade ASVspoof 2019 submissions")
    parser.add_argument(
        "--protocol",
        default="ASVspoof2019.LA.cm.eval.trl.txt",
        help="Path to evaluation protocol file",
    )
    parser.add_argument(
        "--submissions_dir",
        default="students_solutions",
        help="Directory with CSV submission files",
    )

    args = parser.parse_args()

    # Load protocol
    protocol_path = Path(args.protocol)
    if not protocol_path.exists():
        print(f"ERROR: Protocol file not found: {protocol_path}")
        print("Please ensure the protocol file is in the same directory as grading.py")
        return

    print(f"Loading protocol from: {protocol_path}")
    protocol_data = load_protocol(protocol_path)
    print(f"Loaded {len(protocol_data)} evaluation files from protocol")

    # Find submission files
    submissions_dir = Path(args.submissions_dir)
    if not submissions_dir.exists():
        print(f"ERROR: Submissions directory not found: {submissions_dir}")
        print("Please create the directory and place CSV files inside")
        return

    csv_files = list(submissions_dir.glob("*.csv"))
    if not csv_files:
        print(f"ERROR: No CSV files found in {submissions_dir}")
        return

    print(f"\nFound {len(csv_files)} submission files:")

    results = []
    for csv_path in sorted(csv_files):
        print(f"\n{'=' * 50}")
        print(f"Grading: {csv_path.name}")
        print("=" * 50)

        eer, grade = grade_single_submission(csv_path, protocol_data)

        if eer is not None and grade is not None:
            results.append({"filename": csv_path.name, "eer": eer, "grade": grade})

    # Summary
    if results:
        print(f"\n{'=' * 50}")
        print("SUMMARY")
        print("=" * 50)

        for result in results:
            print(
                f"{result['filename']}: EER={result['eer']:.2f}%, Grade={result['grade']:.1f}/10"
            )

        # Best result
        best_result = min(results, key=lambda x: x["eer"])
        print(f"\nBest EER: {best_result['eer']:.2f}% ({best_result['filename']})")


if __name__ == "__main__":
    main()
