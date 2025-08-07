#!/usr/bin/env python3
"""
Evaluation script that creates properly formatted CSV for submission
"""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve

from src.datasets.data_utils import get_dataloaders
from src.utils.init_utils import set_random_seed


def compute_eer(bonafide_scores, spoof_scores):
    """Same EER function as in grading.py"""
    scores = np.concatenate([bonafide_scores, spoof_scores])
    labels = np.concatenate(
        [np.ones(len(bonafide_scores)), np.zeros(len(spoof_scores))]
    )
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        return eer * 100
    except:
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.absolute(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        return eer * 100


@hydra.main(version_base=None, config_path="src/configs", config_name="asvspoof")
def main(config):
    """
    Evaluate model and create submission CSV
    """
    set_random_seed(config.trainer.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data and model
    dataloaders, batch_transforms = get_dataloaders(config, device)
    model = instantiate(config.model).to(device)

    # Load best checkpoint
    checkpoint_path = Path("saved") / config.writer.run_name / "model_best.pth"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded model from: {checkpoint_path}")
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint_path}")
        print("Using untrained model!")

    model.eval()

    # Get test dataloader
    test_loader = dataloaders.get("test")
    if test_loader is None:
        raise ValueError("Test dataloader not found! Check config.")

    print("Starting evaluation on test set...")

    # Collect predictions
    all_predictions = []
    all_file_ids = []
    bonafide_scores = []
    spoof_scores = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move to device
            batch["data_object"] = batch["data_object"].to(device)
            batch["labels"] = batch["labels"].to(device)

            # Forward pass
            outputs = model(**batch)
            logits = outputs["logits"]

            # Get probabilities (score = probability of being bonafide)
            probs = torch.softmax(logits, dim=-1)
            bonafide_probs = probs[:, 0].cpu().numpy()  # P(bonafide)

            # Get labels and file IDs
            labels = batch["labels"].cpu().numpy()

            # Get file IDs from dataset
            batch_start_idx = batch_idx * test_loader.batch_size
            batch_end_idx = min(
                batch_start_idx + len(bonafide_probs), len(test_loader.dataset)
            )

            for i in range(len(bonafide_probs)):
                dataset_idx = batch_start_idx + i
                if dataset_idx < len(test_loader.dataset):
                    # Get file_id from dataset index
                    sample_info = test_loader.dataset._index[dataset_idx]
                    file_id = sample_info.get("file_id", f"file_{dataset_idx}")

                    score = bonafide_probs[i]
                    label = labels[i]

                    all_predictions.append(score)
                    all_file_ids.append(file_id)

                    # Separate by class for EER calculation
                    if label == 0:  # bonafide
                        bonafide_scores.append(score)
                    else:  # spoof
                        spoof_scores.append(score)

            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {(batch_idx + 1) * test_loader.batch_size} samples")

    print(f"Total predictions: {len(all_predictions)}")
    print(f"Bonafide samples: {len(bonafide_scores)}")
    print(f"Spoof samples: {len(spoof_scores)}")

    # Calculate EER
    if len(bonafide_scores) > 0 and len(spoof_scores) > 0:
        eer = compute_eer(np.array(bonafide_scores), np.array(spoof_scores))
        print(f"EER: {eer:.2f}%")

        # Calculate grade
        if eer > 9.5:
            grade = 0
        elif 5.3 <= eer <= 9.5:
            grade = 4 + (9.5 - eer) * (10 - 4) / (9.5 - 5.3)
        else:
            grade = 10
        print(f"Expected grade: {grade:.1f}/10")

    # Create submission CSV
    submission_df = pd.DataFrame({"file_id": all_file_ids, "score": all_predictions})

    # Save CSV (IMPORTANT: use your HSE username!)
    csv_filename = "bduvarov.csv"  # ⚠️ REPLACE WITH YOUR HSE USERNAME!
    submission_df.to_csv(csv_filename, index=False)

    print(f"\n✅ Submission CSV saved: {csv_filename}")
    print(f"📊 Total samples: {len(submission_df)}")
    print(f"🎯 EER: {eer:.2f}%")
    print(f"📈 Expected grade: {grade:.1f}/10")

    # Verify CSV format
    print(f"\n🔍 CSV Preview:")
    print(submission_df.head())
    print(f"\nColumns: {list(submission_df.columns)}")
    print(f"Shape: {submission_df.shape}")

    return eer


if __name__ == "__main__":
    main()
