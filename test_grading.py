#!/usr/bin/env python3
"""
Test script to verify grading setup works correctly
"""

from pathlib import Path

import numpy as np
import pandas as pd


def create_test_submission():
    """Create a test CSV file for grading verification"""

    # Create mock predictions (random scores)
    np.random.seed(42)

    # Generate realistic file IDs for eval set
    file_ids = []
    for i in range(100):  # Small test set
        file_ids.append(f"LA_E_{1000000 + i}")

    # Generate random scores (probabilities)
    scores = np.random.uniform(0.1, 0.9, len(file_ids))

    # Create DataFrame
    test_df = pd.DataFrame({"file_id": file_ids, "score": scores})

    # Save test CSV
    test_dir = Path("students_solutions")
    test_dir.mkdir(exist_ok=True)

    test_csv_path = test_dir / "test_submission.csv"
    test_df.to_csv(test_csv_path, index=False)

    print(f"✅ Test CSV created: {test_csv_path}")
    print(f"📊 Shape: {test_df.shape}")
    print(f"🔍 Preview:")
    print(test_df.head())

    return test_csv_path


def main():
    """Test the grading pipeline"""

    print("🧪 Testing grading pipeline...")

    # Create test submission
    test_csv = create_test_submission()

    # Check if protocol file exists
    protocol_file = Path("ASVspoof2019.LA.cm.eval.trl.txt")
    if not protocol_file.exists():
        print(f"⚠️  Protocol file not found: {protocol_file}")
        print("Please download it from ASVspoof 2019 dataset")
        print("Place it in the same directory as grading.py")
        return

    # Instructions for manual testing
    print(f"\n✅ Setup complete!")
    print(f"📁 Test file created: {test_csv}")
    print(f"📄 Protocol file: {protocol_file}")

    print(f"\n🔧 To test grading, run:")
    print(
        f"python grading.py --protocol {protocol_file} --submissions_dir students_solutions"
    )

    print(f"\n📝 Expected output:")
    print(f"- EER calculation")
    print(f"- Performance grade (P)")
    print(f"- File matching statistics")


if __name__ == "__main__":
    main()
