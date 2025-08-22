# temporal_data_splitter.py - Proper temporal data splitting to prevent data leakage
import json
import torch
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
import numpy as np

def date_to_timestamp(date_str: str) -> float:
    """Convert date string to Unix timestamp"""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.timestamp()
    except ValueError:
        # Fallback for invalid dates
        return datetime(2025, 1, 1).timestamp()

def create_temporal_splits(data: List[Dict], 
                          train_end_date: str = "2025-06-30",
                          val_end_date: str = "2025-09-30",
                          test_start_date: str = "2025-10-01") -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create temporal train/validation/test splits to prevent data leakage.
    
    Args:
        data: Raw data with 'date' field
        train_end_date: Last date for training data (inclusive)
        val_end_date: Last date for validation data (inclusive) 
        test_start_date: First date for test data (inclusive)
    
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    train_end_ts = date_to_timestamp(train_end_date)
    val_end_ts = date_to_timestamp(val_end_date)
    test_start_ts = date_to_timestamp(test_start_date)
    
    train_data = []
    val_data = []
    test_data = []
    
    for item in data:
        item_ts = date_to_timestamp(item['date'])
        
        if item_ts <= train_end_ts:
            train_data.append(item)
        elif item_ts <= val_end_ts:
            val_data.append(item)
        elif item_ts >= test_start_ts:
            test_data.append(item)
        # Items between val_end and test_start are dropped to create a buffer
    
    print(f"Temporal split created:")
    print(f"  Training: {len(train_data)} samples (up to {train_end_date})")
    print(f"  Validation: {len(val_data)} samples ({train_end_date} to {val_end_date})")
    print(f"  Test: {len(test_data)} samples (from {test_start_date})")
    
    return train_data, val_data, test_data

def create_batch_based_splits(data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create splits based on the batch system used in evaluate_dynamo.py
    This ensures consistency with the existing evaluation framework.
    """
    # Updated batch ranges to match actual data (2025-2026)
    batch_1_start = datetime(2025, 1, 1)   # Jan 2025
    batch_1_end = datetime(2025, 8, 31)    # Aug 2025  
    batch_2_start = datetime(2025, 9, 1)   # Sep 2025
    batch_2_end = datetime(2025, 12, 31)   # Dec 2025
    batch_3_start = datetime(2026, 1, 1)   # Jan 2026
    batch_3_end = datetime(2026, 12, 31)   # Dec 2026
    
    train_data = []  # Use Batch 1 for training
    val_data = []    # Use Batch 2 for validation
    test_data = []   # Use Batch 3 for testing
    
    for item in data:
        item_date = datetime.strptime(item['date'], "%Y-%m-%d")
        item_ts = item_date.timestamp()
        
        if batch_1_start.timestamp() <= item_ts <= batch_1_end.timestamp():
            train_data.append(item)
        elif batch_2_start.timestamp() <= item_ts <= batch_2_end.timestamp():
            val_data.append(item)
        elif batch_3_start.timestamp() <= item_ts <= batch_3_end.timestamp():
            test_data.append(item)
    
    print(f"Batch-based temporal split:")
    print(f"  Training (Batch 1): {len(train_data)} samples (Jan-Aug 2025)")
    print(f"  Validation (Batch 2): {len(val_data)} samples (Sep-Dec 2025)")
    print(f"  Test (Batch 3): {len(test_data)} samples (Jan-Dec 2026)")
    
    return train_data, val_data, test_data

def validate_temporal_split(train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]) -> bool:
    """Validate that temporal splits don't have chronological leakage"""
    
    def get_max_date(data_split):
        if not data_split:
            return None
        return max(datetime.strptime(item['date'], "%Y-%m-%d") for item in data_split)
    
    def get_min_date(data_split):
        if not data_split:
            return None
        return min(datetime.strptime(item['date'], "%Y-%m-%d") for item in data_split)
    
    train_max = get_max_date(train_data)
    val_min = get_min_date(val_data)
    val_max = get_max_date(val_data)
    test_min = get_min_date(test_data)
    
    # Check for temporal leakage
    issues = []
    
    if train_max and val_min and train_max >= val_min:
        issues.append(f"Training data extends into validation period: {train_max} >= {val_min}")
    
    if val_max and test_min and val_max >= test_min:
        issues.append(f"Validation data extends into test period: {val_max} >= {test_min}")
    
    if train_max and test_min and train_max >= test_min:
        issues.append(f"Training data extends into test period: {train_max} >= {test_min}")
    
    if issues:
        print("❌ Temporal validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("✅ Temporal validation passed - no chronological data leakage detected")
    return True

def add_temporal_metadata(data: List[Dict]) -> List[Dict]:
    """Add temporal metadata needed for evaluation consistency"""
    
    for item in data:
        # Convert date to timestamp for evaluation compatibility
        item['time'] = date_to_timestamp(item['date'])
        
        # Add split indicators (will be set by the splitting function)
        item['is_train'] = False
        item['is_val'] = False
        item['is_test'] = False
    
    return data

def create_consistent_temporal_splits(data_path: str, 
                                    output_dir: str = "data/splits",
                                    split_method: str = "batch_based") -> Dict[str, str]:
    """
    Create temporally consistent train/val/test splits and save them.
    
    Args:
        data_path: Path to raw dynamodata.json
        output_dir: Directory to save split files
        split_method: Either "batch_based" or "custom"
    
    Returns:
        Dictionary with paths to split files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load raw data
    with open(data_path, 'r') as f:
        raw_data = json.load(f)
    
    print(f"Loaded {len(raw_data)} samples from {data_path}")
    
    # Add temporal metadata
    raw_data = add_temporal_metadata(raw_data)
    
    # Create splits based on method
    if split_method == "batch_based":
        train_data, val_data, test_data = create_batch_based_splits(raw_data)
    else:
        train_data, val_data, test_data = create_temporal_splits(raw_data)
    
    # Validate splits
    if not validate_temporal_split(train_data, val_data, test_data):
        raise ValueError("Temporal split validation failed - data leakage detected!")
    
    # Mark data with split indicators
    for item in train_data:
        item['is_train'] = True
    for item in val_data:
        item['is_val'] = True
    for item in test_data:
        item['is_test'] = True
    
    # Save splits
    train_path = os.path.join(output_dir, "train_data.json")
    val_path = os.path.join(output_dir, "val_data.json")
    test_path = os.path.join(output_dir, "test_data.json")
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=2)
    with open(test_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # Also save combined data with split flags for compatibility
    all_data_with_splits = train_data + val_data + test_data
    combined_path = os.path.join(output_dir, "data_with_splits.json")
    with open(combined_path, 'w') as f:
        json.dump(all_data_with_splits, f, indent=2)
    
    print(f"✅ Temporal splits saved:")
    print(f"  Training: {train_path}")
    print(f"  Validation: {val_path}")
    print(f"  Test: {test_path}")
    print(f"  Combined: {combined_path}")
    
    return {
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'combined': combined_path
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create temporal data splits")
    parser.add_argument('--data_path', type=str, default='data/dynamodata.json',
                        help="Path to raw data file")
    parser.add_argument('--output_dir', type=str, default='data/splits',
                        help="Output directory for split files")
    parser.add_argument('--split_method', type=str, default='batch_based',
                        choices=['batch_based', 'custom'],
                        help="Splitting method to use")
    
    args = parser.parse_args()
    
    try:
        split_paths = create_consistent_temporal_splits(
            args.data_path, args.output_dir, args.split_method
        )
        print("\n🎉 Temporal data splitting completed successfully!")
        print("Use these files for consistent training and evaluation.")
        
    except Exception as e:
        print(f"❌ Failed to create temporal splits: {e}")
        import traceback
        traceback.print_exc()
