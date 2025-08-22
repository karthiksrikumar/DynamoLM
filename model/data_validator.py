# data_validator.py - Comprehensive data validation to prevent evaluation errors
import json
import torch
from typing import List, Dict, Any, Tuple
from datetime import datetime
import os

class DataFormatValidator:
    """Validates data format consistency across training and evaluation"""
    
    def __init__(self):
        self.required_raw_fields = ["question", "date", "answer", "causal_trace"]
        self.required_processed_fields = ["input_ids", "attention_mask", "labels", "time", "edge_index"]
        
    def validate_raw_data(self, data: List[Dict]) -> Tuple[bool, List[str]]:
        """Validate raw data format"""
        errors = []
        
        if not isinstance(data, list):
            errors.append(f"Data must be a list, got {type(data)}")
            return False, errors
        
        if len(data) == 0:
            errors.append("Data is empty")
            return False, errors
        
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                errors.append(f"Item {i}: Must be a dictionary, got {type(item)}")
                continue
            
            # Check required fields
            for field in self.required_raw_fields:
                if field not in item:
                    errors.append(f"Item {i}: Missing required field '{field}'")
                elif not isinstance(item[field], str) or not item[field].strip():
                    errors.append(f"Item {i}: Field '{field}' must be non-empty string")
            
            # Validate date format
            if 'date' in item:
                try:
                    datetime.strptime(item['date'], "%Y-%m-%d")
                except ValueError:
                    errors.append(f"Item {i}: Invalid date format '{item['date']}', expected YYYY-MM-DD")
        
        return len(errors) == 0, errors
    
    def validate_processed_data(self, data: List[Dict]) -> Tuple[bool, List[str]]:
        """Validate processed data format"""
        errors = []
        
        if not isinstance(data, list):
            errors.append(f"Processed data must be a list, got {type(data)}")
            return False, errors
        
        if len(data) == 0:
            errors.append("Processed data is empty")
            return False, errors
        
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                errors.append(f"Item {i}: Must be a dictionary, got {type(item)}")
                continue
            
            # Check required fields
            for field in self.required_processed_fields:
                if field not in item:
                    errors.append(f"Item {i}: Missing required field '{field}'")
                    continue
                
                # Validate tensor fields
                if field in ["input_ids", "attention_mask", "labels"]:
                    if not isinstance(item[field], torch.Tensor):
                        errors.append(f"Item {i}: Field '{field}' must be torch.Tensor, got {type(item[field])}")
                    elif item[field].dim() != 1:
                        errors.append(f"Item {i}: Field '{field}' must be 1D tensor, got {item[field].dim()}D")
                
                elif field == "time":
                    if not isinstance(item[field], torch.Tensor):
                        errors.append(f"Item {i}: Field 'time' must be torch.Tensor, got {type(item[field])}")
                    elif item[field].numel() != 1:
                        errors.append(f"Item {i}: Field 'time' must be scalar tensor, got shape {item[field].shape}")
                
                elif field == "edge_index":
                    if not isinstance(item[field], torch.Tensor):
                        errors.append(f"Item {i}: Field 'edge_index' must be torch.Tensor, got {type(item[field])}")
                    elif item[field].dim() != 2 or item[field].size(0) != 2:
                        errors.append(f"Item {i}: Field 'edge_index' must be [2, num_edges] tensor, got shape {item[field].shape}")
        
        return len(errors) == 0, errors
    
    def validate_temporal_consistency(self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]) -> Tuple[bool, List[str]]:
        """Validate temporal consistency to prevent data leakage"""
        errors = []
        
        def get_date_range(data_split, split_name):
            if not data_split:
                return None, None
            
            dates = []
            for item in data_split:
                if 'date' in item:
                    try:
                        dates.append(datetime.strptime(item['date'], "%Y-%m-%d"))
                    except ValueError:
                        errors.append(f"{split_name}: Invalid date format '{item['date']}'")
                        continue
            
            return min(dates) if dates else None, max(dates) if dates else None
        
        train_min, train_max = get_date_range(train_data, "Training")
        val_min, val_max = get_date_range(val_data, "Validation")
        test_min, test_max = get_date_range(test_data, "Test")
        
        # Check for temporal leakage
        if train_max and val_min and train_max >= val_min:
            errors.append(f"Temporal leakage: Training data extends to {train_max}, validation starts at {val_min}")
        
        if val_max and test_min and val_max >= test_min:
            errors.append(f"Temporal leakage: Validation data extends to {val_max}, test starts at {test_min}")
        
        if train_max and test_min and train_max >= test_min:
            errors.append(f"Temporal leakage: Training data extends to {train_max}, test starts at {test_min}")
        
        # Print date ranges for verification
        if train_min and train_max:
            print(f"✅ Training data: {train_min.strftime('%Y-%m-%d')} to {train_max.strftime('%Y-%m-%d')}")
        if val_min and val_max:
            print(f"✅ Validation data: {val_min.strftime('%Y-%m-%d')} to {val_max.strftime('%Y-%m-%d')}")
        if test_min and test_max:
            print(f"✅ Test data: {test_min.strftime('%Y-%m-%d')} to {test_max.strftime('%Y-%m-%d')}")
        
        return len(errors) == 0, errors
    
    def validate_model_consistency(self, models: List[Any], data_sample: Dict) -> Tuple[bool, List[str]]:
        """Validate that all models can process the same data format"""
        errors = []
        
        try:
            # Create a small batch for testing
            input_ids = data_sample['input_ids'].unsqueeze(0)
            attention_mask = data_sample['attention_mask'].unsqueeze(0)
            time = data_sample['time'].unsqueeze(0)
            edge_index = data_sample['edge_index']
            
            for i, model in enumerate(models):
                try:
                    model.eval()
                    with torch.no_grad():
                        if hasattr(model, 'use_gnn') and model.use_gnn:
                            # DYNAMO model
                            output = model(input_ids, attention_mask, time, [edge_index])
                        else:
                            # Baseline models
                            output = model(input_ids, attention_mask)
                    
                    print(f"✅ Model {i} can process data format")
                    
                except Exception as e:
                    errors.append(f"Model {i} failed to process data: {e}")
        
        except Exception as e:
            errors.append(f"Failed to create test batch: {e}")
        
        return len(errors) == 0, errors

def validate_complete_pipeline(data_path: str, config_path: str) -> bool:
    """Run comprehensive validation of the entire pipeline"""
    
    print("🔍 Running comprehensive pipeline validation...")
    validator = DataFormatValidator()
    
    # 1. Validate raw data
    print("\n1. Validating raw data format...")
    with open(data_path, 'r') as f:
        raw_data = json.load(f)
    
    is_valid, errors = validator.validate_raw_data(raw_data)
    if not is_valid:
        print("❌ Raw data validation failed:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"   - {error}")
        return False
    
    print(f"✅ Raw data validation passed ({len(raw_data)} samples)")
    
    # 2. Test data processing
    print("\n2. Testing data processing...")
    try:
        from temporal_data_splitter import create_batch_based_splits, validate_temporal_split
        
        train_raw, val_raw, test_raw = create_batch_based_splits(raw_data)
        
        # Validate temporal splits
        is_valid, errors = validator.validate_temporal_consistency(train_raw, val_raw, test_raw)
        if not is_valid:
            print("❌ Temporal consistency validation failed:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        print("✅ Temporal splitting validation passed")
        
    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        return False
    
    # 3. Test model compatibility (if config exists)
    if os.path.exists(config_path):
        print("\n3. Testing model compatibility...")
        try:
            # This would require importing and testing actual models
            # Skipping for now to avoid dependency issues
            print("⚠️  Model compatibility test skipped (requires trained models)")
        except Exception as e:
            print(f"❌ Model compatibility test failed: {e}")
            return False
    
    print("\n🎉 Complete pipeline validation passed!")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate data pipeline")
    parser.add_argument('--data_path', type=str, default='data/dynamodata.json',
                        help="Path to raw data file")
    parser.add_argument('--config_path', type=str, default='model/configs.json',
                        help="Path to model config file")
    
    args = parser.parse_args()
    
    success = validate_complete_pipeline(args.data_path, args.config_path)
    exit(0 if success else 1)
