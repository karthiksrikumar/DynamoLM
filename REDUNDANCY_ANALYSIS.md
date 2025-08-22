# Codebase Redundancy Analysis

After fixing the data leakage issues, here's a comprehensive analysis of redundant, unnecessary, or conflicting files and functions in the codebase:

## 🚨 **Critical Redundancies - Causing Conflicts**

### 1. **Duplicate Model Implementations**
- **`model/model.py`** vs **`model/fullpipeline.py`** (lines 261-404)
  - Both contain `DynamoModel` class with similar but **inconsistent** implementations
  - `model.py` has more sophisticated architecture with proper fusion layers
  - `fullpipeline.py` has simplified version with potential bugs
  - **Recommendation**: **DELETE** `DynamoModel` from `fullpipeline.py`, import from `model.py`

### 2. **Duplicate Tokenizer Classes**
- **`model/tokenizer_utils.py`** vs **`model/fullpipeline.py`** (lines 42-99)
  - Both contain `DynamoTokenizer` class with **identical** functionality
  - **Recommendation**: **DELETE** `DynamoTokenizer` from `fullpipeline.py`, import from `tokenizer_utils.py`

### 3. **Duplicate Training Logic**
- **`model/train.py`** vs **`model/fullpipeline.py`** (lines 408-508)
  - Both contain training functions with different interfaces
  - `train.py` has more sophisticated `DynamoTrainer` class
  - `fullpipeline.py` has simplified `train_model` function
  - **Recommendation**: **DELETE** `train_model` from `fullpipeline.py`, use `DynamoTrainer`

### 4. **Duplicate Data Processing**
- **`model/processing.py`** vs **`model/fullpipeline.py`** (lines 104-142)
  - Both contain causal trace parsing and data loading
  - `processing.py` has validation functions not used elsewhere
  - **Recommendation**: **DELETE** redundant functions from `fullpipeline.py`

## 🔧 **Redundant Utility Files**

### 5. **Pipeline Runners**
- **`model/run_pipeline.py`** vs **`model/fullpipeline.py`**
  - `run_pipeline.py` is a wrapper that calls other scripts via subprocess
  - `fullpipeline.py` is a self-contained implementation
  - Both serve the same purpose but with different approaches
  - **Recommendation**: **KEEP** `fullpipeline.py` (more maintainable), **DELETE** `run_pipeline.py`

### 6. **Causal Extraction Duplicates**
- **`causal_graph_extraction/causal_extractor.py`** vs **`causal_graph_extraction/evals/causalbankeval.py`** (lines 27-48)
  - Both contain `CausalExtractor` classes with **different** implementations
  - `causal_extractor.py` has sophisticated `ImprovedCausalExtractor` (440 lines)
  - `causalbankeval.py` has simple `CausalExtractor` (22 lines)
  - **Recommendation**: **DELETE** simple version from `causalbankeval.py`, use improved version

### 7. **Data Processing Duplicates**
- **`causal_graph_extraction/data_processor.py`** vs **`causal_graph_extraction/evals/causalbankeval.py`** (lines 74-91)
  - Both contain `DataProcessor` classes with similar functionality
  - **Recommendation**: **DELETE** duplicate from `causalbankeval.py`

## 📊 **Unused/Incomplete Files**

### 8. **Incomplete Scaling Experiments**
- **`evals/scaling.py`**
  - References non-existent `test_dataset.py` (line 8)
  - Calls undefined `train_model` function with wrong signature (line 31)
  - **Recommendation**: **DELETE** or **REWRITE** completely

### 9. **Broken Plot Scripts**
- **`evals/plots/accuracy_vs_time.py`**
  - Hardcoded time points that don't match actual data
  - Assumes results format that doesn't exist
  - **Recommendation**: **DELETE** or rewrite to use actual results

- **`evals/plots/accvmodlsize.py`**
  - Assumes scaling results that aren't generated
  - **Recommendation**: **DELETE** until scaling experiments are fixed

### 10. **Unused Utility Functions**

#### In `model/utils.py`:
- `calculate_bleu_score()` - Never used, requires NLTK
- `evaluate_predictions()` - Never used
- `EarlyStopping` class - Never used
- `create_output_directory()` - Never used
- `setup_logging()` - Never used
- `cleanup_checkpoints()` - Never used
- **Recommendation**: **DELETE** unused functions, keep only core utilities

#### In `evals/metrics.py`:
- `compute_f1()` - Marked as "not used in Test 2"
- `compute_causal_accuracy()` - Marked as "not used in Test 2"
- **Recommendation**: **DELETE** unused functions

## 🔄 **Inconsistent Implementations**

### 11. **Multiple Model Interfaces**
- **`model/model.py`** - Sophisticated `DynamoModel` with proper config
- **`model/fullpipeline.py`** - Simplified `DynamoModel` with hardcoded values
- **`evals/evaluate_dynamo.py`** - Expects yet another interface
- **Recommendation**: Standardize on `model/model.py` implementation

### 12. **Inconsistent Data Formats**
- Raw JSON format in `data/dynamodata.json`
- Processed format expected by `evals/evaluate_dynamo.py`
- Different processing in `model/fullpipeline.py`
- **Recommendation**: Use standardized processing from new `temporal_data_splitter.py`

## 📋 **Recommended File Actions**

### **DELETE These Files:**
1. `model/run_pipeline.py` - Replaced by improved `fullpipeline.py`
2. `evals/scaling.py` - Broken and incomplete
3. `evals/plots/accuracy_vs_time.py` - Hardcoded and broken
4. `evals/plots/accvmodlsize.py` - Depends on non-existent results

### **CLEAN UP These Files:**
1. **`model/fullpipeline.py`**:
   - Remove duplicate `DynamoTokenizer` class (lines 42-99)
   - Remove duplicate `DynamoModel` class (lines 261-404)
   - Remove duplicate `train_model` function (lines 408-508)
   - Import from proper modules instead

2. **`model/utils.py`**:
   - Remove unused functions: `calculate_bleu_score`, `evaluate_predictions`, `EarlyStopping`, `create_output_directory`, `setup_logging`, `cleanup_checkpoints`
   - Keep only: `get_linear_scheduler`, `count_parameters`, `save_training_plot`, `load_json_config`, `save_json_config`, `format_time`, `format_number`, `get_device`, `print_gpu_memory`, `calculate_accuracy`

3. **`evals/metrics.py`**:
   - Remove unused functions: `compute_f1`, `compute_causal_accuracy`
   - Keep only: `compute_accuracy`, `compute_drift`

4. **`causal_graph_extraction/evals/causalbankeval.py`**:
   - Remove duplicate classes: `CausalExtractor`, `TemporalGraph`, `DataProcessor`
   - Import from proper modules instead

### **KEEP These Files (Core Functionality):**
1. `model/model.py` - Main model architecture
2. `model/train.py` - Sophisticated training framework
3. `model/inference.py` - Inference functionality
4. `model/tokenizer_utils.py` - Tokenization utilities
5. `model/processing.py` - Data validation (after cleanup)
6. `evals/evaluate_dynamo.py` - Main evaluation script (after fixes)
7. `evals/metrics.py` - Core metrics (after cleanup)
8. `model/temporal_data_splitter.py` - NEW: Proper temporal splitting
9. `model/reproducibility.py` - NEW: Reproducibility controls
10. `model/data_validator.py` - NEW: Data validation

## 📈 **Estimated Code Reduction**
- **Before**: ~2,000 lines across 15+ files
- **After cleanup**: ~1,200 lines across 10 files
- **Reduction**: ~40% less code, much better maintainability

## 🎯 **Next Steps**
1. Implement the deletions and cleanups above
2. Update imports in remaining files
3. Test the cleaned pipeline
4. Update documentation to reflect new structure

This cleanup will eliminate the data leakage issues and make the codebase much more maintainable and reliable.
