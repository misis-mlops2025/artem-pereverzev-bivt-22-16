# DVC Pipeline for Machine Learning Project

This project implements a complete DVC (Data Version Control) pipeline for machine learning workflows.

## Pipeline Overview

The DVC pipeline consists of the following stages:

### 1. Prepare Stage
- **Script**: `scripts/prepare_data.py`
- **Purpose**: Creates train/test splits from raw data
- **Dependencies**: Raw data files, preparation script
- **Outputs**: Processed train/test CSV files, feature metadata

### 2. Preprocess Stage
- **Script**: `scripts/preprocess_data.py`
- **Purpose**: Applies feature engineering and preprocessing
- **Dependencies**: Processed data, preprocessing script, processing module
- **Outputs**: Preprocessed features/targets, fitted preprocessor

### 3. Train Stage
- **Script**: `scripts/train_model.py`
- **Purpose**: Trains machine learning models
- **Dependencies**: Preprocessed data, training script, trainer module
- **Outputs**: Trained model, model metadata

### 4. Evaluate Stage
- **Script**: `scripts/evaluate_model.py`
- **Purpose**: Evaluates model performance
- **Dependencies**: Trained model, test data
- **Outputs**: Performance metrics, confusion matrix, ROC curve data

### 5. Predict Stage
- **Script**: `scripts/predict.py`
- **Purpose**: Generates predictions using trained model
- **Dependencies**: Trained model, preprocessor, new data
- **Outputs**: Prediction results

## Key Files

### Configuration Files
- **`dvc.yaml`**: Defines the pipeline stages and dependencies
- **`params.yaml`**: Contains all configurable parameters
- **`.dvcignore`**: Specifies files to ignore in DVC tracking

### Scripts
- **`scripts/prepare_data.py`**: Data preparation and splitting
- **`scripts/preprocess_data.py`**: Feature preprocessing
- **`scripts/train_model.py`**: Model training
- **`scripts/evaluate_model.py`**: Model evaluation
- **`scripts/predict.py`**: Prediction generation

## How to Use

### 1. Initialize DVC (if not already done)
```bash
dvc init
```

### 2. Run the Complete Pipeline
```bash
dvc repro
```

### 3. Run Specific Stages
```bash
# Run only the prepare stage
dvc repro prepare

# Run prepare and preprocess stages
dvc repro preprocess
```

### 4. View Pipeline Status
```bash
# Show pipeline stages
dvc dag

# Show pipeline status
dvc status

# Show metrics
dvc metrics show
```

### 5. Update Parameters and Re-run
```bash
# Edit parameters in params.yaml
vim params.yaml

# Reproduce pipeline with new parameters
dvc repro
```

### 6. Version Control
```bash
# Add DVC files to git
git add dvc.yaml params.yaml .dvcignore .dvc/

# Commit changes
git commit -m "Add DVC pipeline"

# Push to remote
git push
```

## Pipeline Parameters

All configurable parameters are stored in `params.yaml`:

```yaml
prepare:
  seed: 42                    # Random seed for reproducibility
  test_size: 0.2             # Test set size

preprocess:
  scaler_type: "robust"      # Type of scaler for numeric features
  handle_unknown: "ignore"   # How to handle unknown categories

train:
  model_type: "random_forest"  # Type of model to train
  random_state: 42            # Random seed for model
  n_estimators: 100          # Number of trees for random forest
  max_depth: 10              # Maximum tree depth

predict:
  batch_size: 32             # Batch size for predictions
```

## Benefits of This DVC Pipeline

1. **Reproducibility**: Every run is tracked and reproducible
2. **Parameter Management**: All configurable parameters are centralized
3. **Dependency Tracking**: Automatic detection of changes in dependencies
4. **Caching**: Only changed stages are re-executed
5. **Metrics Tracking**: Performance metrics are automatically tracked
6. **Collaboration**: Easy to share and collaborate on ML projects

## Example Commands

```bash
# Run the entire pipeline
dvc repro

# Show pipeline visualization
dvc dag | dot -Tpng -o pipeline.png

# Compare metrics between runs
dvc metrics diff

# Plot metrics over time
dvc plots diff
```

## Troubleshooting

- If DVC commands fail, ensure DVC is installed: `pip install dvc`
- For remote storage setup: `dvc remote add -d myremote /path/to/storage`
- To reset pipeline: `dvc destroy` then `dvc repro`

## Next Steps

1. Set up remote storage for data and models
2. Configure CI/CD for automated pipeline execution
3. Add more sophisticated data validation
4. Implement hyperparameter optimization
5. Add model monitoring and retraining workflows