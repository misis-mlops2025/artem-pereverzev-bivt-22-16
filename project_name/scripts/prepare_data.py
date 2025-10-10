#!/usr/bin/env python3
"""
Data preparation script for DVC pipeline.
Creates train/test splits from raw data.
"""

import pandas as pd
import numpy as np
import json
import yaml
from sklearn.model_selection import train_test_split
import os

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def prepare_data():
    """Prepare data for training and testing"""
    params = load_params()
    prepare_params = params['prepare']
    
    # Create sample data (in real scenario, you'd load from data/raw/)
    np.random.seed(prepare_params['seed'])
    n_samples = 1000
    
    # Generate synthetic data
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(5, 2, n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature4': np.random.choice(['X', 'Y'], n_samples),
        'label': np.random.choice([0, 1], n_samples)
    })
    
    # Split into train and test
    train_df, test_df = train_test_split(
        data, 
        test_size=prepare_params['test_size'],
        random_state=prepare_params['seed'],
        stratify=data['label']
    )
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    
    # Save feature names
    feature_names = {
        'numeric_features': ['feature1', 'feature2'],
        'categorical_features': ['feature3', 'feature4'],
        'target': 'label'
    }
    
    with open('data/processed/feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    print(f"Data prepared successfully:")
    print(f"  - Train samples: {len(train_df)}")
    print(f"  - Test samples: {len(test_df)}")
    print(f"  - Features: {list(data.columns)}")

if __name__ == "__main__":
    prepare_data()