#!/usr/bin/env python3
"""
Data preprocessing script for DVC pipeline.
Applies feature engineering and preprocessing.
"""

import pandas as pd
import numpy as np
import json
import yaml
import joblib
import os
import sys

# Add project_name to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_name.processing import DataPreprocessor

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def preprocess_data():
    """Preprocess training and test data"""
    params = load_params()
    
    # Load raw data
    data = pd.read_csv('data/raw/synthetic_data.csv')

    # Split into train and test
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data['label']
    )
    # Separate features and target
    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']
    X_test = test_df.drop(columns=['label'])
    y_test = test_df['label']
    
    # Initialize and fit preprocessor
    preprocessor = DataPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Transform test data
    X_test_processed = preprocessor.transform(X_test)
    
    # Save preprocessed data
    os.makedirs('data/preprocessed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(X_train_processed, 'data/preprocessed/train_features.pkl')
    joblib.dump(y_train, 'data/preprocessed/train_targets.pkl')
    joblib.dump(X_test_processed, 'data/preprocessed/test_features.pkl')
    joblib.dump(y_test, 'data/preprocessed/test_targets.pkl')
    
    # Save preprocessor
    preprocessor.save('models/preprocessor.pkl')
    
    print(f"Data preprocessing completed:")
    print(f"  - Train features shape: {X_train_processed.shape}")
    print(f"  - Test features shape: {X_test_processed.shape}")
    print(f"  - Preprocessor saved to: models/preprocessor.pkl")

if __name__ == "__main__":
    preprocess_data()