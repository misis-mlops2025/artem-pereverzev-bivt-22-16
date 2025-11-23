#!/usr/bin/env python3
"""
Model training script for DVC pipeline.
Trains machine learning models.
"""

import json
import os
import sys

import joblib
from sklearn.metrics import accuracy_score, classification_report
import yaml

# Add project_name to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_name.configs import ModelConfig
from project_name.trainer import Trainer


def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def train_model():
    """Train machine learning model"""
    params = load_params()
    train_params = params['train']
    
    # Load preprocessed data
    X_train = joblib.load('data/preprocessed/train_features.pkl')
    y_train = joblib.load('data/preprocessed/train_targets.pkl')
    X_test = joblib.load('data/preprocessed/test_features.pkl')
    y_test = joblib.load('data/preprocessed/test_targets.pkl')
    
    # Create config for training
    config_dict = {
        'model_type': train_params['model_type'],
        'random_state': train_params['random_state'],
        'n_estimators': train_params.get('n_estimators', 100),
        'max_depth': train_params.get('max_depth', 10),
        'path_to_data': 'data/processed/train.csv',  # dummy path
        'target_column': 'label',
        'test_size': 0.2,
        'num_epoch': 1
    }
    
    # Train model
    config = ModelConfig(**config_dict)
    trainer = Trainer(config)
    
    # Use the loaded data directly instead of loading from file
    trainer.model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = trainer.model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save model and metadata
    os.makedirs('models', exist_ok=True)
    joblib.dump(trainer.model, 'models/trained_model.pkl')
    
    model_metadata = {
        'model_type': train_params['model_type'],
        'accuracy': float(accuracy),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'feature_count': X_train.shape[1],
        'parameters': train_params
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    print("Model training completed:")
    print(f"  - Model type: {train_params['model_type']}")
    print(f"  - Test accuracy: {accuracy:.4f}")
    print("  - Model saved to: models/trained_model.pkl")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_model()