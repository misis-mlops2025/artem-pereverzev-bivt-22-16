#!/usr/bin/env python3
"""
Prediction script for DVC pipeline.
Generates predictions using trained model.
"""

import pandas as pd
import numpy as np
import joblib
import yaml
import os
import sys

# Add project_name to path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_name.processing import DataPreprocessor

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def generate_predictions():
    """Generate predictions using trained model"""
    params = load_params()
    
    # Load model and preprocessor
    model = joblib.load('models/trained_model.pkl')
    preprocessor = DataPreprocessor.load('models/preprocessor.pkl')
    
    # Create sample new data for prediction
    # In real scenario, you'd load from data/raw/new_data.csv
    np.random.seed(42)
    n_samples = 50
    
    new_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(5, 2, n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature4': np.random.choice(['X', 'Y'], n_samples)
    })
    
    # Preprocess new data
    new_data_processed = preprocessor.transform(new_data)
    
    # Generate predictions
    predictions = model.predict(new_data_processed)
    prediction_probas = model.predict_proba(new_data_processed) if hasattr(model, 'predict_proba') else None
    
    # Create results DataFrame
    results = new_data.copy()
    results['prediction'] = predictions
    
    if prediction_probas is not None:
        results['probability_class_0'] = prediction_probas[:, 0]
        results['probability_class_1'] = prediction_probas[:, 1]
    
    # Save predictions
    os.makedirs('data/predictions', exist_ok=True)
    results.to_csv('data/predictions/predictions.csv', index=False)
    
    print(f"Predictions generated successfully:")
    print(f"  - Samples predicted: {len(results)}")
    print(f"  - Predictions saved to: data/predictions/predictions.csv")
    
    # Show prediction distribution
    prediction_counts = results['prediction'].value_counts().sort_index()
    print(f"\nPrediction distribution:")
    for class_label, count in prediction_counts.items():
        print(f"  - Class {class_label}: {count} samples ({count/len(results)*100:.1f}%)")

if __name__ == "__main__":
    generate_predictions()