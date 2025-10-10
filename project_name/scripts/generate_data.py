#!/usr/bin/env python3
"""
Synthetic data generation script for DVC pipeline.
Generates sample data for the ML pipeline.
"""

import pandas as pd
import numpy as np
import yaml
import os

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def generate_synthetic_data():
    """Generate synthetic data for the ML pipeline"""
    params = load_params()
    data_params = params.get('data', {})
    
    # Set random seed for reproducibility
    np.random.seed(data_params.get('seed', 42))
    
    # Generate parameters
    n_samples = data_params.get('n_samples', 1000)
    noise_level = data_params.get('noise_level', 0.1)
    
    print(f"Generating synthetic data with {n_samples} samples...")
    
    # Generate synthetic features
    # Feature 1: Normal distribution
    feature1 = np.random.normal(0, 1, n_samples)
    
    # Feature 2: Normal distribution with different parameters
    feature2 = np.random.normal(5, 2, n_samples)
    
    # Feature 3: Categorical feature
    feature3 = np.random.choice(['A', 'B', 'C'], n_samples, p=[0.5, 0.3, 0.2])
    
    # Feature 4: Binary categorical feature
    feature4 = np.random.choice(['X', 'Y'], n_samples, p=[0.6, 0.4])
    
    # Generate target variable based on features with some noise
    # Simple rule: higher feature1 + feature2 values more likely to be class 1
    feature_combined = feature1 + feature2/5
    probabilities = 1 / (1 + np.exp(-feature_combined))  # Sigmoid function
    
    # Add noise
    probabilities = np.clip(probabilities + np.random.normal(0, noise_level, n_samples), 0, 1)
    
    # Generate binary labels
    labels = (probabilities > 0.5).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'feature4': feature4,
        'label': labels
    })
    
    # Save raw data
    os.makedirs('data/raw', exist_ok=True)
    data.to_csv('data/raw/synthetic_data.csv', index=False)
    
    # Calculate and print dataset statistics
    class_distribution = data['label'].value_counts().sort_index()
    
    print(f"Synthetic data generation completed:")
    print(f"  - Total samples: {len(data)}")
    print(f"  - Features: {list(data.columns)}")
    print(f"  - Class distribution:")
    for class_label, count in class_distribution.items():
        percentage = count / len(data) * 100
        print(f"    - Class {class_label}: {count} samples ({percentage:.1f}%)")
    
    print(f"  - Data saved to: data/raw/synthetic_data.csv")
    
    return data

if __name__ == "__main__":
    generate_synthetic_data()