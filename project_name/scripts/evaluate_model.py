#!/usr/bin/env python3
"""
Model evaluation script for DVC pipeline.
Evaluates model performance and generates metrics.
"""

import json
import os

import joblib
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def evaluate_model():
    """Evaluate model performance"""
    
    # Load model and test data
    model = joblib.load('models/trained_model.pkl')
    X_test = joblib.load('data/preprocessed/test_features.pkl')
    y_test = joblib.load('data/preprocessed/test_targets.pkl')
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Calculate ROC curve if probabilities are available
    roc_data = {}
    if y_pred_proba is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        roc_data = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': float(roc_auc)
        }
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'test_samples': len(y_test)
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = float(roc_auc)
    
    os.makedirs('reports', exist_ok=True)
    os.makedirs('reports/plots', exist_ok=True)
    
    with open('reports/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save confusion matrix for plotting
    cm_data = {
        'matrix': cm.tolist(),
        'labels': ['Class 0', 'Class 1']
    }
    
    with open('reports/plots/confusion_matrix.json', 'w') as f:
        json.dump(cm_data, f, indent=2)
    
    # Save ROC curve data
    if roc_data:
        with open('reports/plots/roc_curve.json', 'w') as f:
            json.dump(roc_data, f, indent=2)
    
    print("Model evaluation completed:")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - F1 Score: {f1:.4f}")
    
    if y_pred_proba is not None:
        print(f"  - ROC AUC: {roc_auc:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nMetrics saved to: reports/metrics.json")

if __name__ == "__main__":
    evaluate_model()