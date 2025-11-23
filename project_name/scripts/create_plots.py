#!/usr/bin/env python3
"""
Create plots from metrics for DVC pipeline.
Generates visualizations based on evaluation metrics.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def create_plots():
    """Create plots from evaluation metrics"""
    
    # Load metrics
    with open('reports/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    # Load confusion matrix data
    with open('reports/plots/confusion_matrix.json', 'r') as f:
        cm_data = json.load(f)
    
    # Load ROC curve data
    with open('reports/plots/roc_curve.json', 'r') as f:
        roc_data = json.load(f)
    
    # Create plots directory
    os.makedirs('reports/plots', exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Plot 1: Performance Metrics Bar Chart
    plt.figure(figsize=(10, 6))
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'], 
        metrics['recall'],
        metrics['f1_score'],
        metrics.get('roc_auc', 0)
    ]
    
    bars = plt.bar(metric_names, metric_values, color=sns.color_palette("husl", len(metric_names)))
    plt.ylim(0, 1)
    plt.title('Model Performance Metrics', fontsize=16, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('reports/plots/performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    cm = np.array(cm_data['matrix'])
    labels = cm_data['labels']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('reports/plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: ROC Curve
    if roc_data:
        plt.figure(figsize=(8, 6))
        fpr = roc_data['fpr']
        tpr = roc_data['tpr']
        roc_auc = roc_data['auc']
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('reports/plots/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 4: Metrics Comparison (if we had multiple runs)
    plt.figure(figsize=(10, 6))
    
    # For demonstration, create a simple metrics comparison
    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1']
    values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]
    
    # Create a radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics_list), endpoint=False).tolist()
    values += values[:1]  # Complete the circle
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=2, label='Current Model')
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics_list)
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Radar Chart', size=16, fontweight='bold', y=1.08)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig('reports/plots/metrics_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Plots created successfully:")
    print("  - reports/plots/performance_metrics.png")
    print("  - reports/plots/confusion_matrix.png") 
    print("  - reports/plots/roc_curve.png")
    print("  - reports/plots/metrics_radar.png")

if __name__ == "__main__":
    create_plots()