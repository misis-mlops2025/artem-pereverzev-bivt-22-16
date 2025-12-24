"""
Binary Classification Pipeline DAG

This DAG creates a complete ML pipeline for binary classification:
1. Generate synthetic data
2. Preprocess the data
3. Train a machine learning model
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

from airflow.models import Variable



default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
}

model_folder = Variable.get("path_to_models_folder", default_var="/")
data_folder = Variable.get("path_to_train_data_folder", default_var="/")


dag = DAG(
    'binary_classification_pipeline',
    default_args=default_args,
    description='Binary Classification ML Pipeline',
    schedule=timedelta(days=1),
    catchup=False,
    tags=['ml', 'classification', 'pipeline'],
)

def generate_synthetic_data():
    """
    Generate synthetic binary classification data
    """
    print("Generating synthetic binary classification data...")
    

    np.random.seed(42)
    
    n_samples = 1000
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    
    y = (X[:, 0] + 2*X[:, 1] - 1.5*X[:, 2] + 0.8*X[:, 3] - X[:, 4] + np.random.normal(0, 0.5, n_samples)) > 0
    y = y.astype(int)
    
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    os.makedirs(data_folder, exist_ok=True)
    file_path = data_folder + 'data.csv'
    df.to_csv(file_path, index=False)
    
    print(f"Generated {len(df)} samples with {n_features} features")
    print(f"Class distribution: {df['target'].value_counts().to_dict()}")
    print(f"Data saved to: {file_path}")
    
    return file_path

def preprocess_data():
    """
    Preprocess the generated data
    """
    print("Preprocessing data...")
    
    input_path = data_folder + 'data.csv'
    df = pd.read_csv(input_path)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df['target'] = y_train.reset_index(drop=True)
    
    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df['target'] = y_test.reset_index(drop=True)
    
    os.makedirs('/tmp/data', exist_ok=True)
    train_path = data_folder + 'train_data.csv'
    test_path = data_folder + 'test_data.csv'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    scaler_path = data_folder + 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Data split: Train={len(train_df)}, Test={len(test_df)}")
    print(f"Processed data saved to: {train_path} and {test_path}")
    
    return train_path, test_path, scaler_path

def train_model():
    """
    Train a Random Forest classifier
    """
    print("Training Random Forest model...")
    
    train_path = data_folder + 'train_data.csv'
    test_path = data_folder + 'test_data.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model trained successfully!")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    model_path = model_folder + 'model.pkl'

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to: {model_path}")
    
    return model_path, accuracy

generate_data_task = PythonOperator(
    task_id='generate_synthetic_data',
    python_callable=generate_synthetic_data,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)


generate_data_task >> preprocess_task >> train_model_task