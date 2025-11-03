"""
Batch Inference DAG

This DAG performs batch inference using the trained model from the binary classification pipeline.
It loads new data, preprocesses it, and generates predictions.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.models import Variable

import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.metrics import accuracy_score, classification_report


path_to_new_data = Variable.get("path_to_new_data", default_var="/tmp/inference_data/")
model_path = Variable.get("path_to_model", default_var="/opt/airflow/plugins/models/model.pkl")
scaler_path = Variable.get("path_to_scaler", default_var="/opt/airflow/plugins/data/scaler.pkl")

# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
}

# Define the DAG
dag = DAG(
    'batch_inference',
    default_args=default_args,
    description='Batch Inference ML Pipeline',
    schedule=timedelta(days=1),
    catchup=False,
    tags=['ml', 'inference', 'batch', 'classification'],
)

def check_inference_data():
    """
    Check if inference data exists and log information
    """
    print(f"Checking for inference data at: {path_to_new_data}")
    
    if not os.path.exists(path_to_new_data):
        print(f"Directory {path_to_new_data} does not exist")
        return False
    
    csv_files = [f for f in os.listdir(path_to_new_data) if f.endswith('.csv')]
    
    print(f"Found {len(csv_files)} CSV files for inference: {csv_files}")
    return True


def load_model_and_scaler():
    """
    Load the trained model and scaler
    """
    print("Loading model and scaler...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # if not os.path.exists(scaler_path):
    #     raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"Model and scaler loaded successfully")
    print(f"Model type: {type(model)}")
    
    return model, scaler

def perform_batch_inference(date):
    """
    Perform batch inference on new data
    """
    print("Starting batch inference...")
    
    model, scaler = load_model_and_scaler()
    
    csv_files = [f for f in os.listdir(path_to_new_data) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found for inference")
        return
    
    output_dir = os.path.join(path_to_new_data, 'predictions')
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for csv_file in csv_files:
        file_path = os.path.join(path_to_new_data, csv_file)
        print(f"Processing file: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} samples from {csv_file}")
            

            # i want just copy..
            has_target = 'target' in df.columns
            if has_target:
                X = df.drop('target', axis=1)
                y_true = df['target']
            else:
                X = df.copy()
                y_true = None
            

            X_scaled = scaler.transform(X)
            
            predictions = model.predict(X_scaled)
            prediction_proba = model.predict_proba(X_scaled)
            
            result_df = df.copy()
            result_df['prediction'] = predictions
            result_df['prediction_probability'] = prediction_proba[:, 1]
            
            date_pripiska = "_".join(str(date).split(' '))
            output_file = os.path.join(output_dir, f'predictions_{date_pripiska}_{csv_file}')
            result_df.to_csv(output_file, index=False)
            

            if has_target and y_true is not None:
                accuracy = accuracy_score(y_true, predictions)
                results.append({
                    'file': csv_file,
                    'samples': len(df),
                    'accuracy': accuracy,
                    'output_file': output_file
                })
                print(f"Accuracy for {csv_file}: {accuracy:.4f}")
            else:
                results.append({
                    'file': csv_file,
                    'samples': len(df),
                    'accuracy': None,
                    'output_file': output_file
                })
                print(f"Predictions saved for {csv_file}")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            assert False

    
    print("Batch inference completed!")

def cleanup():
    """
    Clean up old prediction files (optional)
    """
    print("Cleaning up old files...")
    
    csv_files = [f for f in os.listdir(path_to_new_data) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found for inference")
        return
    
    for csv_file in csv_files:
        file_path = os.path.join(path_to_new_data, csv_file)
        os.remove(file_path)
    
    print("Complete deletin!")


check_data_task = PythonOperator(
    task_id='check_inference_data',
    python_callable=check_inference_data,
    dag=dag,
)

file_sensor_task = FileSensor(
    task_id='wait_for_new_data',
    filepath=path_to_new_data + 'new_data.csv',
    fs_conn_id='fs_default',
    poke_interval=10,  
    timeout=300,  
    mode='poke',
    dag=dag,
)

inference_task = PythonOperator(
    task_id='perform_batch_inference',
    python_callable=perform_batch_inference,
    op_args=["{{ logical_date }}"],
    dag=dag,
)

cleanup_task = PythonOperator(
    task_id='cleanup_old_predictions',
    python_callable=cleanup,
    dag=dag,
)


check_data_task >> file_sensor_task >> inference_task >> cleanup_task
