"""
DAG для подтягивания актуальных фичей за 10 таймстемпов для всех driver_id
и обучения модели предсказания avg_daily_trips
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.models import Variable
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from dags.feast_utils import collect_features_for_training
from dags.model_training import train_model


default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
}


dag = DAG(
    'feast_training_pipeline',
    default_args=default_args,
    description='DAG для сбора фичей через Feast и обучения модели',
    schedule_interval='@daily',
    catchup=False,
    tags=['feast', 'mlops', 'training'],
)

def collect_features_task(**context):
    """
    Задача для сбора фичей через Feast
    """
    try:
        num_timestamps = Variable.get("feast_num_timestamps", default_var=10)
        num_timestamps = int(num_timestamps)
        
        print(f"Сбор фичей за {num_timestamps} таймстемпов...")
        
        features_df, output_path = collect_features_for_training(num_timestamps)
        
        context['ti'].xcom_push(key='features_df', value=features_df.to_dict())
        context['ti'].xcom_push(key='features_path', value=output_path)
        
        print(f"Сбор фичей завершен. Сохранено {len(features_df)} записей")
        return output_path
        
    except Exception as e:
        print(f"Ошибка при сборе фичей: {e}")
        raise

def train_model_task(**context):
    ti = context['ti']
    features_dict = ti.xcom_pull(task_ids='collect_features', key='features_df')
    
    if features_dict is None:
        raise ValueError("Не удалось получить данные фичей из XCom")
    
    import pandas as pd
    features_df = pd.DataFrame.from_dict(features_dict)
    
    print(f"Обучение модели на {len(features_df)} записях...")
    
    metadata = train_model(features_df)
    
    ti.xcom_push(key='model_metadata', value=metadata)
    ti.xcom_push(key='best_model_r2', value=metadata.get('results', {}).get(metadata['best_model'], {}).get('test_r2', 0))
    
    print(f"Обучение завершено. Лучшая модель: {metadata['best_model']}")
    print(f"R² на тесте: {metadata.get('results', {}).get(metadata['best_model'], {}).get('test_r2', 0):.4f}")
    
    return metadata



start_task = DummyOperator(
    task_id='start',
    dag=dag,
)

collect_features = PythonOperator(
    task_id='collect_features',
    python_callable=collect_features_task,
    provide_context=True,
    dag=dag,
)

train_model_step = PythonOperator(
    task_id='train_model',
    python_callable=train_model_task,
    provide_context=True,
    dag=dag,
)


end_task = DummyOperator(
    task_id='end',
    dag=dag,
)


start_task >> collect_features >> train_model_step >> end_task
