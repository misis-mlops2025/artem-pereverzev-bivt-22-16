"""
Утилиты для работы с Feast в Airflow DAG
"""
from typing import Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from feast import FeatureStore
import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def get_feast_store():
    repo_path = os.path.join(os.path.dirname(__file__), '..', 'feast_project', 'feature_repo')
    return FeatureStore(repo_path=repo_path)

def collect_features_for_training(num_timestamps: int=10) -> Tuple[pd.DataFrame, str]:
    """
    Собирает фичи за указанное количество таймстемпов для всех driver_id
    
    Args:
        num_timestamps: количество таймстемпов для сбора
        
    Returns:
        DataFrame с фичами
    """
    store = get_feast_store()
    

    data_path = os.path.join(os.path.dirname(__file__), '..', 'feast_project', 'feature_repo', 'data', 'driver_stats.parquet')
    df = pd.read_parquet(data_path)
    
    print(f"Всего записей в данных: {len(df)}")
    print(f"Уникальных driver_id: {len(df['driver_id'].unique())}")
    
    unique_timestamps = sorted(df['event_timestamp'].unique())
    
    if len(unique_timestamps) >= num_timestamps:
        selected_timestamps = unique_timestamps[:num_timestamps]
    
    all_driver_ids = df['driver_id'].unique()
    print(f"Всего driver_id для обработки: {len(all_driver_ids)}")
    
    entity_rows = []
    
    for driver_id in all_driver_ids:
        for timestamp in selected_timestamps:
            entity_rows.append({
                "driver_id": driver_id,
                "event_timestamp": timestamp,
                #  случайные значения для on-demand фичей
                "val_to_add": np.random.randint(1, 10),
                "val_to_add_2": np.random.randint(10, 100)
            })
    
    entity_df = pd.DataFrame(entity_rows)
    print(f"Создано {len(entity_df)} записей для получения фичей")
    

    print("Получение фичей из Feast...")

    features_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "driver_hourly_stats:conv_rate",
            "driver_hourly_stats:acc_rate",
            "driver_hourly_stats:avg_daily_trips",
        ],
    ).to_df()
    
    print(f"Успешно получено {len(features_df)} записей с фичами")
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    features_df.to_csv(output_path, index=False)
    print(f"PATH >> Фичи сохранены в {output_path}")
    
    return features_df, output_path


def prepare_training_data(features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Подготавливает данные для обучения модели
    
    Args:
        features_df: DataFrame с фичами
        
    Returns:
        X, y - фичи и таргет для обучения
    """
    required_features = ['conv_rate', 'acc_rate']
    target_column = 'avg_daily_trips'
    

    missing_cols = [col for col in required_features + [target_column] 
                   if col not in features_df.columns]
    
    if missing_cols:
        raise ValueError(f"Отсутствуют необходимые колонки: {missing_cols}")
    
    X = features_df[required_features].copy()
    y = features_df[target_column].copy()
    
    mask = X.isnull().any(axis=1) | y.isnull()
    if mask.any():
        X = X[~mask]
        y = y[~mask]
        print(f"Удалено {mask.sum()} строк с пропущенными значениями")
    
    print(f"Кол-во данных для обучения: {len(X)} записей")
    return X, y

if __name__ == "__main__":
    features_df, output_path = collect_features_for_training(10)
    
    print(f"\nПолученные данные из feast:")
    print(features_df.head())
    
    if features_df is not None:
        X, y = prepare_training_data(features_df)
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")