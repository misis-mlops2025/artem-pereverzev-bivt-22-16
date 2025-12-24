"""
Скрипт для обучения модели в Airflow DAG
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import os
from datetime import datetime

def train_model(features_df: pd.DataFrame, model_output_dir=None):
    """
    Обучает модель для предсказания avg_daily_trips
    
    Args:
        features_df: DataFrame с фичами
        model_output_dir: директория для сохранения модели
        
    Returns:
        dict с результатами обучения
    """
    if model_output_dir is None:
        model_output_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    os.makedirs(model_output_dir, exist_ok=True)
    
    print("Начало обучения модели...")
    
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
    
    if len(X) == 0:
        raise ValueError("Нет данных для обучения после очистки пропущенных значений")
    
    print(f"Данные для обучения: {len(X)} записей")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    results = {}
    
    print("\nОбучение линейной регрессии...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    y_pred_lr_train = lr_model.predict(X_train)
    y_pred_lr_test = lr_model.predict(X_test)
    
    results['linear_regression'] = {
        'train_r2': float(r2_score(y_train, y_pred_lr_train)),
        'test_r2': float(r2_score(y_test, y_pred_lr_test)),
        'train_mse': float(mean_squared_error(y_train, y_pred_lr_train)),
        'test_mse': float(mean_squared_error(y_test, y_pred_lr_test)),
        'train_mae': float(mean_absolute_error(y_train, y_pred_lr_train)),
        'test_mae': float(mean_absolute_error(y_test, y_pred_lr_test)),
        'coefficients': {feature: float(coef) for feature, coef in zip(required_features, lr_model.coef_)},
        'intercept': float(lr_model.intercept_)
    }
    

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f"model_{timestamp}.pkl"
    model_path = os.path.join(model_output_dir, model_filename)
    
    joblib.dump(lr_model, model_path)
    print(f"Модель сохранена в {model_path}")
    
    metadata = {
        'timestamp': timestamp,
        'features': required_features,
        'target': target_column,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'results': results,
        'model_path': model_path
    }
    
    metadata_path = os.path.join(model_output_dir, f"metadata_{timestamp}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Метаданные сохранены в {metadata_path}")
    
    print("\n" + "="*60)
    print("СВОДКА ОБУЧЕНИЯ МОДЕЛИ")
    print("="*60)

    print("\nКоэффициенты линейной регрессии:")
    for feature, coef in results['linear_regression']['coefficients'].items():
        print(f"  {feature}: {coef:.4f}")
    
    return metadata

if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 100
    test_data = pd.DataFrame({
        'conv_rate': np.random.rand(n_samples),
        'acc_rate': np.random.rand(n_samples),
        'avg_daily_trips': 500 + 200 * np.random.rand(n_samples) + 100 * np.random.rand(n_samples)
    })
    

    metadata = train_model(test_data)
    print("\nТестирование завершено успешно!")