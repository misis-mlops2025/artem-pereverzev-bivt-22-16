import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate_model():
    try:
        data = pd.read_csv("collected_data.csv")
    except FileNotFoundError:
        print("Файл collected_data.csv не найден. Собираем данные заново...")
        from collect_data import collect_data_for_10_timestamps
        data = collect_data_for_10_timestamps()
    
    print(f"Загружено {len(data)} записей")
    print(f"Колонки: {data.columns.tolist()}")
    
    required_features = ['conv_rate', 'acc_rate']
    target_column = 'avg_daily_trips'
    
    missing_cols = [col for col in required_features + [target_column] 
                   if col not in data.columns]
    
    if missing_cols:
        raise
    

    X = data[required_features].copy()
    y = data[target_column].copy()
    
    print(f"\nФичи shape: {X.shape}")
    print(f"Таргет shape: {y.shape}")
    
    print(f"\nNone фичах: {X.isnull().sum().sum()}")
    print(f"None в таргете: {y.isnull().sum()}")
    

    mask = X.isnull().any(axis=1) | y.isnull()
    if mask.any():
        X = X[~mask]
        y = y[~mask]
        print(f"Удалено {mask.sum()} строк с пропущенными значениями")
    

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Train LinearRegression")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    y_pred_lr_train = lr_model.predict(X_train)
    y_pred_lr_test = lr_model.predict(X_test)
    
    print("\nМетрики линейной регрессии:")
    print(f"Train R²: {r2_score(y_train, y_pred_lr_train):.4f}")
    print(f"Test R²: {r2_score(y_test, y_pred_lr_test):.4f}")
    print(f"Train MSE: {mean_squared_error(y_train, y_pred_lr_train):.2f}")
    print(f"Test MSE: {mean_squared_error(y_test, y_pred_lr_test):.2f}")
    print(f"Train MAE: {mean_absolute_error(y_train, y_pred_lr_train):.2f}")
    print(f"Test MAE: {mean_absolute_error(y_test, y_pred_lr_test):.2f}")
    

    print(f"\nКоэффициенты модели:")
    for i, feature in enumerate(required_features):
        print(f"  {feature}: {lr_model.coef_[i]:.4f}")

    
    
    model_filename = "best_model.pkl"
    joblib.dump(lr_model, model_filename)
    print(f"Модель сохранена в {model_filename}")
    
    # Сохраняем метаданные модели
    metadata = {
        'features': required_features,
        'target': target_column,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'test_r2': r2_score(y_test, y_pred_lr_test),
        'test_mse': mean_squared_error(y_test, y_pred_lr_test)
    }
    
    import json
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Метаданные(метрики, тип и проч) модели сохранены в model_metadata.json")
    
    return lr_model, metadata

if __name__ == "__main__":

    model, metadata = train_and_evaluate_model()
    
    if model is None:
        raise

    print("Обучение завершено")

    print(f"Модель обучена для предсказания avg_daily_trips по фичам: {metadata['features']}")
    print(f"Качество модели (R² на тесте): {metadata['test_r2']:.4f}")
    print(f"Ошибка модели (MSE на тесте): {metadata['test_mse']:.2f}")
    print(f"\nМодель сохранена в best_model.pkl")
    print(f"Метаданные сохранены в model_metadata.json")
