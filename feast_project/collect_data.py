import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from feast import FeatureStore

def collect_data_for_10_timestamps():
    store = FeatureStore(repo_path="feast_project/feature_repo")
    
    data_path = "feast_project/feature_repo/data/driver_stats.parquet"
    df = pd.read_parquet(data_path)
    
    print(f"Всего записей в данных: {len(df)}")
    print(f"Колонки: {df.columns.tolist()}")
    print(f"Уникальные driver_id: {df['driver_id'].unique()[:10]}")
    
    unique_timestamps = sorted(df['event_timestamp'].unique())
    selected_timestamps = unique_timestamps[:10]
    
    print(f"Выбраныые таймстемпы")
    for i, ts in enumerate(selected_timestamps):
        print(f"  {i+1}. {ts}")
    

    all_driver_ids = df['driver_id'].unique()
    print(f"Всего уникальных driver_id: {len(all_driver_ids)}")
    

    entity_rows = []
    for driver_id in all_driver_ids:
        for timestamp in selected_timestamps:
            entity_rows.append({
                "driver_id": driver_id,
                "event_timestamp": timestamp,
                "val_to_add": np.random.randint(1, 10), # for on demand
                "val_to_add_2": np.random.randint(10, 100) # for on demand
            })
    
    entity_df = pd.DataFrame(entity_rows)
    print(f"\nСоздано {len(entity_df)} записей для получения фичей")
    
    print("\nПолучение исторических фичей...")

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "driver_hourly_stats:conv_rate",
            "driver_hourly_stats:acc_rate",
            "driver_hourly_stats:avg_daily_trips",
            "transformed_conv_rate:conv_rate_plus_val1",
            "transformed_conv_rate:conv_rate_plus_val2",
        ],
    ).to_df()
    
    print(f"Успешно получено {len(training_df)} записей с фичами")
    print("\nПервые 5 строк данных:")
    print(training_df.head())
    
    output_path = "collected_data.csv"
    training_df.to_csv(output_path, index=False)
    print(f"\nДанные сохранены в {output_path}")
    
    return training_df
        


if __name__ == "__main__":
    collected_data = collect_data_for_10_timestamps()
    
    print(f"Всего записей: {len(collected_data)}")
    print(f"Колонки: {collected_data.columns.tolist()}")
    
    required_columns = ['conv_rate', 'acc_rate', 'avg_daily_trips']
    missing_columns = [col for col in required_columns if col not in collected_data.columns]
    
    if missing_columns:
        raise 
    
    print("Готово")