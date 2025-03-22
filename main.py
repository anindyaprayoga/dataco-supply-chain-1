import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from model import predict_future_demand, save_model, save_predictions, train_hybrid_model, train_ml_models, train_time_series_models, tune_best_model, visualize_predictions
from preparing import create_features_for_ml_models, prepare_features_for_product_demand_prediction, prepare_time_series_data, prepare_train_test_data
warnings.filterwarnings('ignore')
import pickle
import os
from datetime import datetime

# Import fungsi-fungsi yang telah kita definisikan sebelumnya
# Asumsikan bahwa semua fungsi dari kode sebelumnya sudah didefinisikan

def main():
    """
    Fungsi utama untuk menjalankan workflow lengkap prediksi permintaan produk
    """
    print("=== Sistem Prediksi Permintaan Produk DataCo Smart Supply Chain ===")
    print(f"Dijalankan pada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Memuat dataset
    print("\n1. Memuat dataset...")
    try:
        data = pd.read_csv('Dataset Kedua.csv', encoding='latin1')
        print(f"  Dataset berhasil dimuat dengan {data.shape[0]} baris dan {data.shape[1]} kolom.")
    except Exception as e:
        print(f"  Error saat memuat dataset: {e}")
        return
    
    # 2. Eksplorasi Data Awal
    print("\n2. Melakukan eksplorasi data awal...")
    
    # Melihat informasi dasar dataset
    print(f"  Jumlah data: {data.shape[0]}")
    
    # Memeriksa missing values
    missing_count = data.isnull().sum().sum()
    print(f"  Total missing values: {missing_count}")
    
    # Memeriksa kolom tanggal dan mengkonversinya
    date_columns = ['order date (DateOrders)', 'shipping date (DateOrders)']
    for col in date_columns:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col])
    
    # Melihat rentang waktu dataset
    if 'order date (DateOrders)' in data.columns:
        min_date = data['order date (DateOrders)'].min()
        max_date = data['order date (DateOrders)'].max()
        print(f"  Rentang waktu dataset: {min_date.strftime('%Y-%m-%d')} hingga {max_date.strftime('%Y-%m-%d')}")
    
    # 3. Persiapan Data untuk Analisis
    print("\n3. Menyiapkan data untuk analisis...")
    
    # Buat fitur waktu
    if 'order date (DateOrders)' in data.columns:
        data['order_year'] = data['order date (DateOrders)'].dt.year
        data['order_month'] = data['order date (DateOrders)'].dt.month
        data['order_day'] = data['order date (DateOrders)'].dt.day
        data['order_dayofweek'] = data['order date (DateOrders)'].dt.dayofweek
        data['order_quarter'] = data['order date (DateOrders)'].dt.quarter
    
    # 4. Analisis Permintaan Produk
    print("\n4. Melakukan analisis permintaan produk...")
    
    # Agregasi berdasarkan produk
    product_demand = data.groupby('Product Name')['Order Item Quantity'].sum().sort_values(ascending=False)
    print(f"  Top 5 produk dengan permintaan tertinggi:")
    for i, (product, demand) in enumerate(product_demand.head(5).items(), 1):
        print(f"    {i}. {product}: {demand} unit")
    
    # Agregasi berdasarkan kategori
    category_demand = data.groupby('Category Name')['Order Item Quantity'].sum().sort_values(ascending=False)
    print(f"  Top 5 kategori dengan permintaan tertinggi:")
    for i, (category, demand) in enumerate(category_demand.head(5).items(), 1):
        print(f"    {i}. {category}: {demand} unit")
    
    # 5. Persiapan Data untuk Pemodelan
    print("\n5. Menyiapkan data untuk pemodelan...")
    
    # Siapkan fitur-fitur
    feature_data = prepare_features_for_product_demand_prediction(data)
    print(f"  Fitur yang digunakan: {feature_data.columns.tolist()}")
    
    # Buat fitur untuk model ML
    X, y, agg_data = create_features_for_ml_models(data)
    print(f"  Jumlah data setelah agregasi: {len(agg_data)}")
    
    # Split data
    X_train, X_test, y_train, y_test, preprocessor, cat_cols, num_cols = prepare_train_test_data(X, y)
    print(f"  Data dibagi menjadi {len(X_train)} data training dan {len(X_test)} data testing")
    
    # 6. Melatih Model Time Series
    print("\n6. Melatih model time series...")
    
    # Pilih produk populer untuk dimodelkan
    top_product = product_demand.index[0]
    print(f"  Melatih model time series untuk produk: {top_product}")
    
    # Siapkan data time series
    time_series_data = prepare_time_series_data(data, product_name=top_product)
    
    # Latih model time series
    ts_results = train_time_series_models(time_series_data)
    
    # 7. Melatih Model Machine Learning
    print("\n7. Melatih model machine learning...")
    
    # Latih beberapa model ML
    ml_results = train_ml_models(X_train, y_train, X_test, y_test, preprocessor)
    
    # Temukan model terbaik
    best_model_name = min(ml_results.items(), key=lambda x: x[1]['mape'])[0]
    print(f"  Model ML terbaik: {best_model_name} dengan MAPE {ml_results[best_model_name]['mape']:.4f}")
    
    # 8. Melatih Model Hibrida
    print("\n8. Melatih model hibrida...")
    
    # Latih model hibrida
    hybrid_result = train_hybrid_model(X_train, y_train, X_test, y_test, preprocessor, time_series_data)
    
    # 9. Tuning Model Terbaik
    print("\n9. Melakukan tuning hyperparameter untuk model terbaik...")
    
    # Lakukan tuning untuk model terbaik
    tuned_result = tune_best_model(X_train, y_train, X_test, y_test, preprocessor, best_model_name)
    
    # 10. Evaluasi dan Perbandingan Model
    print("\n10. Evaluasi dan perbandingan model...")
    
    # Buat dictionary untuk menyimpan MAPE dari semua model
    all_mapes = {
        'Hybrid': hybrid_result['mape'],
        f'Tuned {best_model_name}': tuned_result['mape']
    }
    
    # Tambahkan MAPE dari model ML
    for name, result in ml_results.items():
        all_mapes[name] = result['mape']
    
    # Tambahkan MAPE dari model time series
    for name, result in ts_results.items():
        all_mapes[name] = result['mape']
    
    # Temukan model dengan MAPE terendah
    final_best_model_name = min(all_mapes.items(), key=lambda x: x[1])[0]
    print(f"  Model dengan MAPE terendah: {final_best_model_name} dengan MAPE {all_mapes[final_best_model_name]:.4f}")
    
    # Visualisasi perbandingan MAPE
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(all_mapes.keys()), y=list(all_mapes.values()))
    plt.title('Perbandingan MAPE Antar Model')
    plt.xlabel('Model')
    plt.ylabel('MAPE (Mean Absolute Percentage Error)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    
    # 11. Prediksi Permintaan Masa Depan
    print("\n11. Memprediksi permintaan masa depan...")
    
    # Pilih model untuk prediksi
    if final_best_model_name == 'Hybrid':
        final_model = hybrid_result['model']
    elif final_best_model_name.startswith('Tuned'):
        final_model = tuned_result['model']
    elif final_best_model_name in ml_results:
        final_model = ml_results[final_best_model_name]['model']
    else:
        # Jika model terbaik adalah time series, gunakan model ML terbaik
        final_model = tuned_result['model']
    
    # Pilih top N produk untuk prediksi
    top_products = product_demand.head(10).index.tolist()
    
    # Prediksi 3 bulan ke depan
    future_predictions = predict_future_demand(final_model, agg_data, top_products, months_ahead=3)
    
    # Tampilkan prediksi
    print(f"  Prediksi permintaan untuk 3 bulan ke depan untuk {len(top_products)} produk teratas:")
    print(future_predictions.head(10))
    
    # 12. Simpan Model dan Hasil
    print("\n12. Menyimpan model dan hasil...")
    
    # Buat folder untuk output jika belum ada
    os.makedirs('output', exist_ok=True)
    
    # Simpan model
    save_model(final_model, 'output/best_demand_prediction_model.pkl')
    
    # Simpan hasil prediksi
    save_predictions(future_predictions, 'output/future_demand_predictions.csv')
    
    # Visualisasi hasil prediksi vs aktual
    visualize_predictions(y_test, tuned_result['predictions'], 'Perbandingan Permintaan Aktual vs Prediksi')
    
    # 13. Tampilkan Kesimpulan
    print("\n13. Kesimpulan:")
    print(f"  Model terbaik untuk prediksi permintaan produk adalah {final_best_model_name}")
    print(f"  MAPE (Mean Absolute Percentage Error): {all_mapes[final_best_model_name]:.4f}")
    print(f"  Model berhasil disimpan ke output/best_demand_prediction_model.pkl")
    print(f"  Prediksi permintaan untuk 3 bulan ke depan berhasil disimpan ke output/future_demand_predictions.csv")
    print("\nProses analisis dan pemodelan selesai.")

if __name__ == "__main__":
    main()