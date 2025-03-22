import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime

# Lanjutan dari kode sebelumnya
print("Persiapan data untuk pemodelan prediksi permintaan...")

# 1. Memilih fitur yang relevan untuk prediksi permintaan
def prepare_features_for_product_demand_prediction(data):
    """
    Menyiapkan fitur-fitur untuk prediksi permintaan produk
    """
    # Membuat salinan data untuk menghindari warning SettingWithCopyWarning
    df = data.copy()
    
    # Konversi tanggal jika belum dikonversi
    date_columns = ['order date (DateOrders)', 'shipping date (DateOrders)']
    for col in date_columns:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])
    
    # Membuat fitur waktu jika belum dibuat
    if 'order date (DateOrders)' in df.columns:
        if 'order_year' not in df.columns:
            df['order_year'] = df['order date (DateOrders)'].dt.year
        if 'order_month' not in df.columns:
            df['order_month'] = df['order date (DateOrders)'].dt.month
        if 'order_day' not in df.columns:
            df['order_day'] = df['order date (DateOrders)'].dt.day
        if 'order_dayofweek' not in df.columns:
            df['order_dayofweek'] = df['order date (DateOrders)'].dt.dayofweek
        if 'order_quarter' not in df.columns:
            df['order_quarter'] = df['order date (DateOrders)'].dt.quarter
    
    # Fitur-fitur yang mungkin relevan untuk prediksi permintaan
    relevant_features = [
        'Product Name', 'Category Name', 'order_year', 'order_month', 
        'order_quarter', 'order_dayofweek', 'Order Item Quantity',
        'Market', 'Order Region', 'Order Country', 'Customer Segment'
    ]
    
    # Filter kolom yang ada dalam dataset
    available_features = [col for col in relevant_features if col in df.columns]
    
    # Ekstrak fitur-fitur yang tersedia
    feature_df = df[available_features].copy()
    
    # Menangani missing values
    for col in feature_df.columns:
        if feature_df[col].dtype == 'object':
            feature_df[col] = feature_df[col].fillna('Unknown')
        else:
            feature_df[col] = feature_df[col].fillna(feature_df[col].median())
    
    return feature_df

# 2. Persiapan dataset untuk pemodelan prediksi permintaan produk
def prepare_time_series_data(data, product_name=None, category_name=None):
    """
    Menyiapkan data time series untuk prediksi permintaan.
    Jika product_name ditentukan, fokus pada produk tertentu.
    Jika category_name ditentukan, fokus pada kategori tertentu.
    """
    # Membuat salinan data
    df = data.copy()
    
    # Filter berdasarkan produk jika ditentukan
    if product_name is not None:
        df = df[df['Product Name'] == product_name]
    
    # Filter berdasarkan kategori jika ditentukan
    if category_name is not None:
        df = df[df['Category Name'] == category_name]
    
    # Agregasi data berdasarkan waktu (tahun, bulan)
    time_series = df.groupby(['order_year', 'order_month'])['Order Item Quantity'].sum().reset_index()
    
    # Buat tanggal dari tahun dan bulan
    time_series['date'] = pd.to_datetime(time_series['order_year'].astype(str) + '-' + 
                                        time_series['order_month'].astype(str) + '-01')
    
    # Urutkan berdasarkan tanggal
    time_series = time_series.sort_values('date')
    
    # Tetapkan tanggal sebagai indeks
    time_series.set_index('date', inplace=True)
    
    # Ubah nama kolom untuk kejelasan
    time_series.rename(columns={'Order Item Quantity': 'demand'}, inplace=True)
    
    return time_series

# 3. Persiapan fitur untuk model machine learning
def create_features_for_ml_models(data):
    """
    Membuat fitur-fitur untuk model machine learning
    """
    # Siapkan fitur-fitur
    feature_data = prepare_features_for_product_demand_prediction(data)
    
    # Agregasi data berdasarkan produk, tahun, dan bulan
    agg_data = feature_data.groupby(['Product Name', 'Category Name', 'order_year', 'order_month', 'order_quarter']).\
        agg({
            'Order Item Quantity': 'sum',
            'order_dayofweek': 'mean',  # rata-rata hari dalam seminggu
            'Market': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
            'Order Region': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
            'Order Country': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
            'Customer Segment': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
        }).reset_index()
    
    # Membuat fitur lag untuk time series (permintaan pada bulan-bulan sebelumnya)
    product_groups = agg_data.groupby('Product Name')
    
    # Inisialisasi kolom lag
    for lag in range(1, 4):  # Lag 1, 2, dan 3 bulan
        agg_data[f'demand_lag_{lag}'] = np.nan
    
    # Hitung lag untuk setiap produk
    for product, group in product_groups:
        group = group.sort_values(['order_year', 'order_month'])
        
        # Menghitung lag
        for lag in range(1, 4):
            agg_data.loc[group.index, f'demand_lag_{lag}'] = group['Order Item Quantity'].shift(lag)
    
    # Isi missing values di kolom lag
    for lag in range(1, 4):
        agg_data[f'demand_lag_{lag}'] = agg_data[f'demand_lag_{lag}'].fillna(agg_data['Order Item Quantity'].median())
    
    # Membuat fitur musiman (sin dan cos dari bulan untuk menangkap pola siklik)
    agg_data['month_sin'] = np.sin(2 * np.pi * agg_data['order_month'] / 12)
    agg_data['month_cos'] = np.cos(2 * np.pi * agg_data['order_month'] / 12)
    
    # Variabel target
    y = agg_data['Order Item Quantity']
    
    # Fitur-fitur untuk model
    X = agg_data.drop(['Order Item Quantity'], axis=1)
    
    return X, y, agg_data

# 4. Persiapan untuk train-test split
def prepare_train_test_data(X, y, test_size=0.2, random_state=42):
    """
    Memisahkan data menjadi set pelatihan dan pengujian
    """
    # Identifikasi kolom kategoris dan numerik
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Definisikan preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    return X_train, X_test, y_train, y_test, preprocessor, categorical_cols, numerical_cols

# Fungsi untuk menyimpan data yang sudah diproses
def save_processed_data(X_train, X_test, y_train, y_test, file_prefix='processed'):
    """
    Menyimpan data yang sudah diproses ke dalam file CSV
    """
    # Buat DataFrame dari data training dan testing
    train_df = pd.DataFrame(X_train)
    train_df['target'] = y_train
    
    test_df = pd.DataFrame(X_test)
    test_df['target'] = y_test
    
    # Simpan ke file CSV
    train_df.to_csv(f'{file_prefix}_train.csv', index=False)
    test_df.to_csv(f'{file_prefix}_test.csv', index=False)
    
    print(f"Data telah disimpan ke {file_prefix}_train.csv dan {file_prefix}_test.csv")

# Simpan fungsi-fungsi ini untuk digunakan dalam training model