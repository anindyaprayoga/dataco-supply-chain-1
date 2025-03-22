import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 1. Membaca Dataset
print("Membaca dataset...")
data = pd.read_csv('Dataset Kedua.csv', encoding='latin1')

# 2. Melihat Informasi Dataset
print("\nInformasi Dataset:")
print(f"Jumlah baris: {data.shape[0]}")
print(f"Jumlah kolom: {data.shape[1]}")

# 3. Melihat Kolom-kolom
print("\nNama kolom dalam dataset:")
print(data.columns.tolist())

# 4. Mengecek Missing Values
print("\nMengecek missing values:")
missing_data = data.isnull().sum()
print(missing_data[missing_data > 0])

# 5. Mengecek Tipe Data
print("\nTipe data setiap kolom:")
print(data.dtypes)

# 6. Konversi Tanggal
print("\nMengkonversi kolom tanggal...")
# Mengkonversi kolom tanggal
date_columns = ['order date (DateOrders)', 'shipping date (DateOrders)']
for col in date_columns:
    if col in data.columns:
        data[col] = pd.to_datetime(data[col])

# 7. Membuat fitur waktu
print("Membuat fitur waktu...")
if 'order date (DateOrders)' in data.columns:
    data['order_year'] = data['order date (DateOrders)'].dt.year
    data['order_month'] = data['order date (DateOrders)'].dt.month
    data['order_day'] = data['order date (DateOrders)'].dt.day
    data['order_dayofweek'] = data['order date (DateOrders)'].dt.dayofweek
    data['order_quarter'] = data['order date (DateOrders)'].dt.quarter

# 8. Agregasi data berdasarkan produk dan waktu untuk analisis permintaan
print("\nMengagregasi data untuk analisis permintaan...")
# Grup berdasarkan produk dan bulan pemesanan
product_demand = data.groupby(['Product Name', 'order_year', 'order_month'])['Order Item Quantity'].sum().reset_index()
product_demand.rename(columns={'Order Item Quantity': 'Total_Demand'}, inplace=True)

# 9. Melihat Produk dengan Permintaan Tertinggi
print("\nTop 10 produk dengan permintaan tertinggi:")
top_products = data.groupby('Product Name')['Order Item Quantity'].sum().sort_values(ascending=False).head(10)
print(top_products)

# 10. Visualisasi permintaan produk teratas dari waktu ke waktu
print("\nVisualisasi permintaan produk teratas...")

# Mengambil produk teratas
top_product = top_products.index[0]
print(f"Visualisasi permintaan untuk produk: {top_product}")

# Filter data untuk produk teratas
top_product_demand = product_demand[product_demand['Product Name'] == top_product].copy()

# Membuat kolom tanggal untuk visualisasi
top_product_demand['date'] = pd.to_datetime(top_product_demand['order_year'].astype(str) + '-' + 
                                           top_product_demand['order_month'].astype(str) + '-01')
top_product_demand = top_product_demand.sort_values('date')

# Plot
plt.figure(figsize=(12, 6))
plt.plot(top_product_demand['date'], top_product_demand['Total_Demand'], marker='o')
plt.title(f'Permintaan Bulanan untuk {top_product}')
plt.xlabel('Tanggal')
plt.ylabel('Total Permintaan')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('top_product_demand.png')

# 11. Korelasi antara fitur-fitur numerik
print("\nMenghitung korelasi antara fitur-fitur numerik...")
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
correlation = data[numeric_cols].corr()

# Visualisasi korelasi
plt.figure(figsize=(14, 12))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Korelasi antar Fitur Numerik')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')

# 12. Analisis permintaan berdasarkan kategori produk
print("\nAnalisis permintaan berdasarkan kategori produk...")
category_demand = data.groupby('Category Name')['Order Item Quantity'].sum().sort_values(ascending=False)
print(category_demand)

# Visualisasi permintaan berdasarkan kategori
plt.figure(figsize=(12, 8))
category_demand.plot(kind='bar')
plt.title('Total Permintaan Berdasarkan Kategori Produk')
plt.xlabel('Kategori Produk')
plt.ylabel('Total Permintaan')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('category_demand.png')

# 13. Analisis musiman
print("\nAnalisis musiman permintaan...")
seasonal_demand = data.groupby(['order_year', 'order_month'])['Order Item Quantity'].sum().reset_index()
seasonal_demand['year_month'] = seasonal_demand['order_year'].astype(str) + '-' + seasonal_demand['order_month'].astype(str)

plt.figure(figsize=(14, 6))
plt.plot(seasonal_demand['year_month'], seasonal_demand['Order Item Quantity'], marker='o')
plt.title('Pola Musiman Permintaan')
plt.xlabel('Tahun-Bulan')
plt.ylabel('Total Permintaan')
plt.grid(True)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('seasonal_demand.png')

print("\nPra-pemrosesan data selesai.")