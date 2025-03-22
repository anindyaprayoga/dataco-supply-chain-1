import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error
import pickle
import shap
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

def evaluate_model_performance(y_true, y_pred):
    """
    Evaluasi performa model dengan berbagai metrik dan visualisasi
    """
    # Hitung metrik evaluasi
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # Hitung persentase error per data point
    percent_errors = np.abs((y_true - y_pred) / y_true) * 100
    
    # Visualisasi distribusi error
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(percent_errors, bins=30, alpha=0.7, color='blue')
    plt.axvline(mape * 100, color='red', linestyle='--', label=f'MAPE: {mape * 100:.2f}%')
    plt.xlabel('Percentage Error (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Percentage Errors')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=y_true, y=y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')
    
    plt.tight_layout()
    plt.savefig('error_distribution.png')
    
    # Visualisasi error berdasarkan nilai aktual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, percent_errors, alpha=0.5)
    plt.axhline(mape * 100, color='red', linestyle='--', label=f'MAPE: {mape * 100:.2f}%')
    plt.xlabel('Actual Demand')
    plt.ylabel('Percentage Error (%)')
    plt.title('Percentage Error vs Actual Demand')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('error_vs_actual.png')
    
    return {
        'mape': mape,
        'percent_errors': percent_errors
    }

def analyze_feature_importance(model, X, feature_names=None):
    """
    Analisis feature importance dari model
    """
    # Jika model adalah pipeline, ambil model finalnya
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        model_final = model.named_steps['model']
    else:
        model_final = model
    
    # Check jika model memiliki feature importance
    if hasattr(model_final, 'feature_importances_'):
        # Ambil feature importance
        importances = model_final.feature_importances_
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importances))]
        
        # Buat DataFrame untuk importances
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Visualisasi feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
        return importance_df
    else:
        print("Model tidak memiliki atribut feature_importances_")
        return None

def analyze_with_shap(model, X_test, feature_names=None):
    """
    Analisis model menggunakan SHAP values
    """
    try:
        # Jika model adalah pipeline, ambil model finalnya
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            model_final = model.named_steps['model']
            
            # Jika ada preprocessor, transform data test terlebih dahulu
            if 'preprocessor' in model.named_steps:
                X_processed = model.named_steps['preprocessor'].transform(X_test)
                
                # Jika hasilnya sparse matrix, konversi ke dense
                if hasattr(X_processed, 'toarray'):
                    X_processed = X_processed.toarray()
            else:
                X_processed = X_test
        else:
            model_final = model
            X_processed = X_test
        
        # Buat explainer SHAP
        explainer = shap.Explainer(model_final, X_processed)
        shap_values = explainer(X_processed)
        
        # Plot SHAP summary
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_processed, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig('shap_summary.png')
        
        # Plot SHAP dependency
        if feature_names is not None:
            # Ambil fitur dengan SHAP value tertinggi
            mean_abs_shap = np.abs(shap_values.values).mean(0)
            top_feature_idx = np.argmax(mean_abs_shap)
            top_feature_name = feature_names[top_feature_idx]
            
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(top_feature_idx, shap_values.values, X_processed, 
                                feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(f'shap_dependence_{top_feature_name}.png')
        
        return shap_values
    except Exception as e:
        print(f"Error dalam analisis SHAP: {e}")
        return None

def analyze_permutation_importance(model, X_test, y_test, feature_names=None):
    """
    Analisis feature importance menggunakan permutation importance
    """
    try:
        # Hitung permutation importance
        perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        
        # Buat DataFrame
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(perm_importance.importances_mean))]
        
        perm_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': perm_importance.importances_mean,
            'Std': perm_importance.importances_std
        }).sort_values('Importance', ascending=False)
        
        # Visualisasi
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=perm_importance_df.head(20),
                   xerr=perm_importance_df['Std'].head(20))
        plt.title('Permutation Feature Importance')
        plt.tight_layout()
        plt.savefig('permutation_importance.png')
        
        return perm_importance_df
    except Exception as e:
        print(f"Error dalam analisis permutation importance: {e}")
        return None

def analyze_demand_patterns(data, date_col, demand_col, product_col=None, category_col=None):
    """
    Analisis pola permintaan secara temporal
    """
    # Pastikan kolom tanggal dalam format datetime
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        data[date_col] = pd.to_datetime(data[date_col])
    
    # Ekstrak komponen tanggal
    data['year'] = data[date_col].dt.year
    data['month'] = data[date_col].dt.month
    data['quarter'] = data[date_col].dt.quarter
    data['day_of_week'] = data[date_col].dt.dayofweek
    
    # 1. Analisis tren bulanan
    monthly_demand = data.groupby(['year', 'month'])[demand_col].sum().reset_index()
    monthly_demand['year_month'] = monthly_demand['year'].astype(str) + '-' + monthly_demand['month'].astype(str).str.zfill(2)
    
    plt.figure(figsize=(14, 6))
    plt.plot(monthly_demand['year_month'], monthly_demand[demand_col], marker='o')
    plt.title('Tren Permintaan Bulanan')
    plt.xlabel('Tahun-Bulan')
    plt.ylabel('Total Permintaan')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('monthly_demand_trend.png')
    
    # 2. Analisis musiman berdasarkan bulan
    monthly_seasonal = data.groupby('month')[demand_col].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='month', y=demand_col, data=monthly_seasonal)
    plt.title('Permintaan Rata-rata per Bulan')
    plt.xlabel('Bulan')
    plt.ylabel('Permintaan Rata-rata')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('monthly_seasonal_pattern.png')
    
    # 3. Analisis berdasarkan hari dalam seminggu
    daily_demand = data.groupby('day_of_week')[demand_col].mean().reset_index()
    days = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
    daily_demand['day_name'] = daily_demand['day_of_week'].apply(lambda x: days[x])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='day_name', y=demand_col, data=daily_demand)
    plt.title('Permintaan Rata-rata per Hari')
    plt.xlabel('Hari')
    plt.ylabel('Permintaan Rata-rata')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('daily_demand_pattern.png')
    
    # 4. Analisis berdasarkan kuartal
    quarterly_demand = data.groupby(['year', 'quarter'])[demand_col].mean().reset_index()
    quarterly_demand['year_quarter'] = quarterly_demand['year'].astype(str) + '-Q' + quarterly_demand['quarter'].astype(str)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='year_quarter', y=demand_col, data=quarterly_demand)
    plt.title('Permintaan Rata-rata per Kuartal')
    plt.xlabel('Tahun-Kuartal')
    plt.ylabel('Permintaan Rata-rata')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('quarterly_demand_pattern.png')
    
    # 5. Jika produk atau kategori disediakan, lakukan analisis berdasarkan itu
    if product_col is not None and product_col in data.columns:
        # Ambil produk dengan permintaan tertinggi
        top_products = data.groupby(product_col)[demand_col].sum().nlargest(5).index.tolist()
        
        # Filter data untuk produk-produk teratas
        top_product_data = data[data[product_col].isin(top_products)]
        
        # Analisis tren bulanan per produk
        product_monthly = top_product_data.groupby([product_col, 'year', 'month'])[demand_col].sum().reset_index()
        product_monthly['year_month'] = product_monthly['year'].astype(str) + '-' + product_monthly['month'].astype(str).str.zfill(2)
        
        plt.figure(figsize=(14, 8))
        for product in top_products:
            product_data = product_monthly[product_monthly[product_col] == product]
            plt.plot(product_data['year_month'], product_data[demand_col], marker='o', label=product)
        
        plt.title('Tren Permintaan Bulanan per Produk (Top 5)')
        plt.xlabel('Tahun-Bulan')
        plt.ylabel('Total Permintaan')
        plt.xticks(rotation=90)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('product_monthly_demand.png')
    
    if category_col is not None and category_col in data.columns:
        # Ambil kategori dengan permintaan tertinggi
        top_categories = data.groupby(category_col)[demand_col].sum().nlargest(5).index.tolist()
        
        # Filter data untuk kategori-kategori teratas
        top_category_data = data[data[category_col].isin(top_categories)]
        
        # Analisis tren bulanan per kategori
        category_monthly = top_category_data.groupby([category_col, 'year', 'month'])[demand_col].sum().reset_index()
        category_monthly['year_month'] = category_monthly['year'].astype(str) + '-' + category_monthly['month'].astype(str).str.zfill(2)
        
        plt.figure(figsize=(14, 8))
        for category in top_categories:
            category_data = category_monthly[category_monthly[category_col] == category]
            plt.plot(category_data['year_month'], category_data[demand_col], marker='o', label=category)
        
        plt.title('Tren Permintaan Bulanan per Kategori (Top 5)')
        plt.xlabel('Tahun-Bulan')
        plt.ylabel('Total Permintaan')
        plt.xticks(rotation=90)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('category_monthly_demand.png')
    
    # 6. Tambahkan analisis outlier dan anomali
    z_scores = np.abs((data[demand_col] - data[demand_col].mean()) / data[demand_col].std())
    outliers = data[z_scores > 3]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(data.index, data[demand_col], alpha=0.5, label='Normal')
    plt.scatter(outliers.index, outliers[demand_col], color='red', label='Outlier')
    plt.title('Deteksi Outlier Permintaan')
    plt.xlabel('Index')
    plt.ylabel('Permintaan')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('demand_outliers.png')
    
    # Perhitungan statistik deskriptif
    stats = {
        'total_demand': data[demand_col].sum(),
        'avg_demand': data[demand_col].mean(),
        'median_demand': data[demand_col].median(),
        'min_demand': data[demand_col].min(),
        'max_demand': data[demand_col].max(),
        'std_demand': data[demand_col].std(),
        'peak_month': monthly_seasonal.loc[monthly_seasonal[demand_col].idxmax(), 'month'],
        'lowest_month': monthly_seasonal.loc[monthly_seasonal[demand_col].idxmin(), 'month'],
        'peak_day': daily_demand.loc[daily_demand[demand_col].idxmax(), 'day_name'],
        'outliers_count': len(outliers)
    }
    
    return {
        'monthly_demand': monthly_demand,
        'monthly_seasonal': monthly_seasonal,
        'daily_demand': daily_demand,
        'quarterly_demand': quarterly_demand,
        'outliers': outliers,
        'stats': stats
    }

def save_analysis_results(results, filename='analysis_results.pkl'):
    """
    Menyimpan hasil analisis ke file pickle
    """
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Hasil analisis berhasil disimpan ke {filename}")
    
def load_analysis_results(filename='analysis_results.pkl'):
    """
    Memuat hasil analisis dari file pickle
    """
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    return results

def create_demand_forecast_report(data, analysis_results, model_results, predictions, output_file='forecast_report.html'):
    """
    Membuat laporan prediksi permintaan dalam format HTML
    """
    # Buat template HTML sederhana
    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Laporan Prediksi Permintaan Produk</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            img {{ max-width: 100%; height: auto; margin: 10px 0; }}
            .highlight {{ background-color: #e8f4f8; padding: 10px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Laporan Prediksi Permintaan Produk</h1>
        <p>Laporan ini berisi analisis permintaan produk dan prediksi untuk periode mendatang.</p>
        
        <h2>Statistik Permintaan</h2>
        <div class="highlight">
            <p>Total Permintaan: {analysis_results['stats']['total_demand']:,.0f} unit</p>
            <p>Rata-rata Permintaan: {analysis_results['stats']['avg_demand']:,.2f} unit</p>
            <p>Bulan dengan Permintaan Tertinggi: {analysis_results['stats']['peak_month']}</p>
            <p>Hari dengan Permintaan Tertinggi: {analysis_results['stats']['peak_day']}</p>
        </div>
        
        <h2>Performa Model</h2>
        <p>MAPE (Mean Absolute Percentage Error): {model_results['mape']*100:.2f}%</p>
        
        <h2>Prediksi Permintaan Masa Depan</h2>
        <table>
            <tr>
                <th>Produk</th>
                <th>Bulan</th>
                <th>Prediksi Permintaan</th>
            </tr>
    '''
    
    # Tambahkan baris tabel untuk setiap prediksi
    for _, row in predictions.head(20).iterrows():
        html += f'''
            <tr>
                <td>{row.get('Product Name', 'N/A')}</td>
                <td>{row.get('Forecast Month', 'N/A')}</td>
                <td>{row.get('Predicted Demand', 0):,.0f}</td>
            </tr>
        '''
    
    # Tutup tabel dan tambahkan gambar
    html += '''
        </table>
        
        <h2>Visualisasi</h2>
        <h3>Tren Permintaan Bulanan</h3>
        <img src="monthly_demand_trend.png" alt="Tren Permintaan Bulanan">
        
        <h3>Permintaan Rata-rata per Bulan</h3>
        <img src="monthly_seasonal_pattern.png" alt="Permintaan Rata-rata per Bulan">
        
        <h3>Perbandingan Permintaan Aktual vs Prediksi</h3>
        <img src="actual_vs_predicted.png" alt="Perbandingan Permintaan Aktual vs Prediksi">
        
        <h3>Distribusi Error</h3>
        <img src="error_distribution.png" alt="Distribusi Error">
        
        <h2>Kesimpulan</h2>
        <p>Berdasarkan analisis dan prediksi yang telah dilakukan, pola permintaan produk menunjukkan
        tren musiman dengan puncak pada bulan tertentu. Model prediksi yang digunakan memiliki tingkat
        akurasi yang memadai dan dapat digunakan untuk perencanaan inventory dan supply chain.</p>
        
        <p>Disarankan untuk memperhatikan fluktuasi permintaan terutama pada produk-produk populer
        dan menyesuaikan strategi inventory sesuai dengan hasil prediksi.</p>
        
        <footer>
            <p>Laporan ini dibuat secara otomatis pada: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </body>
    </html>
    '''
    
    # Simpan ke file
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"Laporan prediksi permintaan berhasil disimpan ke {output_file}")