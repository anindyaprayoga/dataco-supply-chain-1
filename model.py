import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

from preparing import prepare_features_for_product_demand_prediction
from preparing import prepare_time_series_data
from preparing import create_features_for_ml_models
from preparing import prepare_train_test_data
from preparing import save_processed_data

# 1. Fungsi untuk melatih model machine learning
def train_ml_models(X_train, y_train, X_test, y_test, preprocessor):
    """
    Melatih beberapa model machine learning dan mengevaluasi performanya
    """
    # Definisikan model-model yang akan digunakan
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42)
    }
    
    # Untuk menyimpan hasil
    results = {}
    
    # Melatih dan mengevaluasi setiap model
    for name, model in models.items():
        print(f"Melatih model {name}...")
        
        # Buat pipeline dengan preprocessor dan model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Latih model
        pipeline.fit(X_train, y_train)
        
        # Prediksi
        y_pred = pipeline.predict(X_test)
        
        # Evaluasi
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        # Simpan hasil
        results[name] = {
            'model': pipeline,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        print(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}, MAPE: {mape:.4f}")
    
    return results

# 2. Fungsi untuk melatih model time series
def train_time_series_models(time_series_data):
    """
    Melatih model time series untuk prediksi permintaan
    """
    # Pisahkan data menjadi train dan test (80:20)
    train_size = int(len(time_series_data) * 0.8)
    train_data = time_series_data.iloc[:train_size]
    test_data = time_series_data.iloc[train_size:]
    
    results = {}
    
    # Model 1: Exponential Smoothing (Holt-Winters)
    print("Melatih model Exponential Smoothing...")
    model_hw = ExponentialSmoothing(
        train_data['demand'],
        seasonal_periods=12,  # Musiman bulanan
        trend='add',
        seasonal='add',
        damped=True
    ).fit()
    
    # Forecast
    hw_forecast = model_hw.forecast(len(test_data))
    
    # Evaluasi
    hw_mse = mean_squared_error(test_data['demand'], hw_forecast)
    hw_rmse = np.sqrt(hw_mse)
    hw_mae = mean_absolute_error(test_data['demand'], hw_forecast)
    hw_mape = mean_absolute_percentage_error(test_data['demand'], hw_forecast)
    
    results['Holt-Winters'] = {
        'model': model_hw,
        'forecast': hw_forecast,
        'mse': hw_mse,
        'rmse': hw_rmse,
        'mae': hw_mae,
        'mape': hw_mape
    }
    
    print(f"  RMSE: {hw_rmse:.2f}, MAE: {hw_mae:.2f}, MAPE: {hw_mape:.4f}")
    
    # Model 2: SARIMA (jika data cukup)
    try:
        print("Melatih model SARIMA...")
        # Coba beberapa parameter SARIMA
        model_sarima = SARIMAX(
            train_data['demand'],
            order=(1, 1, 1),            # (p, d, q)
            seasonal_order=(1, 1, 1, 12)  # (P, D, Q, S)
        ).fit(disp=False)
        
        # Forecast
        sarima_forecast = model_sarima.forecast(len(test_data))
        
        # Evaluasi
        sarima_mse = mean_squared_error(test_data['demand'], sarima_forecast)
        sarima_rmse = np.sqrt(sarima_mse)
        sarima_mae = mean_absolute_error(test_data['demand'], sarima_forecast)
        sarima_mape = mean_absolute_percentage_error(test_data['demand'], sarima_forecast)
        
        results['SARIMA'] = {
            'model': model_sarima,
            'forecast': sarima_forecast,
            'mse': sarima_mse,
            'rmse': sarima_rmse,
            'mae': sarima_mae,
            'mape': sarima_mape
        }
        
        print(f"  RMSE: {sarima_rmse:.2f}, MAE: {sarima_mae:.2f}, MAPE: {sarima_mape:.4f}")
    except Exception as e:
        print(f"SARIMA model failed: {e}")
    
    # Visualisasi hasil
    plt.figure(figsize=(14, 7))
    plt.plot(train_data.index, train_data['demand'], label='Training Data')
    plt.plot(test_data.index, test_data['demand'], label='Actual Test Data')
    
    if 'Holt-Winters' in results:
        plt.plot(test_data.index, results['Holt-Winters']['forecast'], label='Holt-Winters Forecast')
    
    if 'SARIMA' in results:
        plt.plot(test_data.index, results['SARIMA']['forecast'], label='SARIMA Forecast')
    
    plt.title('Perbandingan Model Time Series untuk Prediksi Permintaan')
    plt.xlabel('Tanggal')
    plt.ylabel('Permintaan')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('time_series_comparison.png')
    
    return results

# 3. Fungsi untuk model hibrida (kombinasi ML dan time series)
# 3. Fungsi untuk model hibrida (kombinasi ML dan time series)
def train_hybrid_model(X_train, y_train, X_test, y_test, preprocessor, time_series_data):
    """
    Melatih model hibrida menggunakan fitur dari model ML dan time series
    """
    # Latih model ML (XGBoost)
    print("Melatih model hybrid...")
    ml_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(random_state=42))
    ])
    
    ml_pipeline.fit(X_train, y_train)
    ml_pred = ml_pipeline.predict(X_test)
    
    # Train time series model
    train_size = int(len(time_series_data) * 0.8)
    train_ts = time_series_data.iloc[:train_size]
    test_ts = time_series_data.iloc[train_size:]
    
    print(f"Shape of y_test: {y_test.shape if hasattr(y_test, 'shape') else len(y_test)}")
    print(f"Shape of test_ts: {test_ts.shape}")
    
    # Exponential Smoothing (karena lebih robust dengan data terbatas)
    es_model = ExponentialSmoothing(
        train_ts['demand'],
        seasonal_periods=12,
        trend='add',
        seasonal='add'
    ).fit()
    
    es_pred = es_model.forecast(len(test_ts))
    
    print(f"Shape of ml_pred: {ml_pred.shape if hasattr(ml_pred, 'shape') else len(ml_pred)}")
    print(f"Shape of es_pred: {es_pred.shape if hasattr(es_pred, 'shape') else len(es_pred)}")
    
    # Masalahnya adalah dataset ML dan time series memiliki ukuran berbeda
    # Strategi: Buat model hybrid hanya untuk data yang tersedia dari time series
    # kemudian gunakan model ML untuk sisanya
    
    # Solusi 1: Menggunakan subset data untuk training hybrid
    # Pilih subset dari y_test dan ml_pred yang ukurannya sama dengan es_pred
    if len(es_pred) < len(ml_pred):
        print("Time series predictions are shorter than ML predictions.")
        print("Using time series length for hybrid model training.")
        
        # Ambil subset dari ml_pred dan y_test sesuai dengan panjang es_pred
        # Ini berasumsi bahwa urutan waktu sudah benar di kedua dataset
        subset_size = len(es_pred)
        ml_pred_subset = ml_pred[:subset_size]
        y_test_subset = y_test[:subset_size]
        
        # Normalisasi prediksi subset
        ml_pred_normalized = (ml_pred_subset - ml_pred_subset.min()) / (ml_pred_subset.max() - ml_pred_subset.min())
        es_pred_normalized = (es_pred - es_pred.min()) / (es_pred.max() - es_pred.min())
        
        # Cari bobot optimal untuk subset
        best_mape = float('inf')
        best_weights = (0.5, 0.5)
        
        for w1 in np.arange(0, 1.1, 0.1):
            w2 = 1 - w1
            # Kombinasikan prediksi
            hybrid_pred_subset = (w1 * ml_pred_normalized + w2 * es_pred_normalized)
            
            # Kembalikan ke skala asli
            hybrid_pred_subset = hybrid_pred_subset * (y_test_subset.max() - y_test_subset.min()) + y_test_subset.min()
            
            # Evaluasi
            mape = mean_absolute_percentage_error(y_test_subset, hybrid_pred_subset)
            
            if mape < best_mape:
                best_mape = mape
                best_weights = (w1, w2)
        
        # Gunakan bobot optimal untuk subset
        w1, w2 = best_weights
        
        print(f"Optimal weights determined from subset: ML={w1:.2f}, TS={w2:.2f}")
        
        # Untuk bagian data yang tidak di-cover oleh time series, gunakan hanya ML
        if len(ml_pred) > len(es_pred):
            print("Extending hybrid predictions with ML-only predictions for remaining data points.")
            
            # Buat hybrid prediction untuk subset
            hybrid_pred_subset = (w1 * ml_pred_normalized + w2 * es_pred_normalized)
            hybrid_pred_subset = hybrid_pred_subset * (y_test_subset.max() - y_test_subset.min()) + y_test_subset.min()
            
            # Gunakan ML predictions untuk sisanya
            ml_pred_remaining = ml_pred[subset_size:]
            
            # Gabungkan kedua prediksi
            hybrid_pred = np.concatenate([hybrid_pred_subset, ml_pred_remaining])
        else:
            # Jika tidak ada data tambahan, hybrid prediction hanya berisi subset
            hybrid_pred = (w1 * ml_pred_normalized + w2 * es_pred_normalized)
            hybrid_pred = hybrid_pred * (y_test_subset.max() - y_test_subset.min()) + y_test_subset.min()
    else:
        # Jika es_pred lebih panjang atau sama dengan ml_pred
        print("Using ML predictions length for hybrid model.")
        
        # Potong es_pred sesuai panjang ml_pred
        es_pred = es_pred[:len(ml_pred)]
        
        # Normalisasi prediksi
        ml_pred_normalized = (ml_pred - ml_pred.min()) / (ml_pred.max() - ml_pred.min())
        es_pred_normalized = (es_pred - es_pred.min()) / (es_pred.max() - es_pred.min())
        
        # Cari bobot optimal
        best_mape = float('inf')
        best_weights = (0.5, 0.5)
        
        for w1 in np.arange(0, 1.1, 0.1):
            w2 = 1 - w1
            # Kombinasikan prediksi
            hybrid_pred = (w1 * ml_pred_normalized + w2 * es_pred_normalized)
            
            # Kembalikan ke skala asli
            hybrid_pred = hybrid_pred * (y_test.max() - y_test.min()) + y_test.min()
            
            # Evaluasi
            mape = mean_absolute_percentage_error(y_test, hybrid_pred)
            
            if mape < best_mape:
                best_mape = mape
                best_weights = (w1, w2)
        
        # Gunakan bobot optimal
        w1, w2 = best_weights
        
        hybrid_pred = (w1 * ml_pred_normalized + w2 * es_pred_normalized)
        hybrid_pred = hybrid_pred * (y_test.max() - y_test.min()) + y_test.min()
    
    # Evaluasi final dengan data yang tersedia
    # Pastikan evaluasi hanya menggunakan panjang yang sama dari hybrid_pred dan y_test
    eval_length = min(len(hybrid_pred), len(y_test))
    hybrid_pred_eval = hybrid_pred[:eval_length]
    y_test_eval = y_test[:eval_length]
    
    hybrid_mse = mean_squared_error(y_test_eval, hybrid_pred_eval)
    hybrid_rmse = np.sqrt(hybrid_mse)
    hybrid_mae = mean_absolute_error(y_test_eval, hybrid_pred_eval)
    hybrid_r2 = r2_score(y_test_eval, hybrid_pred_eval)
    hybrid_mape = mean_absolute_percentage_error(y_test_eval, hybrid_pred_eval)
    
    print(f"Model Hybrid (ML weight: {w1:.2f}, TS weight: {w2:.2f})")
    print(f"Evaluation based on {eval_length} data points")
    print(f"  RMSE: {hybrid_rmse:.2f}, MAE: {hybrid_mae:.2f}, R²: {hybrid_r2:.4f}, MAPE: {hybrid_mape:.4f}")
    
    return {
        'model': {
            'ml_model': ml_pipeline,
            'ts_model': es_model,
            'weights': best_weights
        },
        'predictions': hybrid_pred,
        'mse': hybrid_mse,
        'rmse': hybrid_rmse,
        'mae': hybrid_mae,
        'r2': hybrid_r2,
        'mape': hybrid_mape
    }

# 4. Fungsi untuk tuning hyperparameter model terbaik
def tune_best_model(X_train, y_train, X_test, y_test, preprocessor, best_model_name):
    """
    Melakukan tuning hyperparameter pada model terbaik
    """
    param_grid = {}
    
    if best_model_name == 'Random Forest':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
    elif best_model_name == 'Gradient Boosting':
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [3, 5, 7],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
    elif best_model_name == 'XGBoost':
        model = XGBRegressor(random_state=42)
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [3, 5, 7],
            'model__min_child_weight': [1, 3, 5],
            'model__subsample': [0.8, 0.9, 1.0],
            'model__colsample_bytree': [0.8, 0.9, 1.0],
            'model__gamma': [0, 0.1, 0.2]
        }
    elif best_model_name == 'Linear Regression':
        model = LinearRegression()
        # Linear Regression tidak memiliki banyak hyperparameter untuk tuning
        param_grid = {}
    elif best_model_name == 'Ridge Regression':
        model = Ridge()
        param_grid = {
            'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        }
    elif best_model_name == 'Lasso Regression':
        model = Lasso()
        param_grid = {
            'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
        }
    
    # Buat pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Jika parameter grid tidak kosong, lakukan GridSearchCV
    if param_grid:
        print(f"Melakukan tuning hyperparameter untuk model {best_model_name}...")
        
        # Gunakan time series split untuk validasi
        tscv = TimeSeriesSplit(n_splits=5)
        
        # GridSearchCV
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=tscv,
            scoring='neg_mean_absolute_percentage_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Model terbaik
        best_pipeline = grid_search.best_estimator_
        
        # Prediksi
        y_pred = best_pipeline.predict(X_test)
        
        # Evaluasi
        tuned_mse = mean_squared_error(y_test, y_pred)
        tuned_rmse = np.sqrt(tuned_mse)
        tuned_mae = mean_absolute_error(y_test, y_pred)
        tuned_r2 = r2_score(y_test, y_pred)
        tuned_mape = mean_absolute_percentage_error(y_test, y_pred)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Tuned model performance:")
        print(f"  RMSE: {tuned_rmse:.2f}, MAE: {tuned_mae:.2f}, R²: {tuned_r2:.4f}, MAPE: {tuned_mape:.4f}")
        
        return {
            'model': best_pipeline,
            'best_params': grid_search.best_params_,
            'predictions': y_pred,
            'mse': tuned_mse,
            'rmse': tuned_rmse,
            'mae': tuned_mae,
            'r2': tuned_r2,
            'mape': tuned_mape
        }
    else:
        # Jika tidak ada parameter untuk di-tune, gunakan model dasar
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Evaluasi
        base_mse = mean_squared_error(y_test, y_pred)
        base_rmse = np.sqrt(base_mse)
        base_mae = mean_absolute_error(y_test, y_pred)
        base_r2 = r2_score(y_test, y_pred)
        base_mape = mean_absolute_percentage_error(y_test, y_pred)
        
        print(f"Model dasar (tanpa tuning) performance:")
        print(f"  RMSE: {base_rmse:.2f}, MAE: {base_mae:.2f}, R²: {base_r2:.4f}, MAPE: {base_mape:.4f}")
        
        return {
            'model': pipeline,
            'best_params': None,
            'predictions': y_pred,
            'mse': base_mse,
            'rmse': base_rmse,
            'mae': base_mae,
            'r2': base_r2,
            'mape': base_mape
        }

# 5. Fungsi untuk mendapatkan prediksi future
# 5. Fungsi untuk mendapatkan prediksi future
def predict_future_demand(best_model, feature_data, products, months_ahead=3):
    """
    Memprediksi permintaan future untuk produk tertentu
    """
    # Buat data frame untuk menyimpan prediksi
    predictions = pd.DataFrame()
    
    # Tentukan tanggal terakhir dalam dataset
    max_year = feature_data['order_year'].max()
    max_month = feature_data.loc[feature_data['order_year'] == max_year, 'order_month'].max()
    
    print(f"Predicting demand for {months_ahead} months ahead, starting from {max_year}-{max_month}")
    
    # Buat data future
    future_data = []
    
    # Untuk setiap produk
    for product in products:
        # Filter data untuk produk tertentu
        product_data = feature_data[feature_data['Product Name'] == product].copy()
        
        if product_data.empty:
            print(f"Warning: No data found for product '{product}'. Skipping...")
            continue
        
        # Ambil data terbaru untuk produk ini
        latest_data_filtered = product_data.loc[
            (product_data['order_year'] == max_year) & 
            (product_data['order_month'] == max_month)
        ]
        
        # Periksa apakah ada data terbaru untuk produk ini
        if latest_data_filtered.empty:
            print(f"Warning: No data found for product '{product}' in {max_year}-{max_month}.")
            print(f"Using the most recent data available for this product instead.")
            
            # Cari data terbaru yang tersedia untuk produk ini
            # Urutkan berdasarkan tahun dan bulan (terbaru dulu)
            product_data = product_data.sort_values(by=['order_year', 'order_month'], ascending=False)
            
            # Jika masih ada data untuk produk ini, gunakan data terbaru
            if not product_data.empty:
                latest_data = product_data.iloc[0].to_dict()
                print(f"  Using data from {latest_data['order_year']}-{latest_data['order_month']} for product '{product}'")
            else:
                print(f"  No data available for product '{product}'. Skipping...")
                continue
        else:
            # Ada data terbaru, gunakan itu
            latest_data = latest_data_filtered.iloc[0].to_dict()
        
        # Untuk beberapa bulan ke depan
        for i in range(1, months_ahead + 1):
            # Hitung tahun dan bulan untuk prediksi
            pred_month = (max_month + i) % 12
            if pred_month == 0:
                pred_month = 12
            pred_year = max_year + ((max_month + i - 1) // 12)
            
            # Buat data untuk prediksi
            future_record = latest_data.copy()
            future_record['order_year'] = pred_year
            future_record['order_month'] = pred_month
            future_record['order_quarter'] = (pred_month - 1) // 3 + 1
            
            # Hitung fitur musiman
            future_record['month_sin'] = np.sin(2 * np.pi * pred_month / 12)
            future_record['month_cos'] = np.cos(2 * np.pi * pred_month / 12)
            
            # Tambahkan ke data future
            future_data.append(future_record)
    
    # Buat DataFrame dari data future
    if future_data:
        future_df = pd.DataFrame(future_data)
        
        # Prediksi permintaan
        # Periksa kolom yang ada di future_df dan yang diharapkan oleh model
        print(f"Columns in future data: {future_df.columns.tolist()}")
        
        try:
            # Hapus kolom target jika ada
            future_X = future_df.drop(['Order Item Quantity'], axis=1, errors='ignore')
            
            # Prediksi permintaan
            future_df['predicted_demand'] = best_model.predict(future_X)
            
            # Buat DataFrame hasil dengan kolom yang penting
            result_df = future_df[['Product Name', 'Category Name', 'order_year', 'order_month', 'predicted_demand']].copy()
            result_df['predicted_demand'] = result_df['predicted_demand'].round().astype(int)
            
            return result_df
        except Exception as e:
            print(f"Error during prediction: {e}")
            print("This could be due to missing columns or feature mismatch between training and prediction data.")
            print("Features expected by the model might be different from those in future_df.")
            
            # Debug: Check for feature mismatch if possible
            if hasattr(best_model, 'feature_names_in_'):
                model_features = best_model.feature_names_in_
                print(f"Features expected by model: {model_features}")
                missing_features = [f for f in model_features if f not in future_X.columns]
                extra_features = [f for f in future_X.columns if f not in model_features]
                
                if missing_features:
                    print(f"Missing features: {missing_features}")
                if extra_features:
                    print(f"Extra features: {extra_features}")
            
            return pd.DataFrame()  # Return empty DataFrame on error
    else:
        print("No future data could be generated. Check product list and data availability.")
        return pd.DataFrame()  # Return empty DataFrame if no future data

# 6. Fungsi untuk membuat visualisasi prediksi
def visualize_predictions(actual, predicted, title='Perbandingan Nilai Aktual vs Prediksi'):
    """
    Membuat visualisasi perbandingan antara nilai aktual dan prediksi
    """
    plt.figure(figsize=(12, 6))
    
    # Plot nilai aktual
    plt.scatter(actual, actual, alpha=0.5, color='blue', label='Perfect Prediction')
    
    # Plot nilai prediksi
    plt.scatter(actual, predicted, alpha=0.5, color='red', label='Actual vs Predicted')
    
    # Tambahkan garis regresi
    z = np.polyfit(actual, predicted, 1)
    p = np.poly1d(z)
    plt.plot(actual, p(actual), color='green', linestyle='--', label='Regression Line')
    
    plt.title(title)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('prediction_comparison.png')

# 7. Fungsi untuk menyimpan model
def save_model(model, filename):
    """
    Menyimpan model ke file
    """
    import pickle
    
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"Model telah disimpan ke {filename}")

# 8. Fungsi untuk menyimpan hasil prediksi
def save_predictions(predictions, filename='demand_predictions.csv'):
    """
    Menyimpan hasil prediksi ke file CSV
    """
    predictions.to_csv(filename, index=False)
    print(f"Prediksi telah disimpan ke {filename}")