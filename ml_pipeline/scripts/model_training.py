import pandas as pd
import numpy as np
import os
import pickle

# Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_evaluate_models():
    print("Starting Model Training Pipeline...")
    
    # Define reliable paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data', 'preprocessed')
    models_dir = os.path.join(base_dir, '..', '..', 'backend_api', 'models')
    
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load the pristine preprocessed data
    print(f"Loading preprocessed data from: {data_dir}")
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    
    # 2. Initialize exactly 4 models (SVR is strictly excluded)
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost Regressor": XGBRegressor(objective='reg:squarederror', random_state=42)
    }
    
    results = []
    
    # 3. Train all models in a heavy loop
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate exactly the 4 requested metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results.append([name, mae, mse, rmse, r2])
        
    # 4. Determine the best model using the R2 Score (Highest wins)
    results_df = pd.DataFrame(results, columns=["Model", "MAE", "MSE", "RMSE", "R2"])
    results_df = results_df.sort_values(by="R2", ascending=False).reset_index(drop=True)
    
    # Save the mathematical metrics table
    perf_path = os.path.join(data_dir, "model_performance.csv")
    results_df.to_csv(perf_path, index=False)
    print(f"\nModel Performance saved to: {perf_path}")
    print("\n=== COMPLETE BATTLE REPORT ===")
    print(results_df.to_string())
    
    # 5. Crown and Save the Absolute Best Model
    best_model_name = results_df.iloc[0]["Model"]
    best_model = models[best_model_name]
    
    print(f"\n👑 Best Model Crowned: {best_model_name} (Highest R² Score)")
    
    # Export it straight to backend_api/models
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    with open(best_model_path, "wb") as f:
        pickle.dump(best_model, f)
        
    print(f"Successfully saved {best_model_name} to {best_model_path}")
    print("\nPIPELINE LOGIC COMPLETED AUTOMATICALLY.")

if __name__ == "__main__":
    train_and_evaluate_models()
