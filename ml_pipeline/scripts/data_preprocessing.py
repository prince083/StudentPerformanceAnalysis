import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def preprocess_data():
    print("Starting Preprocessing Pipeline...")
    
    # 1. Setup specific pipeline architecture paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "raw", "Indian_School_Student_Dataset_10000_Final.csv")
    out_dir = os.path.join(script_dir, "..", "data", "preprocessed")
    model_out_dir = os.path.join(script_dir, "..", "..", "backend_api", "models")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_out_dir, exist_ok=True)
    
    # 2. Load the heavy raw data
    print(f"Loading data from {data_path}...")
    import csv
    
    # Use python engine to auto-detect separators like \t, clean whitespace
    try:
        df = pd.read_csv(data_path, sep='\t')
        if len(df.columns) < 2:
            df = pd.read_csv(data_path) # Fallback to comma
    except:
        df = pd.read_csv(data_path)
        
    print(f"Loaded raw dataset with shape: {df.shape}")
    
    # Fix broken headers (e.g. empty tab space where 'Gender' should be)
    if 'Unnamed: 2' in df.columns:
        df = df.rename(columns={'Unnamed: 2': 'Gender'})
    
    # 3. Handle Missing Values
    print("Handling Missing Values...")
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    
    # 4. Encode Categorical Columns using cleanly tracked instances
    print("Encoding Categories...")
    encoders = {}
    for col in cat_cols:
        if col != 'Student_ID' and col != 'Name': # Do not encode identifiers
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
    # 5. Define ML Targets and Drop Keys
    target = 'Final_Percentage'
    
    # Drop IDs tracking variables before mathematical scaling
    drop_candidates = ['Student_ID', 'Name']
    for candidate in drop_candidates:
        if candidate in df.columns:
            df = df.drop(columns=[candidate])
            
    X = df.drop(columns=[target])
    y = df[target]
    feature_names = X.columns.tolist()
    
    # 6. Normalize Feature Scales 
    print("Applying StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
    
    # 7. Save Encoders and Scalers so the API/React App can cleanly parse JSON user inputs
    metadata = {
        'feature_names': feature_names,
        'encoders': encoders
    }
    joblib.dump(metadata, os.path.join(model_out_dir, 'model_metadata.pkl'))
    joblib.dump(scaler, os.path.join(model_out_dir, 'scaler.pkl'))
    print(f"Saved Metadata and Scalers to backend at: {model_out_dir}")
    
    # 8. Train/Test Split
    print("Splitting the dataset into Train/Test chunks...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=0.20, random_state=42
    )
    
    # 9. Dump prepped data
    X_train.to_csv(os.path.join(out_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(out_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(out_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(out_dir, "y_test.csv"), index=False)
    
    print(f"Successfully split and exported {len(X_train)} training records and {len(X_test)} test records to {out_dir}")
    print("PREPROCESSING COMPLETE. READY FOR MODEL TRAINING.")

if __name__ == "__main__":
    preprocess_data()
