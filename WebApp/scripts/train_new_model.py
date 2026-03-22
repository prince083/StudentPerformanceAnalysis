
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# 1. Load Data
script_dir = os.path.dirname(os.path.abspath(__file__))
# Redirecting to the Indian College Student Dataset
data_path = os.path.join(script_dir, "..", "data", "Indian_College_Student_Dataset.csv")

if not os.path.exists(data_path):
    print(f"Data file not found at {data_path}. Please ensure Indian_College_Student_Dataset.csv exists.")
    exit()

df = pd.read_csv(data_path)
print("Indian Student Dataset Loaded Successfully!")
print(f"Dataset Shape: {df.shape}")

# 2. Feature Engineering & Preprocessing
# Target variable is now 'Final_Percentage' 
# We drop ID and the target itself from features.
drop_cols = ['Student_ID', 'Final_Percentage'] 

features = df.drop(drop_cols, axis=1, errors='ignore')
target = df['Final_Percentage'] 

print("Training features:", features.columns.tolist())

# Encode Categorical Variables
categorical_cols = features.select_dtypes(include=['object']).columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col].astype(str))
    encoders[col] = le # Save for later (Dashboard/SHAP)

# Scale Numerical Variables
scaler = StandardScaler()
X = scaler.fit_transform(features)
y = target

# 3. Model Training (Regression)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using 200 estimators for higher precision on real-world data
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluation
y_pred = model.predict(X_test)
print("\nModel Performance (Regression):")
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"R2 Score: {r2:.3f} (Explains {r2*100:.1f}% of variance)")
print(f"Mean Absolute Error: {mae:.2f}%")

# 5. Feature Importance
importances = model.feature_importances_
feature_names = features.columns
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

print("\nTop Influencing Factors on Student Performance:")
print(feature_imp_df.head(10))

# Plot Feature Importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df)
plt.title('Root Cause Analysis: Drivers of Indian Student Performance')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'feature_importance.png'))
print(f"Updated feature_importance.png saved in {script_dir}")

# Save Model and Metadata for Dashboard usage
model_path = os.path.join(script_dir, 'student_performance_model.pkl')
scaler_path = os.path.join(script_dir, 'scaler.pkl')
metadata_path = os.path.join(script_dir, 'model_metadata.pkl')

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

# Save metadata (feature names and encoders) for SHAP explainability
metadata = {
    'feature_names': features.columns.tolist(),
    'encoders': encoders
}
joblib.dump(metadata, metadata_path)

print(f"\nModel and Metadata successfully saved to {script_dir}")
