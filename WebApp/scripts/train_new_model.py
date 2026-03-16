
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# 1. Load Data
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "new_student_data.csv")

if not os.path.exists(data_path):
    print(f"Data file not found at {data_path}. Please ensure new_student_data.csv exists.")
    exit()

df = pd.read_csv(data_path)
print("Data Loaded Successfully!")
print(df.head())

# 2. Feature Engineering & Preprocessing
# Goal: Find major influencing factors to CGPA (Root Cause Analysis)
# We remove 'CGPA' and semester grades from INPUTS to see what *drives* them.

# Drop identifiers AND Academic Outputs (to avoid data leakage)
drop_cols = ['Student_ID', 'Name', 'Performance_Category', 'CGPA'] 
# Also drop individual semester grades as they are just parts of the CGPA
for i in range(1, 9):
    drop_cols.append(f'Sem_{i}_GPA')

features = df.drop(drop_cols, axis=1, errors='ignore')
target = df['CGPA'] # New Target: Predict the Score directly

print("Training features:", features.columns.tolist())

# Encode Categorical Variables
categorical_cols = features.select_dtypes(include=['object']).columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col])
    encoders[col] = le # Save for later if needed

# Scale Numerical Variables
scaler = StandardScaler()
X = scaler.fit_transform(features)
y = target

# 3. Model Training (Regression)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluation
y_pred = model.predict(X_test)
print("\nModel Performance (Regression):")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f} (Explains {r2_score(y_test, y_pred)*100:.1f}% of variance)")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")

# 5. Feature Importance
importances = model.feature_importances_
feature_names = features.columns
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

print("\nTop Influencing Factors on CGPA:")
print(feature_imp_df)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df)
plt.title('What Drives Student CGPA? (Factor Analysis)')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Updated feature_importance.png saved.")

# Save Model and Metadata for Dashboard
model_path = os.path.join(script_dir, 'student_performance_model.pkl')
scaler_path = os.path.join(script_dir, 'scaler.pkl')
metadata_path = os.path.join(script_dir, 'model_metadata.pkl')

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

# Save metadata (feature names and encoders)
metadata = {
    'feature_names': features.columns.tolist(),
    'encoders': encoders
}
joblib.dump(metadata, metadata_path)

print(f"Model and Metadata saved to {script_dir}")
