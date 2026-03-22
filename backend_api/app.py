from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import shap

app = Flask(__name__)
CORS(app)

# Load Models & Metadata
model_path = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")
metadata_path = os.path.join(os.path.dirname(__file__), "models", "model_metadata.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "models", "scaler.pkl")

# We use try-except around loading to gracefully handle missing files in dev
try:
    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    scaler = joblib.load(scaler_path)
    
    feature_names = metadata['feature_names']
    encoders = metadata['encoders']
    
    # Initialize the SHAP Explainer using the XGBoost Model
    explainer = shap.TreeExplainer(model)
    print("✅ Model, Metadata, Scaler, and SHAP Explainer successfully loaded.")
except Exception as e:
    print(f"⚠️ Error loading models: {e}")
    model, metadata, scaler, feature_names, encoders, explainer = None, None, None, None, None, None

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Student Performance Prediction API & XAI Engine is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        
        # 1. Parse JSON into DataFrame matching exactly the expected feature order
        df_input = pd.DataFrame([data])
        
        # Ensure all features exist in df_input
        for f in feature_names:
            if f not in df_input.columns:
                df_input[f] = 0 # or some default

        df_input = df_input[feature_names]

        # 2. Encode categorical variables using the loaded encoders
        for col, le in encoders.items():
            if col in df_input.columns:
                # Handle unseen labels gracefully by mapping them to the mode/most frequent class, 
                # but since we don't have the original mode, we map to index 0.
                df_input[col] = df_input[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )

        # 3. Scale numerical variables
        X_scaled = scaler.transform(df_input)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
        
        # 4. Predict
        prediction = model.predict(X_scaled_df)[0]
        
        # 5. Local XAI (SHAP Local Context)
        shap_values_local = explainer.shap_values(X_scaled_df)
        
        # Construct the Impact Factors list for the React Frontend Waterfall Chart
        # shap_values_local[0] is an array of impact values for each feature
        impact_factors = []
        for i, feature in enumerate(feature_names):
            impact_factors.append({
                "name": feature,
                "impact": float(shap_values_local[0][i])
            })
            
        # Sort by absolute impact to show the biggest drivers first
        impact_factors.sort(key=lambda x: abs(x["impact"]), reverse=True)
        # But React expects them sorted by raw impact (negative to positive) for the waterfall
        impact_factors.sort(key=lambda x: x["impact"])

        # SHAP Base Value (the average prediction across the sample)
        base_val = explainer.expected_value
        if isinstance(base_val, np.ndarray): base_val = float(base_val[0])
        else: base_val = float(base_val)

        return jsonify({
            "predicted_percentage": float(prediction),
            "shap_local": impact_factors,
            "base_value": base_val
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/global_shap", methods=["POST"])
def global_shap():
    try:
        # Instead of storing the massive X_train in memory, we ask the frontend 
        # to send a representative sample of strictly numerical/encoded data, 
        # OR we just use a small mock sample if they send the raw JSON dashboard data.
        
        # Since the frontend sends the raw JSON array of all students:
        raw_data = request.json
        if not raw_data or len(raw_data) == 0:
            return jsonify({"error": "No data provided"}), 400
            
        # Limit to 500 records for speed
        df_raw = pd.DataFrame(raw_data[:500])
        
        # Ensure correct columns
        for f in feature_names:
            if f not in df_raw.columns:
                df_raw[f] = 0
                
        df_raw = df_raw[feature_names]
        
        # Encode
        for col, le in encoders.items():
            if col in df_raw.columns:
                df_raw[col] = df_raw[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
                
        # Scale
        X_scaled = scaler.transform(df_raw)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
        
        # Calculate Global SHAP
        shap_values_global = explainer.shap_values(X_scaled_df) # shape: (500, num_features)
        
        # Calculate Mean Absolute SHAP Value per feature
        mean_abs_shap = np.abs(shap_values_global).mean(axis=0)
        
        global_impacts = []
        for i, feature in enumerate(feature_names):
            global_impacts.append({
                "name": feature,
                "importance": float(mean_abs_shap[i])
            })
            
        # Sort by importance descending
        global_impacts.sort(key=lambda x: x["importance"], reverse=True)
        
        # Return Summary and Raw Distribution for Beeswarm Plot
        # Limit raw distribution for performance
        raw_distribution = []
        top_N = min(200, len(X_scaled_df))
        
        # Scale X_scaled_df for color mapping (0 to 1)
        X_normalized = (X_scaled_df - X_scaled_df.min()) / (X_scaled_df.max() - X_scaled_df.min() + 1e-9)
        
        for i in range(top_N):
            student_data = {}
            for j, feature in enumerate(feature_names):
                student_data[feature] = {
                    "shap": float(shap_values_global[i][j]),
                    "value": float(X_normalized.iloc[i][feature])
                }
            raw_distribution.append(student_data)

        return jsonify({
            "global_shap": global_impacts[:15],
            "raw_distribution": raw_distribution
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

from mlxtend.frequent_patterns import apriori, association_rules

@app.route("/mine_patterns", methods=["POST"])
def mine_patterns():
    try:
        data = request.json
        df = pd.DataFrame(data)
        
        # 1. Discretization for Association Rule Mining
        # We want to convert numerical values to categorical bins for Apriori
        df_rules = pd.DataFrame()
        
        if 'Monthly_Income_INR' in df.columns:
            df_rules['Income_Low'] = df['Monthly_Income_INR'] < 20000
        if 'Attendance_Percentage' in df.columns:
            df_rules['Attendance_Low'] = df['Attendance_Percentage'] < 75
        if 'Previous_Percentage' in df.columns:
            df_rules['Score_Low'] = df['Previous_Percentage'] < 65
        if 'Stress_Level' in df.columns:
            df_rules['Stress_High'] = df['Stress_Level'].isin(['High'])
        if 'Study_Hours_Per_Day' in df.columns:
            df_rules['Study_Low'] = df['Study_Hours_Per_Day'] < 3
        if 'Internet_Access' in df.columns:
            df_rules['No_Internet'] = df['Internet_Access'] == 'No'
            
        # 2. Apply Apriori
        frequent_itemsets = apriori(df_rules, min_support=0.05, use_colnames=True)
        
        if len(frequent_itemsets) == 0:
            return jsonify({"patterns": []})
            
        # 3. Generate Rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5, num_itemsets=len(frequent_itemsets))
        
        # Filter rules to only those that lead to "Score_Low"
        target_rules = rules[rules['consequents'] == frozenset({'Score_Low'})]
        
        # Sort by confidence
        target_rules = target_rules.sort_values(by="confidence", ascending=False).head(5)
        
        patterns = []
        for _, row in target_rules.iterrows():
            antecedents = list(row['antecedents'])
            cause = " AND ".join(antecedents).replace("_", " ")
            patterns.append({
                "cause": cause,
                "effect": "Low Academic Score",
                "confidence": round(row['confidence'] * 100, 1),
                "support": round(row['support'] * 100, 1),
                "lift": round(row['lift'], 2)
            })
            
        return jsonify({"patterns": patterns})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
