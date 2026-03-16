import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import shap

# --- Configuration ---
st.set_page_config(
    page_title="Student Performance AI Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Models & Data ---
@st.cache_resource
def load_models():
    dashboard_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming app.py is in dashboard/ and models are in scripts/
    base_path = os.path.join(dashboard_dir, "..", "scripts")
    model = joblib.load(os.path.join(base_path, "student_performance_model.pkl"))
    scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
    metadata = joblib.load(os.path.join(base_path, "model_metadata.pkl"))
    return model, scaler, metadata

try:
    model, scaler, metadata = load_models()
except FileNotFoundError:
    st.error("Model files or metadata not found. Please run the training script first.")
    st.stop()

@st.cache_data
def get_shap_analysis(student_id, student_row):
    # Prepare individual student data for SHAP
    feature_names = metadata['feature_names']
    encoders = metadata['encoders']
    
    # Create copy and encode
    student_data = pd.DataFrame([student_row])
    student_features = student_data[feature_names].copy()
    
    # Basic encoding matching training logic
    for col, le in encoders.items():
        if col in student_features.columns:
            # Handle potential unseen labels (though student comes from same dataset here)
            try:
                student_features[col] = le.transform(student_features[col].astype(str))
            except:
                # Default to 0 or another logic if unseen
                student_features[col] = 0
                
    # Scale
    X_scaled = scaler.transform(student_features)
    
    # Calculate SHAP values
    # We use TreeExplainer for Random Forest
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    
    # Shap values for our single row
    # In regression, shap_values is an array/list of values
    if isinstance(shap_values, list): # For some versions/multi-output
        sv = shap_values[0][0]
    else:
        sv = shap_values[0]
        
    # Create importance dataframe
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Impact': sv
    }).sort_values(by='Impact', ascending=True)
    
    return shap_df

# --- Helper Functions ---
def predict_score(input_data):
    # Prepare input dataframe matching training features
    # Note: Encoders should ideally be saved/loaded. For this demo, we re-encode simply 
    # or assume input forms match numerical mapping.
    # In production, use the saved LabelEncoders for new inputs!
    
    # We'll use the scaler
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    return prediction[0]

@st.cache_data
def get_association_rules(data):
    # 1. Prepare Data for Mining (Discretize numerical values)
    df_mining = data.copy()
    
    # Bucketize CGPA
    df_mining['Grade_Bucket'] = pd.cut(df_mining['CGPA'], bins=[0, 5, 7, 8.5, 10], labels=['Fail/Low', 'Average', 'Good', 'Excellent'])
    
    # Bucketize Study Hours
    df_mining['Study_Load'] = pd.cut(df_mining['Study_Hours_Week'], bins=[0, 10, 25, 100], labels=['Low Study', 'Moderate Study', 'High Study'])
    
    # Keep only relevant categorical columns for pattern mining
    mining_cols = ['Family_Income', 'Parent_Education', 'Stress_Level', 'Health_Status', 'Grade_Bucket', 'Study_Load', 'Resources_Access', 'Community_Involvement']
    df_mining_cats = df_mining[mining_cols].astype(str)
    
    # One-Hot Encode
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        
        # Transaction format for Apriori
        df_onehot = pd.get_dummies(df_mining_cats)
        
        # Run Apriori
        frequent_itemsets = apriori(df_onehot, min_support=0.05, use_colnames=True)
        
        # Generate Rules
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
        return rules
            
    except ImportError:
        return "Library 'mlxtend' not installed."
    except Exception as e:
        return f"Analysis failed: {e}"

def format_rule_to_english(row):
    ants = [a.replace('_', ' ') for a in list(row['antecedents'])]
    cons = [c.replace('_', ' ') for c in list(row['consequents'])]
    
    # Clean up common keys
    def clean(items):
        res = []
        for item in items:
            item = item.replace('Grade Bucket ', '').replace('Study Load ', '')
            res.append(f"**{item}**")
        return " + ".join(res)

    ant_str = clean(ants)
    cons_str = clean(cons)
    
    # Categorize based on outcome
    theme = "🔍 Academic Insight"
    icon = "📘"
    if any(x in str(cons).lower() for x in ['low', 'average', 'high stress', 'poor']):
        theme = "🚨 Risk Warning"
        icon = "🛑"
    elif any(x in str(cons).lower() for x in ['excellent', 'good', 'low stress']):
        theme = "🌱 Growth Pattern"
        icon = "⭐"

    sentence = f"Students with {ant_str} show a **{row['confidence']*100:.1f}% probability** of having {cons_str}."
    return theme, icon, sentence

# --- Sidebar ---
st.sidebar.title("🎓 Faculty Dashboard")
st.sidebar.info("AI-Powered Student Growth Analysis")

# Initialize Session State for Page Navigation
if 'page' not in st.session_state:
    st.session_state.page = "📊 Class Overview"
if 'selected_student' not in st.session_state:
    st.session_state.selected_student = None

# Sidebar Navigation (Updates Session State)
# We use a callback to sync sidebar with session state
def update_page():
    st.session_state.page = st.session_state.nav_radio

page = st.sidebar.radio("Navigation", 
                        ["📊 Class Overview", "🔍 Student Analysis", "📥 Upload Data"], 
                        index=["📊 Class Overview", "🔍 Student Analysis", "📥 Upload Data"].index(st.session_state.page),
                        key="nav_radio",
                        on_change=update_page)

# --- DATA LOADER ---
# In a real app, this would be the uploaded file.
# For demo, we load the synthetic data we generated.
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "new_student_data.csv")
if os.path.exists(data_path):
    df_raw = pd.read_csv(data_path)
else:
    st.error("Data file not found.")
    st.stop()

# --- PAGES ---

if page == "📊 Class Overview":
    st.title("Class Performance Overview")
    
    # metrics
    avg_cgpa = df_raw['CGPA'].mean()
    at_risk_count = df_raw[df_raw['Performance_Category'] == 'Low/At-Risk'].shape[0]
    avg_attendance = df_raw['Attendance_Rate'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Class CGPA", f"{avg_cgpa:.2f}", delta_color="normal")
    col2.metric("At-Risk Students", f"{at_risk_count}", delta="-2" if at_risk_count > 0 else "0", delta_color="inverse")
    col3.metric("Avg Attendance", f"{avg_attendance:.1f}%")
    col4.metric("Total Students", len(df_raw))
    
    # --- DRILL DOWN: View At-Risk Students ---
    if at_risk_count > 0:
        with st.expander(f"🚨 View List of {at_risk_count} At-Risk Students", expanded=False):
            at_risk_df = df_raw[df_raw['Performance_Category'] == 'Low/At-Risk']
            
            # 1. Create a simplified dataframe for display
            display_df = at_risk_df[['Student_ID', 'Name', 'Branch', 'Current_Year', 'CGPA', 'Attendance_Rate', 'Stress_Level']].reset_index(drop=True)
            
            # 2. Use Dataframe Selection (Streamlit Feature)
            st.write("👇 **Click on a row to analyze that student:**")
            selection = st.dataframe(
                display_df.style.format({'CGPA': '{:.2f}', 'Attendance_Rate': '{:.1f}'}), 
                on_select="rerun", 
                selection_mode="single-row",
                use_container_width=True
            )
            
            # 3. Handle Selection
            if selection.selection.rows:
                idx = selection.selection.rows[0]
                student_id_selected = display_df.iloc[idx]['Student_ID']
                
                # Set Session State to navigate
                st.session_state.selected_student = student_id_selected
                st.session_state.page = "🔍 Student Analysis"
                st.rerun() # Force reload to switch page immediately

    # Visuals

    # Visuals
    st.markdown("### 📈 Growth Trends & Distribution")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Distribution of Categories
        fig_pie = px.pie(df_raw, names='Performance_Category', title='Student Performance Distribution', 
                         color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_chart2:
        # CGPA vs Study Hours (The correlation we found)
        fig_scatter = px.scatter(df_raw, x='Study_Hours_Week', y='CGPA', color='Performance_Category',
                                 title='Impact of Study Hours on CGPA', opacity=0.7)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Branch Comparison
    # Branch Comparison
    st.markdown("### 🏛️ Comparison by Branch")
    branch_perf = df_raw.groupby('Branch')['CGPA'].mean().reset_index().sort_values(by='CGPA')
    fig_bar = px.bar(branch_perf, x='Branch', y='CGPA', color='CGPA', title='Average CGPA by Branch',
                     color_continuous_scale='Viridis')
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Class-Wide Pattern Discovery (Association Rules) ---
    st.markdown("---")
    st.markdown("### 🔗 Class-Wide Pattern Discovery")
    st.markdown("""
    This advanced AI module uses **Association Rule Mining (Apriori)** to find hidden patterns across the entire class. 
    It reveals how deep-rooted Socio-Economic factors (like Income, Education) trigger behavioral issues.
    """)
    
    if st.button("Run Global Analysis"):
        with st.spinner("Mining patterns..."):
            rules = get_association_rules(df_raw)
            if isinstance(rules, pd.DataFrame) and not rules.empty:
                 # Filter for rules implying 'Low Grades' or 'High Stress'
                target_outcomes = {'Grade_Bucket_Fail/Low', 'Study_Load_Low Study', 'Stress_Level_High'}
                
                def is_relevant(row):
                    cons = set(row['consequents'])
                    return len(cons.intersection(target_outcomes)) > 0

                filtered_rules = rules[rules.apply(is_relevant, axis=1)].sort_values(by='confidence', ascending=False)
                
                # Prepare data for table
                table_data = []
                for i, row in filtered_rules.head(10).iterrows():
                    theme, icon, sentence = format_rule_to_english(row)
                    # Strip markdown bolding from sentence for cleaner table view if needed, 
                    # but Streamlit dataframe supports some markdown or we can just keep it.
                    clean_sentence = sentence.replace('**', '') 
                    
                    table_data.append({
                        "Type": f"{icon} {theme}",
                        "AI Observation": clean_sentence,
                        "Confidence (%)": f"{row['confidence']*100:.1f}%"
                    })
                
                if table_data:
                    st.table(table_data)
                    st.info("💡 **Tip:** Higher confidence means the AI is more certain about this specific pattern.")
                else:
                    st.write("No significant patterns found for the selected outcomes.")
            else:
                st.write(rules)


elif page == "🔍 Student Analysis":
    st.title("Individual Student Report")
    
    student_list = df_raw['Student_ID'].tolist()
    
    # Auto-select if coming from Drill Down
    default_index = 0
    if st.session_state.selected_student in student_list:
        default_index = student_list.index(st.session_state.selected_student)
    
    selected_id = st.selectbox("Search Student ID", student_list, index=default_index)
    
    if selected_id:
        student_data = df_raw[df_raw['Student_ID'] == selected_id].iloc[0]
        
        # Header Profile
        col_prof1, col_prof2 = st.columns([1, 3])
        with col_prof1:
            st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=120)
        with col_prof2:
            st.subheader(f"{student_data['Name']} ({selected_id})")
            st.write(f"**Branch:** {student_data['Branch']} | **Year:** {student_data['Current_Year']}")
            st.write(f"**Current Status:** {student_data['Performance_Category']}")
            
        # Growth Pattern Chart
        st.markdown("---")
        st.markdown("### 🚀 Academic Growth Trajectory")
        
        # Extract Semester GPAs
        sem_cols = [f'Sem_{i}_GPA' for i in range(1,9)]
        student_grades = student_data[sem_cols]
        # Filter None
        y_values = [val for val in student_grades.values if pd.notna(val)]
        x_values = [f"Sem {i+1}" for i in range(len(y_values))]
        
        if len(y_values) > 0:
            fig_line = go.Figure()
            # Student Line
            fig_line.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines+markers', name='Student GPA',
                                          line=dict(color='#00CC96', width=4)))
            
            # Class Average Line (for context)
            class_avg = df_raw[sem_cols].mean().values[:len(y_values)]
            fig_line.add_trace(go.Scatter(x=x_values, y=class_avg, mode='lines', name='Class Average',
                                          line=dict(color='gray', width=2, dash='dot')))
            
            fig_line.update_layout(title="Semester-wise Performance Trend", yaxis_title="GPA", yaxis_range=[0, 10])
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.warning("No academic history available for 1st Semester students yet.")


        # Root Cause Analysis (Factors)
        st.markdown("### 🧩 Key Influencing Factors")
        
        col_fact1, col_fact2, col_fact3 = st.columns(3)
        
        with col_fact1:
            st.info("📚 **Study Habits**")
            st.metric("Study Hours/Week", f"{student_data['Study_Hours_Week']} hrs")
            if student_data['Study_Hours_Week'] < 10:
                st.error("⚠️ Low Study Time")
            else:
                st.success("✅ Good Consistency")
                
        with col_fact2:
            st.info("🏫 **Attendance**")
            st.metric("Attendance Rate", f"{student_data['Attendance_Rate']}%")
            if student_data['Attendance_Rate'] < 75:
                st.error("⚠️ Attendance Alert!")
                
        with col_fact3:
            st.info("🧠 **Well-being**")
            st.write(f"**Stress Level:** {student_data['Stress_Level']}")
            st.write(f"**Health:** {student_data['Health_Status']}")

        # --- PERSONALIZED ROOT CAUSE ANALYSIS (Apriori Match) ---
        st.markdown("---")
        st.markdown("### 🧬 Exact Root Cause for this Student (AI Pattern Matching)")
        
        # 1. Get Rules (Cached)
        rules = get_association_rules(df_raw)
        
        if isinstance(rules, str): # Error message
            st.warning(rules)
        elif rules.empty:
            st.info("No significant causal patterns found for this class dataset yet.")
        else:
            # 2. Match Student to Rules
            # We look for rules where this student matches the ANTECEDENTS (Causes)
            # and the rule predicts a negative outcome.
            
            student_profile = student_data.astype(str).to_dict()
            # Add buckets manually to match rule format
            if student_data['CGPA'] < 5: student_profile['Grade_Bucket'] = 'Fail/Low'
            elif student_data['CGPA'] < 7: student_profile['Grade_Bucket'] = 'Average'
            elif student_data['CGPA'] < 8.5: student_profile['Grade_Bucket'] = 'Good'
            else: student_profile['Grade_Bucket'] = 'Excellent'
            
            if student_data['Study_Hours_Week'] <= 10: student_profile['Study_Load'] = 'Low Study'
            elif student_data['Study_Hours_Week'] <= 25: student_profile['Study_Load'] = 'Moderate Study'
            else: student_profile['Study_Load'] = 'High Study'

            found_cause = False
            
            # Filter for rules that lead to NEGATIVE things
            negative_outcomes = {'Grade_Bucket_Fail/Low', 'Grade_Bucket_Average', 'Stress_Level_High', 'Study_Load_Low Study'}
            
            for i, row in rules.iterrows():
                ants = list(row['antecedents'])
                cons = list(row['consequents'])
                
                # Check if Consequent is negative
                if not any(c in negative_outcomes for c in cons):
                    continue
                
                # Check if Student matches Antecedents (The Cause)
                match = True
                for condition in ants:
                    key, val = condition.split('_', 1)
                    # Handle special bucket names if needed, or simple match
                    if key in student_profile:
                        if student_profile[key] != val:
                            match = False
                            break
                    else:
                        match = False # Column not in student profile
                
                if match:
                    found_cause = True
                    confidence = row['confidence'] * 100
                    lift = row['lift']
                    
                    ant_display = " + ".join([a.replace('_', ': ') for a in ants])
                    cons_display = " + ".join([c.replace('_', ': ') for c in cons])
                    
                    st.error(f"🔴 **Root Cause Identified:** {ant_display}")
                    st.write(f"This specific combination represents a **{confidence:.1f}% probability** of causing **{cons_display}**.")
                    
                    # Socio-economic highlighter
                    if 'Family' in ant_display or 'Parent' in ant_display or 'Resources' in ant_display:
                        st.caption(f"💡 **Insight:** This is a Structural/Socio-Economic barrier, not just behavioral.")
            
            if not found_cause:
                st.success("No hidden risk patterns matched for this student. Their performance drivers appear to be standard (Study Hours/Attendance).")

        # Recommendation Engine
        st.markdown("### 💡 AI Recommendations")
        recs = []
        if student_data['Attendance_Rate'] < 75:
            recs.append("- 🔴 **Immediate Intervention:** Attendance is below 75%. Schedule a counseling session.")
        if student_data['Study_Hours_Week'] < 15 and student_data['CGPA'] < 7.0:
            recs.append("- 🟡 **Academic:** Increase self-study hours. Consider peer study groups.")
        if student_data['Stress_Level'] == 'High':
            recs.append("- 🔵 **Support:** Student indicates high stress. Refer to student well-being center.")
            
        if not recs:
            st.success("🌟 This student is on a great track! Keep up the good work.")
        else:
            for r in recs:
                st.markdown(r)

        # --- SHAP ANALYSIS DEEP DIVE ---
        st.markdown("---")
        st.markdown("### 🧠 Deep Dive: Individual Impact Analysis")
        st.markdown("This AI explanation reveals which specific factors are pushing this student's performance up or down.")
        
        if st.button("🔍 Generate AI Performance Breakdown"):
            with st.spinner("Calculating factor contributions..."):
                shap_df = get_shap_analysis(selected_id, student_data)
                
                # Colors based on existing theme
                # Success Green: #00CC96, Alert Red: #EF553B
                shap_df['Color'] = shap_df['Impact'].apply(lambda x: '#00CC96' if x > 0 else '#EF553B')
                shap_df['Direction'] = shap_df['Impact'].apply(lambda x: 'Positive (+)' if x > 0 else 'Negative (-)')

                fig_shap = px.bar(
                    shap_df,
                    x='Impact',
                    y='Feature',
                    orientation='h',
                    color='Direction',
                    color_discrete_map={'Positive (+)': '#00CC96', 'Negative (-)': '#EF553B'},
                    title=f"What is driving {student_data['Name']}'s predicted score?",
                    labels={'Impact': 'Impact on Predicted Score', 'Feature': 'Factor'}
                )
                
                # Layout adjustments for transparency and theme matching
                fig_shap.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#ffffff',
                    showlegend=True,
                    height=500
                )
                fig_shap.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                fig_shap.update_yaxes(showgrid=False)
                
                st.plotly_chart(fig_shap, use_container_width=True)
                st.info("💡 **How to read:** Bars pointing to the right (Green) are strengths helping the student. Bars pointing to the left (Red) are barriers reducing their predicted score.")

elif page == "📥 Upload Data":
    st.title("Upload Class Data")
    st.write("Upload a CSV file containing student records (Branch/Year wise).")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        df_upload = pd.read_csv(uploaded_file)
        st.write("Preview:")
        st.dataframe(df_upload.head())
        
        if st.button("Analyze Uploaded Data"):
            st.warning("Analysis module for custom uploads would connect here in production.")
