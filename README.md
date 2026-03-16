# AI-Powered Student Performance Diagnostic System 🎓
### *Leveraging Machine Learning & Explainable AI (XAI) for Educational Intervention*

---

## 🚀 Project Vision
This is not just a grade predictor; it is a **Clinical Diagnostic Tool for Educators**. While traditional systems focus on "what" a student might score, this project uses **Explainable AI (SHAP)** and **Association Rule Mining (Apriori)** to answer the "why"—identifying root causes like socioeconomic stressors, environmental barriers, and behavioral patterns.

---

## � Project Lifecycle & Methodology

### Phase 1: Data Engineering & Environment Modeling
We established a robust data pipeline that captures 25+ features across four dimensions:
- **Demographics & Socio-economics**: Age, Location, Monthly Family Income.
- **Academic Foundation**: Attendance %, Study Hours, Previous Performance.
- **Support Systems**: Scholarship status, Internet access, Tuition support.
- **Well-being Indicators**: Stress levels, Sleep patterns.

### Phase 2: Model Benchmarking & Selection
After benchmarking multiple algorithms (Linear Regression, XGBoost, SVR), the **Random Forest Regressor** was selected as our core engine. 
- **Reasoning**: It captures complex, non-linear interactions between variables (e.g., how the impact of "Internet Access" changes based on "Location") without overfitting.

### Phase 3: Intelligence Layer (The "Brain")
- **Explainable AI (SHAP)**: We integrated SHAP (SHapley Additive exPlanations) to decompose every prediction. This allows us to see exactly which factor (e.g., -12.1% due to Stress) pushed a student's score down.
- **Pattern discovery (Apriori)**: Beyond individual students, we run the Apriori algorithm across the entire batch to discover institutional patterns—like identifying that "Low Attendance + High Stress" is a recurring failure pattern for a specific income group.

---

## 💻 System Architecture

The project is split into two specialized environments:

### 1. **Research Dashboard (Streamlit)**
*Located in `/WebApp/dashboard/`*
- **Purpose**: Deep-dive data science research.
- **Features**: Interactive SHAP plots, model training logs, and raw feature distribution analysis.

### 2. **Production/Deployment App (Vite + React)**
*Located in `/WebApp/dashboard/dashboard-app/`*
- **Purpose**: A high-performance, deployment-ready interface for administrators.
- **Features**:
  - **Individual Student Diagnostic**: Visual "Impact Analysis" charts showing drivers of performance.
  - **AI Counselor Recommendation Engine**: Automated, data-driven intervention advice.
  - **Global Analysis Engine**: One-click pattern mining across the dataset.

---

## 📁 Directory Structure
```text
StudentPerformanceAnalysis/
├── WebApp/
│   ├── dashboard/              # Core Intelligence Logic
│   │   ├── app.py              # Streamlit Research UI
│   │   └── dashboard-app/      # Vite-based Production UI
│   ├── scripts/                # Data pipelines & Model training
│   │   ├── train_new_model.py  # Automated Model Benchmarking
│   │   └── generate_data.py    # Environment Modeling Engine
│   └── data/                   # Digitized Student Records
├── requirements.txt            # Python Backend Dependencies
└── README.md                   # Project Documentation
```

---

## � Installation & Setup

### Backend & Streamlit
1. **Environment Setup**:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Launch Research App**:
   ```bash
   streamlit run WebApp/dashboard/app.py
   ```

### Frontend (Vite)
1. **Navigate to App**:
   ```bash
   cd WebApp/dashboard/dashboard-app
   ```
2. **Install & Start**:
   ```bash
   npm install
   npm run dev
   ```

---

## 📊 Performance Benchmark
| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| **Random Forest** | **0.466** | **0.555** | **0.160** |
| XGBoost | 0.448 | 0.609 | -0.012 |
| SVR | 0.506 | 0.621 | -0.054 |

---

## 🤝 Project Authorship
**Developed as a Final Year Major Project**
- **Lead Developer**: [Pawnesh Kumar](https://github.com/prince083)
- **Domain**: Machine Learning & Educational Data Mining (EDM)

---

> [!TIP]
> **Why Random Forest?**
> We used an ensemble approach to ensure that the "Feature Importance" derived is stable across different student populations, making our "Diagnostic" mode statistically significant.
