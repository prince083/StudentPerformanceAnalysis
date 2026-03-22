# AI-Powered Student Performance Diagnostic System 🎓
### *Leveraging Machine Learning & Explainable AI (XAI) for Educational Intervention*

---

## 🚀 Project Vision
This is a **Clinical Diagnostic Tool for Educators**. While traditional systems focus on "what" a student might score, this project uses **Explainable AI (SHAP)** and **Association Rule Mining (Apriori)** to answer the "why"—identifying root causes like socioeconomic stressors, environmental barriers, and behavioral patterns.

---

## 💻 System Architecture
The project is split into three core specialized environments for maximum performance and modularity:

### 1. **Intelligence Engine (Python Backend API)**
*Located in `/backend_api/`*
- **Purpose**: Server-side processing of complex AI logic.
- **Features**: Generates real-time **SHAP Beeswarm** distributions, **Local Impact factors**, and **Association rules** for students.

### 2. **Professional Admin Dashboard (React + Vite)**
*Located in `/frontend/react_dashboard/`*
- **Purpose**: The primary high-performance interface for faculty and administrators.
- **Features**: 
  - **Individual AI Diagnostics**: Waterfall-style impact analysis.
  - **Global Context Panel**: Real-time SHAP Beeswarm plots for identifying class-wide trends.
  - **Premium Dashboard UI**: Built with a "Lumière Commerce" aesthetic (Glassmorphism & High-Contrast).

### 3. **Research & Benchmarking Dashboard (Streamlit)**
*Located in `/frontend/streamlit_dashboard/`*
- **Purpose**: Statistical deep-dives and model research.
- **Features**: Interactive scientific plots, raw dataset exploration, and XAI distribution analysis.

---

## 📁 Directory Structure
```text
StudentPerformanceAnalysis/
├── ml_pipeline/              # Module 1: The Model Factory
│   ├── scripts/              # Preprocessing & Training Logic
│   └── data/                 # Raw & Preprocessed datasets
├── backend_api/              # Module 2: The Intelligence Engine (Flask)
│   ├── models/               # Production Model Weights (.pkl)
│   └── app.py                # Main Prediction & XAI API
├── frontend/                 # Module 3: The Presentation Layer
│   ├── react_dashboard/      # High-end Admin UI (Vite/Tailwind)
│   └── streamlit_dashboard/  # Scientific Research UI
├── myVenv/                   # Python Virtual Environment
└── requirements.txt          # Global project dependencies
```

---

## 🛠️ Installation & Setup (Quick Guide)

### 1. Prerequisites
- Python 3.10+
- Node.js (for React Dashboard)

### 2. Environment Setup
```bash
# Clone the repo and enter it
cd StudentPerformanceAnalysis

# Create and activate virtual environment
python -m venv myVenv
# Windows:
myVenv\Scripts\activate
# Linux/Mac:
source myVenv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### 3. Running the Intelligence Backend (Step 1)
The backend MUST be running for the dashboards to show AI insights.
```bash
cd backend_api
python app.py
# Server will start on http://127.0.0.1:5000
```

### 4. Running the React Dashboard (Step 2)
```bash
cd frontend/react_dashboard
npm install
npm run dev
# Dashboard launches on http://localhost:5173
```

### 5. Running the Streamlit Research App (Optional)
```bash
cd frontend/streamlit_dashboard
streamlit run app.py
```

---

## 📊 Performance Benchmark
| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| **XGBoost / Random Forest** | **~0.4** | **~0.5** | **0.8+** |
| Linear Regression | 0.448 | 0.609 | 0.12 |

---

## 🤝 Project Authorship
**Developed as a Final Year Major Project**
- **Lead Developer**: [Pawnesh Kumar](https://github.com/prince083)
- **Domain**: Machine Learning & Educational Data Mining (EDM)

---

> [!TIP]
> **Data Loading**: Ensure the CSV dataset is uploaded via the **"Upload Data"** tab in the dashboard if you want to perform real-time analysis on a specific student cohort.
