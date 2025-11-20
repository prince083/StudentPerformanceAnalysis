# Student Performance Analysis

A machine learning project that analyzes and predicts student academic performance using various regression models. This project compares multiple machine learning algorithms to identify the best model for predicting student grade averages.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a comprehensive machine learning pipeline for predicting student academic performance. It includes data preprocessing, feature engineering, model training, and evaluation using multiple regression algorithms including Linear Regression, Decision Tree, Random Forest, SVR, and XGBoost.

## ✨ Features

- **Data Preprocessing**: Handles missing values, encodes categorical variables, and normalizes numerical features
- **Multiple Models**: Compares 5 different regression models
- **Model Evaluation**: Comprehensive evaluation using MAE, MSE, RMSE, and R² score
- **Best Model Selection**: Automatically identifies and saves the best performing model
- **Visualizations**: Generates comparison plots and prediction visualizations

## 📁 Project Structure

```
StudentPerformanceAnalysis/
│
├── data/
│   ├── dataset.csv                    # Original dataset
│   └── preprocessed/
│       ├── X_train.csv                # Training features
│       ├── X_test.csv                 # Testing features
│       ├── y_train.csv                # Training target
│       ├── y_test.csv                 # Testing target
│       └── model_performance.csv      # Model comparison results
│
├── notebooks/
│   ├── data_preprocessing.ipynb       # Data cleaning and preprocessing notebook
│   └── model_training.ipynb           # Model training and evaluation notebook
│
├── dashboard/
│   └── best_model.pkl                 # Saved best model (Random Forest)
│
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

## 🔧 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+** (Python 3.12 recommended)
- **pip** (Python package installer)
- **Git** (for cloning the repository)
- **Jupyter Notebook** or **JupyterLab** (will be installed via requirements.txt)

## 📦 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/StudentPerformanceAnalysis.git
cd StudentPerformanceAnalysis
```

### Step 2: Create a Virtual Environment

It's recommended to use a virtual environment to isolate project dependencies:

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- jupyter
- jupyterlab

### Step 4: Verify Installation

Launch Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

## 🚀 Usage

### Running the Complete Pipeline

1. **Data Preprocessing**:
   - Open `notebooks/data_preprocessing.ipynb` in Jupyter
   - Run all cells to preprocess the dataset
   - This will generate the preprocessed data files in `data/preprocessed/`

2. **Model Training**:
   - Open `notebooks/model_training.ipynb` in Jupyter
   - Run all cells to train and evaluate all models
   - The best model will be automatically saved to `dashboard/best_model.pkl`

### Notebook Execution Order

1. First, run `data_preprocessing.ipynb` completely
2. Then, run `model_training.ipynb`

**Important**: Make sure to run the preprocessing notebook first, as the training notebook depends on the preprocessed data files.

### Loading and Using the Trained Model

```python
import pickle
import pandas as pd

# Load the saved model
with open('dashboard/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare your data (preprocessed in the same way as training data)
# Then make predictions
predictions = model.predict(your_preprocessed_data)
```

## 📊 Results

The project evaluates models using multiple metrics:

| Model | MAE | MSE | RMSE | R² Score |
|-------|-----|-----|------|----------|
| **Random Forest** | 0.466 | 0.308 | 0.555 | **0.160** |
| XGBoost Regressor | 0.448 | 0.371 | 0.609 | -0.012 |
| SVR | 0.506 | 0.386 | 0.621 | -0.054 |
| Linear Regression | 0.567 | 0.456 | 0.676 | -0.246 |
| Decision Tree | 0.565 | 0.626 | 0.791 | -0.708 |

**Best Model**: Random Forest Regressor (Highest R² Score)

## 🛠 Technologies Used

- **Python 3.12**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms and utilities
- **xgboost** - Gradient boosting framework
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **Jupyter Notebook** - Interactive development environment

## 📝 Dataset Information

The dataset contains 504 student records with 25 features including:
- Demographic information (Age, Gender, Marital Status)
- Academic information (Course, Application Mode, Semester Units)
- Family background (Parental Education, Income Level)
- Economic factors (Unemployment Rate, Inflation Rate, Regional GDP)
- Academic performance metrics (Grade Average, Attendance)

**Target Variable**: Grade Average

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Your Name**
- GitHub: [@Pawnesh-Kumar](https://github.com/prince083)

## 🙏 Acknowledgments

- Dataset providers
- Open source community for the amazing tools and libraries

---

**Note**: Make sure you have activated your virtual environment before running any commands or notebooks. If you encounter any issues during installation or execution, please check that all prerequisites are met and dependencies are correctly installed.

