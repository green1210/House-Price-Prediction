# 🏠 House Price Prediction using XGBoost

This project predicts house prices based on property features using **Machine Learning**.  
It uses the **Ames Housing Dataset** and implements an **XGBoost Regressor** for high-accuracy predictions.  

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Technologies Used](#-technologies-used)
- [Project Workflow](#-project-workflow)
- [Results](#-results)
- [Installation & Usage](#-installation--usage)
- [Future Improvements](#-future-improvements)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 📌 Overview
The goal is to **predict house sale prices** given various property attributes such as:
- Lot size
- Year built
- Overall quality
- Neighborhood
- Living area size
- And more...

The project includes:
- Data preprocessing (handling missing values, encoding categories, scaling)
- Model training using **XGBoost**
- Model evaluation (RMSE, R² Score)
- Visualization of predictions and errors
- Saving the trained model for future use

---

## 📂 Dataset
- **Source:** [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Rows:** 1460  
- **Columns:** 81 (including both numerical and categorical features)  
- **Target Variable:** `SalePrice`

---

## 🛠 Technologies Used
- **Python** (3.x)
- **Pandas** – Data handling
- **NumPy** – Numerical operations
- **Matplotlib & Seaborn** – Visualization
- **Scikit-learn** – Preprocessing & metrics
- **XGBoost** – Gradient boosting regression
- **Joblib** – Model persistence

---

## 🔄 Project Workflow

1. **Load Data**
   ```python
   df = pd.read_csv("train.csv")
