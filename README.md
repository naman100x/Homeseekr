# 🏠 HomeSeekr - House Price Prediction Model

HomeSeekr is a machine learning project that predicts house prices based on various input features using a Random Forest model. The project leverages data preprocessing, feature scaling, and model evaluation techniques to deliver accurate predictions.

---

## 📊 Overview

The goal of this project is to build a regression model that can predict the sale price of a house given its attributes (e.g., area, location, number of rooms, etc.).

The project includes:

- Data preprocessing and cleaning
- Feature scaling using `StandardScaler`
- Model building using `RandomForestRegressor`
- Model evaluation using R² Score and Mean Absolute Error

---

## 🔧 Technologies Used

- Python
- Jupyter Notebook
- scikit-learn
- pandas
- numpy
- matplotlib / seaborn (optional for visualization)

---

## 🧠 Model Pipeline

1. **Data Loading**  
   Load dataset using `pandas`.

2. **Preprocessing**  
   - Handle missing values  
   - Encode categorical variables (if applicable)  
   - Feature scaling using `StandardScaler`

3. **Train-Test Split**  
   Data is split using `train_test_split` (80/20 ratio).

4. **Model Training**  
   A `RandomForestRegressor` is trained on scaled data.

5. **Prediction and Evaluation**  
   Predictions are compared with actual values using:
   - R² Score
   - Mean Absolute Error (MAE)

---
