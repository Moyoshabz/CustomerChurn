# Telco Customer Churn Prediction
This project analyzes customer churn data from a telecommunications company to identify patterns and predict churn using machine learning models. The primary goal is to understand key factors contributing to customer churn and build a predictive model that helps the business retain valuable customers.

## Dataset
Source: Kaggle - Telco Customer Churn

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset contains customer demographic information, account details, service usage patterns, and churn labels.

---
## Project Steps

#### 1. Data Ingestion

Loaded the Telco Customer Churn dataset from Kaggle.

---

#### 2. Data Preprocessing

Checked for Null & NaN values.

Removed customerID column – not useful for modeling.

Converted TotalCharges column from object to float.

Identified and handled any rows with conversion issues (e.g., blank spaces).

---

#### 3. Exploratory Data Analysis (EDA)

Visualized feature distributions and correlations with the churn label.

Analyzed how tenure and monthly charges relate to churn behavior.

Plotted churn rate across various service features and demographic groups.

---

#### 4. Feature Engineering

Encoded binary columns using Label Encoding (e.g., Yes/No → 1/0).

Encoded multi-category columns using One-Hot Encoding.

Normalized numeric values as needed for certain models (especially neural network).

---

#### 5. Handling Class Imbalance

Analyzed churn distribution — identified significant class imbalance.

Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance classes for some models.

---

#### 6. Model Building

Train-test split: 70% training, 30% testing.

Built the following models:

Logistic Regression with class weight balancing

Decision Tree with class weight balancing

Random Forest with SMOTE oversampling

Neural Network (Sequential model using Keras)

---

#### 7. Model Evaluation
Evaluated models using:

Accuracy

Precision

Recall

F1 Score

ROC AUC Curve (Visualized for all models)
