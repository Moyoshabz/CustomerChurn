# Telco Customer Churn Prediction
This project analyzes customer churn data from a telecommunications company to identify patterns and predict churn using machine learning models. The primary goal is to understand key factors contributing to customer churn and build a predictive model that helps the business retain valuable customers.

## Dataset
- Source: Kaggle - Telco Customer Churn

  https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset contains customer demographic information, account details, service usage patterns, and churn labels.

---
## Project Steps

#### 1. Data Ingestion

Loaded the Telco Customer Churn dataset from Kaggle.

---

#### 2. Data Preprocessing

- Checked for Null & NaN values.

- Removed customerID column - not useful for modeling.

- Converted TotalCharges column from object to float.

- Identified and handled any rows with conversion issues (e.g., blank spaces).

---

#### 3. Exploratory Data Analysis (EDA)

- Visualized feature distributions and correlations with the churn label.

- Analyzed how tenure and monthly charges relate to churn behavior.

- Plotted churn rate across various service features and demographic groups.

---

#### 4. Feature Engineering

- Encoded binary columns using Label Encoding (Yes/No -> 1/0).

- Encoded multi-category columns using One-Hot Encoding.

- Normalized numeric values as needed for certain models (especially neural network).

---

#### 5. Handling Class Imbalance

- Analyzed churn distribution - identified significant class imbalance.

- Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance classes for some models.

---

#### 6. Model Building

- Train-test split: 70% training, 30% testing.

Built the following models:

- Logistic Regression with class weight balancing

- Decision Tree with class weight balancing

- Random Forest with SMOTE oversampling

- Neural Network (Sequential model using Keras)

---

#### 7. Model Evaluation

Evaluated models using:

- Accuracy

- Precision

- Recall

- F1 Score

- ROC AUC Curve (Visualized for all models)

---

### Model Performance Summary

Best Model Based on F1 Score:

```
Model: Logistic Regression
Accuracy: 0.7605
Precision: 0.5394
Recall: 0.8101
F1 Score: 0.6476
```

Why? Logistic Regression achieved the highest recall, making it ideal when the priority is to catch as many churners as possible. It minimizes false negatives, which is essential for proactive customer retention.


Final Selected Model (Balanced Performance):

```
Model: Neural Network
Accuracy: 0.7989
Precision: 0.6207
Recall: 0.6672
F1 Score: 0.6432
```

Why? The Neural Network offers a more balanced performance across all metrics, making it the most robust and generalizable model for real-world deployment, where both precision and recall matter.

----

### Conclusion

- Logistic Regression is best when recall is the top priority - for example, identifying churners at all costs.

- Neural Network is best overall when considering trade-offs between all performance metrics.

- Business decisions will drive final model deployment based on whether catching all churners (recall) or avoiding false alarms (precision) is more important.

### Next Steps

- Hyperparameter tuning (e.g., GridSearchCV for tree-based models, dropout/batch size optimization for NN).

- Add SHAP/feature importance visualizations.

- Integrate into a dashboard or prediction API.

- Continuous model retraining with real-time data.

---

### Technologies Used

- Python (pandas, numpy, scikit-learn, imbalanced-learn, seaborn, matplotlib)

- TensorFlow / Keras (Neural Network)

- Google Colab

