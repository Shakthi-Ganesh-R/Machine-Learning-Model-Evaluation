# Bank Marketing Classification – Machine Learning Project

## Project Description
This project builds and evaluates multiple machine learning classification models using the UCI Bank Marketing Dataset.

The objective is to predict whether a customer will subscribe to a term deposit (`y = yes/no`) based on demographic and campaign-related features.

The models are evaluated using multiple performance metrics and deployed using Streamlit.

---

## Dataset Information
- Dataset Name: Bank Marketing Dataset  
- Source: UCI Machine Learning Repository  
- Link: https://archive.ics.uci.edu/dataset/222/bank+marketing  
- Problem Type: Binary Classification  
- Records: 45,211  
- Features: 16 input features  
- Target: `y` (yes / no)

---

## Data Preprocessing
- One-hot encoding for categorical features  
- Standard scaling for numerical features  
- Target encoding:
  - `yes` → 1  
  - `no` → 0  

---

## Models Implemented
1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes  
5. Random Forest  
6. XGBoost  

---

## Evaluation Metrics
- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  
- Confusion Matrix  
- Classification Report  

---

## Streamlit Features
- Upload CSV dataset  
- Automatic detection of target column `y`  
- Model selection dropdown  
- Display evaluation metrics  
- Confusion matrix visualization  
- Classification report  

## Conclusion
This project demonstrates end-to-end ML model training, evaluation, and deployment using Streamlit. Ensemble models such as Random Forest and XGBoost generally provide strong performance on this dataset.
