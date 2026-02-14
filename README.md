# Machine Learning Classification models and Evaluation metrics

## Project Statement
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

### Data Preprocessing
- One-hot encoding for categorical features  
- Standard scaling for numerical features  
- Target encoding:
  - `yes` = 1  
  - `no` = 0  

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

## Model Performance Comparison

| ML Model Name              | Accuracy | AUC    | Precision | Recall  | F1 Score | MCC    |
|----------------------------|----------|--------|-----------|---------|----------|--------|
| Logistic Regression        | 0.9012   | 0.9054 | 0.8865    | 0.9012  | 0.8880   | 0.4264 |
| Decision Tree              | 0.8777   | 0.7135 | 0.8800    | 0.8777  | 0.8788   | 0.4191 |
| KNN                        | 0.8952   | 0.8422 | 0.8762    | 0.8952  | 0.8773   | 0.3665 |
| Naive Bayes                | 0.8639   | 0.8088 | 0.8720    | 0.8639  | 0.8677   | 0.3797 |
| Random Forest (Ensemble)   | 0.9045   | 0.9272 | 0.8916    | 0.9045  | 0.8934   | 0.4561 |
| XGBoost (Ensemble)         | 0.9080   | 0.9291 | 0.9007    | 0.9080  | 0.9033   | 0.5149 |

---

## Model Performance Observations

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| Logistic Regression | Performed strongly with high accuracy (0.9012) and AUC (0.9054). It provides stable and reliable predictions. However, the MCC (0.4264) indicates moderate performance in handling class imbalance. |
| Decision Tree | Achieved good accuracy (0.8777) but lower AUC (0.7135), indicating weaker probability estimation. It may overfit the training data, resulting in slightly lower generalization performance. |
| KNN | Showed competitive accuracy (0.8952) but relatively lower MCC (0.3665). Performance may be affected by feature scaling and high dimensionality due to one-hot encoding. |
| Naive Bayes | Produced moderate performance with decent AUC (0.8088). Since it assumes feature independence, performance is slightly limited for this dataset where features may be correlated. |
| Random Forest (Ensemble) | Delivered strong performance across all metrics with improved AUC (0.9272) and MCC (0.4561). Ensemble learning helped reduce overfitting and improved overall robustness. |
| XGBoost (Ensemble) | Achieved the best overall performance with highest accuracy (0.9080), AUC (0.9291), F1 score (0.9033), and MCC (0.5149). It effectively handles class imbalance and captures complex feature interactions. |

---

## Streamlit Features
- Upload CSV dataset  
- Model selection dropdown  
- Display evaluation metrics  
- Confusion matrix visualization  
- Classification report  

---

## Conclusion

Ensemble models (Random Forest and XGBoost) outperformed individual models.  
Among all models, **XGBoost** demonstrated the best balance between precision, recall, and robustness (highest MCC), making it the most suitable model for this dataset.
