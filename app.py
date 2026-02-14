import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model.models import (
    preprocess_data,
    train_logistic_regression,
    train_decision_tree,
    train_knn,
    train_bayes,
    train_randomforest,
    train_xgboost,
    evaluate_model
)

st.set_page_config(page_title="ML Model Comparison", layout="wide")

st.title("Machine Learning Classification Models")

# Upload & Download
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload your dataset CSV", type="csv")

with col2:
    st.write("Download Sample Dataset")
    try:
        with open("bank.csv", "rb") as file:
            st.download_button(
                label="Download bank.csv",
                data=file,
                file_name="bank.csv",
                mime="text/csv"
            )
    except FileNotFoundError:
        st.warning("bank.csv not found in repository.")

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=';')
    st.success("Dataset loaded successfully!")
    st.write("Preview:", df.head())

    X = df.drop('y', axis=1)
    y = df['y']

    model_name = st.selectbox(
        "Select a Model to Train and Evaluate",
        [
            "Logistic Regression",
            "Decision Tree",
            "K-Nearest Neighbors",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        ]
    )

    if st.button("Run Model"):

        st.info(f"Training and evaluating {model_name}...")

        X_train, X_test, y_train, y_test = preprocess_data(X, y)

        model_funcs = {
            "Logistic Regression": train_logistic_regression,
            "Decision Tree": train_decision_tree,
            "K-Nearest Neighbors": train_knn,
            "Naive Bayes": train_bayes,
            "Random Forest": train_randomforest,
            "XGBoost": train_xgboost
        }

        model = model_funcs[model_name](X_train, y_train)

        metrics, cm, report = evaluate_model(model, X_test, y_test)

        st.success(f"{model_name} completed!")
        st.subheader("Evaluation Metrics")
        st.table(metrics)

        st.subheader("Confusion Matrix")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)