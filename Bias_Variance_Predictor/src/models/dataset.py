import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

class BreastCancerDataset:
    @staticmethod
    @st.cache_data(show_spinner="Loading Breast Cancer dataset...")
    def get_data():
        data = load_breast_cancer()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test
