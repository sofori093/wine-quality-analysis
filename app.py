import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Set page config
st.set_page_config(page_title="White Wine Quality Dashboard", layout="wide")

# Debugging
st.write("🔄 Dashboard initializing...")

# Title and Description
st.title("🍷 White Wine Quality Analysis & Prediction")
st.markdown("""
This dashboard explores the physicochemical properties of white wine and predicts its quality based on a Random Forest model.
Based on the analysis of the `whitewine.csv` dataset.
""")

# Load and Process Data
@st.cache_data
def get_processed_data():
    df = pd.read_csv('whitewine.csv')
    X = df.drop('quality', axis=1)
    y = df['quality']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X, y, X_scaled, scaler

df, X, y, X_scaled, scaler = get_processed_data()

# Sidebar - Tabs
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dataset Overview", "Exploratory Data Analysis", "Quality Prediction"])

if page == "Dataset Overview":
    st.header("Dataset Overview")
    st.write("First 5 rows of the dataset:")
    st.dataframe(df.head())
    
    st.write("Statistical Summary:")
    st.dataframe(df.describe())
    
    st.write(f"Total entries: {df.shape[0]}, Total features: {df.shape[1]}")

elif page == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
        st.caption("Notice the high correlation between density and residual sugar.")

    with col2:
        st.subheader("Alcohol vs Quality")
        fig, ax = plt.subplots()
        sns.boxplot(x='quality', y='alcohol', data=df, ax=ax, palette="Blues")
        ax.set_title("Alcohol Level by Quality Score")
        st.pyplot(fig)
        st.caption("Higher quality wines generally have higher alcohol content.")

    st.subheader("Feature Distributions")
    feature_to_plot = st.selectbox("Select feature to view distribution:", df.columns[:-1])
    fig, ax = plt.subplots()
    sns.histplot(df[feature_to_plot], kde=True, ax=ax, color='green')
    st.pyplot(fig)

elif page == "Quality Prediction":
    st.header("Predict Wine Quality")
    
    @st.cache_resource
    def train_model(X_scaled_local, y_local):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled_local, y_local, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        return model, r2
    
    with st.spinner("Training model..."):
        model, r2 = train_model(X_scaled, y)
    
    st.info(f"Model Performance (Random Forest): R² = {r2:.4f}")
    
    st.write("Adjust the parameters below to predict the quality of a wine:")
    
    # Inputs
    cols = st.columns(3)
    inputs = {}
    for i, col_name in enumerate(X.columns):
        with cols[i % 3]:
            # Use columns from X to ensure order
            inputs[col_name] = st.slider(col_name, 
                                          float(df[col_name].min()), 
                                          float(df[col_name].max()), 
                                          float(df[col_name].mean()))
    
    # Prediction
    # Ensure inputs are in the same order as X.columns
    input_values = [inputs[col] for col in X.columns]
    features = np.array([input_values])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    
    st.divider()
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric("Predicted Quality Score", f"{prediction[0]:.2f}")
    with res_col2:
        # Interpret prediction
        if prediction[0] >= 7:
            st.success("High Quality Wine! 🥂")
        elif prediction[0] >= 5:
            st.warning("Average Quality Wine.")
        else:
            st.error("Low Quality Wine.")

    # Feature Importance Plot
    st.subheader("Feature Importance (Model Insight)")
    importances = model.feature_importances_
    feat_importances = pd.Series(importances, index=X.columns).sort_values()
    fig, ax = plt.subplots()
    feat_importances.plot(kind='barh', ax=ax, color='purple')
    st.pyplot(fig)
