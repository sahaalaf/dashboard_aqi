import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os

# Set page configuration
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")

# Title
st.title("Air Quality Analysis Dashboard")

# Load data and add AQI Category
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('air_quality_data.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        data.dropna(inplace=True)
        # Add AQI Category column
        data['AQI Category'] = data['Value'].apply(
            lambda x: 'Good' if x <= 50 else
                      'Moderate' if x <= 100 else
                      'Unhealthy for Sensitive Groups' if x <= 150 else
                      'Unhealthy' if x <= 200 else
                      'Very Unhealthy' if x <= 300 else 'Hazardous'
        )
        return data
    except FileNotFoundError:
        st.error("Error: 'air_quality_data.csv' not found. Please ensure the file is in the same directory as this script.")
        return None

data = load_data()

if data is not None:
    # Sidebar for navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Select a page:", ["Overview", "Statistics", "Visualizations", "Predictions", "Export"])

    # Overview Page
    if page == "Overview":
        st.header("Dataset Overview")
        st.write("This dashboard analyzes air quality data, including PM2.5 (AQI) levels, and provides statistical insights, visualizations, and predictive models.")
        st.subheader("Sample Data")
        st.dataframe(data.head())
        st.write(f"Total Records: {len(data)}")
        st.write(f"Columns: {', '.join(data.columns)}")

    # Statistics Page
    elif page == "Statistics":
        st.header("Statistical Analysis")
        st.subheader("Missing Values")
        st.write(data.isnull().sum())

        st.subheader("AQI Basic Statistics")
        stats = data['Value'].describe()
        st.write(stats)
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean AQI", f"{data['Value'].mean():.2f}")
        col2.metric("Median AQI", f"{data['Value'].median():.2f}")
        col3.metric("Standard Deviation", f"{data['Value'].std():.2f}")

    # Visualizations Page
    elif page == "Visualizations":
        st.header("Data Visualizations")

        # Line Plot
        st.subheader("AQI Levels Over Time")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=data, x='Date', y='Value', marker='o', ax=ax)
        ax.set_title('PM2.5 (AQI) Levels Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('AQI')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Bar Plot
        st.subheader("AQI Levels by Date")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=data, x='Date', y='Value', ax=ax)
        ax.set_title('PM2.5 (AQI) Levels')
        ax.set_xlabel('Date')
        ax.set_ylabel('AQI')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # AQI Category Distribution
        st.subheader("AQI Category Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(data=data, x='AQI Category', order=data['AQI Category'].value_counts().index, ax=ax)
        ax.set_title('AQI Category Distribution')
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig)

    # Predictions Page
    elif page == "Predictions":
        st.header("Predictive Modeling")

        # Simulated Respiratory Health Data
        data['Respiratory Health'] = data['Value'] * 0.1 + np.random.normal(0, 1, size=len(data))

        # Linear Regression for Respiratory Health
        st.subheader("Respiratory Health Prediction (Linear Regression)")
        X = data[['Value']]
        y = data['Respiratory Health']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

        # Scatter Plot: Actual vs Predicted
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y_test, y_pred, color='teal')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel('Actual Respiratory Health')
        ax.set_ylabel('Predicted Respiratory Health')
        ax.set_title('Actual vs Predicted Respiratory Health')
        plt.tight_layout()
        st.pyplot(fig)

        # Correlation
        correlation = data[['Value', 'Respiratory Health']].corr().iloc[0, 1]
        st.write(f"Pearson Correlation (AQI vs Respiratory Health): {correlation:.3f}")

        # Classification for AQI Categories
        st.subheader("AQI Category Prediction (Random Forest)")
        Xc = data[['Value']]
        yc = data['AQI Category']
        label_encoder = LabelEncoder()
        yc_encoded = label_encoder.fit_transform(yc)
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc_encoded, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(random_state=42)
        clf.fit(Xc_train, yc_train)
        yc_pred = clf.predict(Xc_test)
        accuracy = accuracy_score(yc_test, yc_pred)
        st.write(f"Classification Accuracy: {accuracy * 100:.2f}%")

        # Classification Report
        unique_classes = np.unique(yc_test)
        unique_class_names = label_encoder.inverse_transform(unique_classes)
        st.write("Classification Report:")
        report = classification_report(yc_test, yc_pred, labels=unique_classes, target_names=unique_class_names, output_dict=True)
        st.table(pd.DataFrame(report).transpose())

        # Confusion Matrix
        cm = confusion_matrix(yc_test, yc_pred, labels=unique_classes)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_class_names, yticklabels=unique_class_names, ax=ax)
        ax.set_title('Confusion Matrix for AQI Category Prediction')
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        plt.tight_layout()
        st.pyplot(fig)

    # Export Page
    elif page == "Export":
        st.header("Export Data")
        output_dir = 'dashboard_output'
        output_path = os.path.join(output_dir, 'processed_air_quality_data.csv')
        if st.button("Export Processed Data"):
            try:
                os.makedirs(output_dir, exist_ok=True)
                data.to_csv(output_path, index=False)
                st.success(f"File saved successfully to {output_path}")
            except PermissionError:
                st.error(f"Permission denied: Unable to write to {output_path}. Try running as administrator or using a different directory.")
            except Exception as e:
                st.error(f"An error occurred while saving the file: {e}")

# Footer
st.sidebar.markdown("---")
