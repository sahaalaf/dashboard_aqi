# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Step 1: Load Data
data = pd.read_csv('air_quality_data.csv')

# Step 2: Parse Date
data['Date'] = pd.to_datetime(data['Date'])

# Step 3: Handle Missing Values
print("Missing values in each column:")
print(data.isnull().sum())
data.dropna(inplace=True)

# Step 4: Basic Statistics
print("\nBasic statistics on AQI values:")
print(data['Value'].describe())
mean_aqi = data['Value'].mean()
median_aqi = data['Value'].median()
std_aqi = data['Value'].std()

print(f'\nMean AQI: {mean_aqi}')
print(f'Median AQI: {median_aqi}')
print(f'Standard Deviation: {std_aqi}')

# Step 5: Data Visualization - Line Plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='Date', y='Value', marker='o')
plt.title('PM2.5 (AQI) Levels Over Time')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 6: Data Visualization - Bar Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=data, x='Date', y='Value')
plt.title('PM2.5 (AQI) Levels')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 7: Simulated Respiratory Health Data (with noise)
np.random.seed(42)
data['Respiratory Health'] = data['Value'] * 0.1 + np.random.normal(0, 1, size=len(data))

# Step 8: AQI Category Classification
def categorize_aqi(aqi):
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Moderate'
    elif aqi <= 150:
        return 'Unhealthy for Sensitive Groups'
    elif aqi <= 200:
        return 'Unhealthy'
    elif aqi <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'

data['AQI Category'] = data['Value'].apply(categorize_aqi)

# Optional: Visualize AQI Categories
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='AQI Category', order=data['AQI Category'].value_counts().index)
plt.title('AQI Category Distribution')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Step 9: Linear Regression Model for Respiratory Health (Regression Task)
X = data[['Value']]
y = data['Respiratory Health']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 10: Model Evaluation (RMSE for Regression)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'\nRoot Mean Squared Error (RMSE) for Respiratory Health Prediction: {rmse}')

# Step 11: Correlation Analysis for Regression
correlation = data[['Value', 'Respiratory Health']].corr(method='pearson')
print("\nPearson Correlation between AQI and Respiratory Health:")
print(correlation)

# Step 12: Scatter plot - Actual vs Predicted (for Regression)
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Respiratory Health')
plt.ylabel('Predicted Respiratory Health')
plt.title('Actual vs Predicted Respiratory Health')
plt.tight_layout()
plt.show()

# Step 13: Classification - Predict AQI Categories with RandomForestClassifier
# Prepare data for classification task
Xc = data[['Value']]
yc = data['AQI Category']

# Encode categorical labels (AQI Categories) into numerical values
label_encoder = LabelEncoder()
yc_encoded = label_encoder.fit_transform(yc)

# Split data for training and testing classification model
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc_encoded, test_size=0.2, random_state=42)

# Train RandomForestClassifier model
clf = RandomForestClassifier(random_state=42)
clf.fit(Xc_train, yc_train)
yc_pred = clf.predict(Xc_test)

# Step 14: Classification Evaluation
accuracy = accuracy_score(yc_test, yc_pred)
print(f"\n🎯 Classification Accuracy (AQI Category): {round(accuracy * 100, 2)}%")

# Get the unique classes in yc_test to avoid mismatch
unique_classes = np.unique(yc_test)
unique_class_names = label_encoder.inverse_transform(unique_classes)

print("\nClassification Report:")
print(classification_report(yc_test, yc_pred, labels=unique_classes, target_names=unique_class_names))

# Step 15: Confusion Matrix for Classification
cm = confusion_matrix(yc_test, yc_pred, labels=unique_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_class_names, yticklabels=unique_class_names)
plt.title('Confusion Matrix for AQI Category Prediction')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.show()
# Step 16: Export for Power BI
data.to_csv('processed_air_quality_data.csv', index=False)
