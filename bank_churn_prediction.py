# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib as jb
from imblearn.over_sampling import SMOTE

# Load the data
data = pd.read_csv('Churn_Modelling.csv')

# Drop irrelevant columns
data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# One-hot encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Separate features and target variable
x = data.drop(['Exited'], axis=1)
y = data['Exited']

# Resample the data to handle class imbalance
x_res, y_res = SMOTE().fit_resample(x, y)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.2, random_state=42)

# Train the RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Make predictions
y_pred = rf.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Save the model and feature names
jb.dump(rf, 'Churn_Predict_model.pkl')
jb.dump(list(x_train.columns), 'model_columns.pkl')
