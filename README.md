---

# ChurnPrediction Using Random Forest

This project aims to predict customer churn for a bank using machine learning techniques. The model has been trained to classify whether a customer will churn (leave the bank) or not, based on various features like credit score, age, gender, balance, and more.

## Overview

Customer churn prediction is a critical task for businesses as it helps in identifying customers who are likely to leave the service, enabling proactive retention strategies. This project utilizes the Random Forest algorithm, a versatile and widely-used ensemble learning method, to build a predictive model for customer churn.

## Table of Contents

- Installation
- Usage
- Comparison
- Model Training
- Algorithm Explanation
- Results

## Installation

To run this project, you need to install the required packages listed in `requirements.txt`. Use the following command to install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository.
2. Ensure you have the dataset `Churn_Modelling.csv` in the project directory.
3. Run the `bank_churn_prediction.py` script to train the model.
4. Use `app.py` to serve the model and make predictions via a web interface.

## Comparison

In the `Bank_churn_prediction.ipynb` notebook, I compared the performance of five different machine learning algorithms for predicting customer churn. The algorithms under comparison are:

- Logistic Regression
- Random Forest
- DecisionTreeClassifier
- K-Nearest Neighbors (KNN)
- GradientBoostingClassifier

## Model Training

The model training process is outlined in the `bank_churn_prediction.py` script. Below are the main steps involved:

- **Data Loading**: Load the customer data from the CSV file.
- **Data Preprocessing**: Encode categorical variables, drop unnecessary columns, and split the data into features and target variable.
- **Data Balancing**: Use SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance in the training data.
- **Model Training**: Train a Random Forest classifier on the balanced training data.
- **Model Evaluation**: Evaluate the model on both training and test datasets to check for performance metrics.

## Algorithm Explanation

### Random Forest

Random Forest is an ensemble learning method that operates by constructing a multitude of decision trees during training and outputting the class that is the mode of the classes (classification) of the individual trees. Here's a brief explanation of how it works:

- **Decision Trees**: Random Forest builds multiple decision trees during the training phase. Each tree is trained on a different subset of the data, which helps in reducing variance.
- **Bootstrap Aggregation (Bagging)**: Random Forest uses a technique called bagging, where multiple models (decision trees) are trained on different subsets of the data, and their predictions are averaged.
- **Feature Randomness**: During the construction of each tree, a random subset of features is chosen to split the data at each node. This reduces the correlation between trees, making the model more robust.
- **Final Prediction**: The final prediction is made based on the majority vote from all the individual trees, which improves accuracy and generalization.

## Results

The model's performance is evaluated using classification metrics such as precision, recall, F1-score, and accuracy. Below are the results for the test dataset:

![Model Performance](https://github.com/user-attachments/assets/d71d7dc0-1b01-4e5c-8ea0-f35363fed0e2)

## Final Output

The final output, including model predictions, can be viewed in the accompanying video:

[Final Output Video](https://github.com/user-attachments/assets/2e014dfb-e739-4ed0-bfdc-dbe52ae93c6c)

---
