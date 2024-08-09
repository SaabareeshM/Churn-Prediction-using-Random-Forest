from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pre-trained model and feature names
model = joblib.load('Churn_Predict_model.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        p1 = int(request.form['CreditScore'])
        p2 = int(request.form['Gender'])
        p3 = int(request.form['Age'])
        p4 = int(request.form['Tenure'])
        p5 = float(request.form['Balance'])
        p6 = int(request.form['NumOfProducts'])
        p7 = int(request.form['HasCrCard'])
        p8 = int(request.form['IsActiveMember'])
        p9 = int(request.form['Geography'])
        p10 = float(request.form['EstimatedSalary'])
        
        # Geography encoding
        Geography_Germany = 0
        Geography_Spain = 0
        Geography_France = 0
        if p9 == 1:
            Geography_Germany = 1
        elif p9 == 2:
            Geography_Spain = 1
        elif p9 == 3:
            Geography_France = 1

        # Gender encoding
        Gender_Male = p2

        # Create DataFrame with input values
        input_data = pd.DataFrame({
            'CreditScore': [p1],
            'Gender_Male': [Gender_Male],
            'Age': [p3],
            'Tenure': [p4],
            'Balance': [p5],
            'NumOfProducts': [p6],
            'HasCrCard': [p7],
            'IsActiveMember': [p8],
            'Geography_Germany': [Geography_Germany],
            'Geography_Spain': [Geography_Spain],
            'Geography_France': [Geography_France],
            'EstimatedSalary': [p10]
        })

        # Reorder the DataFrame columns to match training phase
        input_data = input_data.reindex(columns=model_columns, fill_value=0)

        # Predict the outcome
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            result = 'The customer will churn.'
        else:
            result = 'The customer will not churn.'

        return render_template('index.html', prediction_text=result)
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
