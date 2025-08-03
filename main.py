from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)
class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        for col in X.select_dtypes(include='object').columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.encoders[col] = le
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, le in self.encoders.items():
            X_copy[col] = le.transform(X_copy[col])
        return X_copy

# Load the trained pipeline
model = joblib.load("heart_disease_knn_pipeline.pkl")
from sklearn.base import BaseEstimator, TransformerMixin

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    form_data = {
        'Age': int(request.form['Age']),
        'Sex': request.form['Sex'],
        'ChestPainType': request.form['ChestPainType'],
        'RestingBP': int(request.form['RestingBP']),
        'Cholesterol': int(request.form['Cholesterol']),
        'FastingBS': int(request.form['FastingBS']),
        'RestingECG': request.form['RestingECG'],
        'MaxHR': int(request.form['MaxHR']),
        'ExerciseAngina': request.form['ExerciseAngina'],
        'Oldpeak': float(request.form['Oldpeak']),
        'ST_Slope': request.form['ST_Slope']
    }

    # Convert to DataFrame and predict
    input_df = pd.DataFrame([form_data])
    prediction = model.predict(input_df)[0]

    result = "Positive (Heart Disease Detected)" if prediction == 1 else "Negative (No Heart Disease)"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
