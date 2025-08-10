from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Load the trained pipeline with error handling
try:
    model = joblib.load("heart_disease_knn_pipeline.pkl")
    logger.info("Model loaded successfully")
except FileNotFoundError:
    logger.error("Model file not found. Please ensure heart_disease_knn_pipeline.pkl exists.")
    model = None
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Input validation ranges and options
INPUT_CONSTRAINTS = {
    'Age': {'min': 1, 'max': 120},
    'RestingBP': {'min': 50, 'max': 250},
    'Cholesterol': {'min': 0, 'max': 1000},
    'MaxHR': {'min': 50, 'max': 220},
    'Oldpeak': {'min': -5.0, 'max': 10.0},
    'Sex': ['M', 'F'],
    'ChestPainType': ['ATA', 'NAP', 'ASY', 'TA'],
    'FastingBS': [0, 1],
    'RestingECG': ['Normal', 'ST', 'LVH'],
    'ExerciseAngina': ['Y', 'N'],
    'ST_Slope': ['Up', 'Flat', 'Down']
}

def validate_input(form_data):
    """Validate input data against constraints"""
    errors = []
    
    # Numeric validations
    for field, constraints in INPUT_CONSTRAINTS.items():
        if field in ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']:
            value = form_data.get(field)
            if value is not None:
                try:
                    value = int(value)
                    if value < constraints['min'] or value > constraints['max']:
                        errors.append(f"{field} must be between {constraints['min']} and {constraints['max']}")
                except ValueError:
                    errors.append(f"{field} must be a valid number")
    
    # Oldpeak validation
    if 'Oldpeak' in form_data:
        try:
            value = float(form_data['Oldpeak'])
            if value < INPUT_CONSTRAINTS['Oldpeak']['min'] or value > INPUT_CONSTRAINTS['Oldpeak']['max']:
                errors.append(f"Oldpeak must be between {INPUT_CONSTRAINTS['Oldpeak']['min']} and {INPUT_CONSTRAINTS['Oldpeak']['max']}")
        except ValueError:
            errors.append("Oldpeak must be a valid number")
    
    # Categorical validations
    for field in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
        if form_data.get(field) not in INPUT_CONSTRAINTS[field]:
            errors.append(f"Invalid value for {field}")
    
    # FastingBS validation
    try:
        fasting_bs = int(form_data.get('FastingBS', 0))
        if fasting_bs not in INPUT_CONSTRAINTS['FastingBS']:
            errors.append("Invalid value for Fasting Blood Sugar")
    except ValueError:
        errors.append("Fasting Blood Sugar must be 0 or 1")
    
    return errors

@app.route('/')
def home():
    return render_template('index.html', constraints=INPUT_CONSTRAINTS)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'error': 'Model not available. Please contact administrator.',
            'success': False
        }), 500
    
    try:
        # Extract data from form
        form_data = {
            'Age': request.form.get('Age'),
            'Sex': request.form.get('Sex'),
            'ChestPainType': request.form.get('ChestPainType'),
            'RestingBP': request.form.get('RestingBP'),
            'Cholesterol': request.form.get('Cholesterol'),
            'FastingBS': request.form.get('FastingBS'),
            'RestingECG': request.form.get('RestingECG'),
            'MaxHR': request.form.get('MaxHR'),
            'ExerciseAngina': request.form.get('ExerciseAngina'),
            'Oldpeak': request.form.get('Oldpeak'),
            'ST_Slope': request.form.get('ST_Slope')
        }
        
        # Validate input
        validation_errors = validate_input(form_data)
        if validation_errors:
            return jsonify({
                'error': 'Validation errors: ' + '; '.join(validation_errors),
                'success': False
            }), 400
        
        # Convert to proper types
        processed_data = {
            'Age': int(form_data['Age']),
            'Sex': form_data['Sex'],
            'ChestPainType': form_data['ChestPainType'],
            'RestingBP': int(form_data['RestingBP']),
            'Cholesterol': int(form_data['Cholesterol']),
            'FastingBS': int(form_data['FastingBS']),
            'RestingECG': form_data['RestingECG'],
            'MaxHR': int(form_data['MaxHR']),
            'ExerciseAngina': form_data['ExerciseAngina'],
            'Oldpeak': float(form_data['Oldpeak']),
            'ST_Slope': form_data['ST_Slope']
        }

        # Convert to DataFrame and predict
        input_df = pd.DataFrame([processed_data])
        prediction = model.predict(input_df)[0]
        
        # Get prediction probability if available
        try:
            prediction_proba = model.predict_proba(input_df)[0]
            confidence = max(prediction_proba) * 100
        except AttributeError:
            confidence = None

        result_text = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"
        risk_level = "High Risk" if prediction == 1 else "Low Risk"
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'result': result_text,
            'risk_level': risk_level,
            'confidence': round(confidence, 1) if confidence else None
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': f'An error occurred during prediction: {str(e)}',
            'success': False
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)