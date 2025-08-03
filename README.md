# Heart Disease Detection

Heart Disease Detection is a professional-grade machine learning web application designed to predict the likelihood of heart disease in patients based on clinical parameters. The project leverages a robust K-Nearest Neighbors (KNN) pipeline and provides a user-friendly web interface built with Flask, enabling healthcare professionals and researchers to input patient data and receive instant, data-driven predictions.

---

## Table of Contents
- [Features](#features)
- [Training Data](#training-data)
- [Model Training Process](#model-training-process)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features
- **Intuitive Web Interface:** Streamlined data entry for clinical parameters.
- **Accurate Machine Learning Predictions:** Utilizes a validated KNN model for reliable results.
- **Comprehensive Preprocessing:** Handles both categorical and numerical features with custom encoding and scaling.
- **Immediate Feedback:** Delivers real-time risk assessment for heart disease.

---

## Training Data
The model is trained on the `heart.csv` dataset, which contains anonymized clinical records of patients. The dataset includes the following features:

- **Age**
- **Sex**
- **ChestPainType**
- **RestingBP** (Resting Blood Pressure)
- **Cholesterol**
- **FastingBS** (Fasting Blood Sugar)
- **RestingECG**
- **MaxHR** (Maximum Heart Rate)
- **ExerciseAngina**
- **Oldpeak**
- **ST_Slope**
- **HeartDisease** (Target: 1 = disease present, 0 = not present)

The dataset comprises both numerical and categorical variables, necessitating advanced preprocessing techniques for optimal model performance.

---

## Model Training Process

1. **Preprocessing:**
   - Categorical features are label-encoded using a custom `MultiColumnLabelEncoder` transformer.
   - All features are standardized using `StandardScaler` to ensure uniformity.

2. **Model Selection:**
   - The classification model is a `KNeighborsClassifier` (KNN), selected for its effectiveness in medical datasets.
   - The entire workflow is encapsulated in a scikit-learn `Pipeline` for reproducibility and maintainability.

3. **Training Procedure:**
   - The dataset is split into training and test sets (80/20 split, stratified by the target variable).
   - The pipeline is trained on the training set and evaluated on the test set.

4. **Evaluation Metrics:**
   - Model performance is assessed using accuracy, classification report, and confusion matrix.

**Sample Training Code:**
```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

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

pipeline = Pipeline([
    ('label_encoding', MultiColumnLabelEncoder()),
    ('scaling', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
pipeline.fit(X_train, y_train)
```

**Evaluation Example:**
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

---

## Model Details
- **Algorithm:** K-Nearest Neighbors (KNN)
- **Preprocessing:** Multi-column label encoding and feature scaling
- **Input Features:** Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting BS, Resting ECG, Max HR, Exercise Angina, Oldpeak, ST Slope

---

## Project Structure
```
heart_disease_detection/
├── .gitignore
├── heart.csv                        # Training data
├── heart_disease_knn_pipeline.pkl   # Trained ML pipeline
├── LICENSE
├── main.py                          # Flask application
├── README.md
└── templates/
    └── index.html                   # Web interface template
```

---

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
1. **Clone the repository:**
   ```zsh
   git clone <repository-url>
   cd heart_disease_detection
   ```
2. **Install dependencies:**
   ```zsh
   pip install flask pandas scikit-learn joblib
   ```
3. **Ensure the model file is present:**
   - `heart_disease_knn_pipeline.pkl` should be in the project root.

### Running the Application
```zsh
python main.py
```
- Open your browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Usage
1. Fill in the required patient information on the homepage.
2. Click "Predict" to see the result.
3. The application will display whether heart disease is detected or not.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements
- [Flask](https://flask.palletsprojects.com/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)

---

*Developed by Rayan Haroon*
