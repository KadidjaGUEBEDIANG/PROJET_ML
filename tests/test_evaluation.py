# test/test_evaluation.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from evaluation import evaluate_model, print_evaluation
from preprocessing import get_preprocessor  
from sklearn.pipeline import Pipeline


# Chargement des données
base_url = "https://github.com/KadidjaGUEBEDIANG/project-machine-learning-student-performance/raw/main/StudentPerformanceFactors.xlsx"
base = pd.read_excel(base_url, engine="openpyxl")

X = base.drop(columns=['Exam_Score'])
y = base['Exam_Score']

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Construction du pipeline avec preprocessing + modèle
preprocessor = get_preprocessor()
model = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', LinearRegression())
])

# Entraînement
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluation
results = evaluate_model(y_test, y_pred)
print_evaluation(results)

