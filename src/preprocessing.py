
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

base_url= "https://github.com/KadidjaGUEBEDIANG/project-machine-learning-student-performance/raw/main/StudentPerformanceFactors.xlsx"#les données sont disponibles sur github dans le compte kadidjaGUEBEDIANG 
base= pd.read_excel(base_url, engine="openpyxl")#la variable base contient notre base de donnée
X = base.drop(columns=['Exam_Score'])  
y = base.Exam_Score

# Définition des colonnes selon le type de transformation souhaitée
ordinal_cols = {
    'Parental_Involvement': ['Low', 'Medium', 'High'],
    'Access_to_Resources': ['Low', 'Medium', 'High'],
    'Motivation_Level': ['Low', 'Medium', 'High'],
    'Family_Income': ['Low', 'Medium', 'High'],
    'Teacher_Quality': ['Low', 'Medium', 'High'],
    'Parental_Education_Level': ['High School', 'College', 'Postgraduate'],
    'Distance_from_Home': ['Near', 'Moderate', 'Far']
}

nominal_cols = [
    'Extracurricular_Activities',   # Yes / No
    'Internet_Access',              # Yes / No
    'Peer_Influence',               # Positive / Negative / Neutral
    'Learning_Disabilities',        # Yes / No
]
colonnes_standard = ['Hours_Studied', 'Sleep_Hours', 'Physical_Activity']
colonnes_minmax = ['Attendance', 'Previous_Scores', 'Tutoring_Sessions']

# Imputateurs
imputation_num = SimpleImputer(strategy='median')
imputation_cat = SimpleImputer(strategy='most_frequent')

# Pipelines numériques
pipeline_standard = Pipeline([
    ('imputation', imputation_num),
    ('standardisation', StandardScaler())
])

pipeline_minmax = Pipeline([
    ('imputation', imputation_num),
    ('minmax', MinMaxScaler())
])

# Pipelines catégorielles
pipeline_ord = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(
        categories=[ordinal_cols[col] for col in ordinal_cols],
        handle_unknown='use_encoded_value',
        unknown_value=-1,
        encoded_missing_value=-2
    ))
])

pipeline_nom = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(
        handle_unknown='ignore',
        drop='if_binary',
        sparse_output=False
    ))
])

# Combinaison des transformations
preprocessor = ColumnTransformer(transformers=[
    ('standard', pipeline_standard, colonnes_standard),
    ('minmax', pipeline_minmax, colonnes_minmax),
    ('ord', pipeline_ord, list(ordinal_cols.keys())),
    ('nom', pipeline_nom, nominal_cols)
])
preprocessor

