<<<<<<< HEAD
=======

>>>>>>> 91ae6a4b28936398d5b2746d6389654b6819a4ea
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

<<<<<<< HEAD
=======
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
>>>>>>> 91ae6a4b28936398d5b2746d6389654b6819a4ea

def get_preprocessor():
    # Colonnes ordinales avec ordre explicite
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
        'Extracurricular_Activities',
        'Internet_Access',
        'Peer_Influence',
        'Learning_Disabilities'
    ]
    
    colonnes_standard = ['Hours_Studied', 'Sleep_Hours', 'Physical_Activity']
    colonnes_minmax = ['Attendance', 'Previous_Scores', 'Tutoring_Sessions']

    # Pipelines numériques
    pipeline_standard = Pipeline([
        ('imputation', SimpleImputer(strategy='median')),
        ('standardisation', StandardScaler())
    ])

    pipeline_minmax = Pipeline([
        ('imputation', SimpleImputer(strategy='median')),
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

    # Assemble tous les pipelines
    preprocessor = ColumnTransformer(transformers=[
        ('standard', pipeline_standard, colonnes_standard),
        ('minmax', pipeline_minmax, colonnes_minmax),
        ('ord', pipeline_ord, list(ordinal_cols.keys())),
        ('nom', pipeline_nom, nominal_cols)
    ])

    return preprocessor
