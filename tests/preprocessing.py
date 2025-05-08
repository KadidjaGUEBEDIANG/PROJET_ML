from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer


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
