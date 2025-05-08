# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:04:26 2025

@author: GANK
"""

import numpy as np
from sklearn import metrics

def evaluate_model(y_true, y_pred):
    """
    Calcule les principaux indicateurs d’évaluation d’un modèle de régression.

    Paramètres:
    -----------
    y_true : array-like
        Valeurs réelles de la variable cible.
    y_pred : array-like
        Prédictions du modèle.

    Retourne:
    ---------
    dict
        Dictionnaire contenant MAE, MAPE, MSE, RMSE, R2.
    """
    return {
        'MAE': metrics.mean_absolute_error(y_true, y_pred),
        'MAPE': metrics.mean_absolute_percentage_error(y_true, y_pred) * 100,
        'MSE': metrics.mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(metrics.mean_squared_error(y_true, y_pred)),
        'R2': metrics.r2_score(y_true, y_pred) * 100
    }

def print_evaluation(results):
    """
    Affiche les résultats d’évaluation de manière lisible.

    Paramètres:
    -----------
    results : dict
        Dictionnaire contenant les métriques calculées.
    """
    print("\n=== Performance du modèle ===")
    print(f"- MAE  : {results['MAE']:.2f}")
    print(f"- MAPE : {results['MAPE']:.2f}%")
    print(f"- MSE  : {results['MSE']:.2f}")
    print(f"- RMSE : {results['RMSE']:.2f}")
    print(f"- R²   : {results['R2']:.2f}%")
