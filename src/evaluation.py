from sklearn import metrics

def evaluate_optimized_model(model_name, grid_search, X_test, y_test):
    """
    Évalue un modèle optimisé et affiche les métriques principales
    
    Args:
        model_name (str): Nom du modèle
        grid_search: Objet GridSearchCV après fitting
        X_test: Features de test
        y_test: Cible de test
    
    Returns:
        dict: Dictionnaire des métriques calculées
    """
    # Prédictions avec le meilleur estimateur
    y_pred = grid_search.best_estimator_.predict(X_test)
    
    # Calcul des métriques
    model_metrics = {
        'MAE': metrics.mean_absolute_error(y_test, y_pred),
        'MAPE': metrics.mean_absolute_percentage_error(y_test, y_pred) * 100,
        'MSE': metrics.mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
        'R²': metrics.r2_score(y_test, y_pred)*100,
        'Max Error': metrics.max_error(y_test, y_pred),
        'Explained Variance': metrics.explained_variance_score(y_test, y_pred)
    }
    
    # Affichage des résultats
    print(f"\n=== Performance du {model_name} optimisé ===")
    print(f"Meilleurs paramètres: {grid_search.best_params_}")
    
    for metric, value in model_metrics.items():
        print(f"- {metric}: {value:.4f}")
    
    return model_metrics


# Évaluation des modèles optimisés
lr_metrics = evaluate_optimized_model("Régression Linéaire", lr_grid_search, X_test, y_test)
knn_metrics = evaluate_optimized_model("KNN", knn_grid_search, X_test, y_test)
dt_metrics = evaluate_optimized_model("Decision Tree", tree_grid_search, X_test, y_test)
gb_metrics = evaluate_optimized_model("Gradient Boosting", gb_grid_search, X_test, y_test)
xgb_metrics = evaluate_optimized_model("XGBoost", xgb_grid_search, X_test, y_test)
lgbm_metrics = evaluate_optimized_model("LightGBM", lgbm_grid_search, X_test, y_test)
rf_metrics = evaluate_optimized_model("Random Forest", rf_grid_search, X_test, y_test)



# Comparaison globale des modèles
print("\n=== Comparaison des performances ===")
all_metrics = {
    'Régression Linéaire': lr_metrics,
    'KNN': knn_metrics,
    'Decision Tree': dt_metrics,
    'Gradient Boosting': gb_metrics,
    'XGBoost': xgb_metrics,
    'LightGBM': lgbm_metrics,
    'Random Forest':rf_metrics
}

# Création d'un DataFrame pour visualisation
metrics_df = pd.DataFrame(all_metrics).T
print("\nRésumé des métriques (trié par R2):")
print(metrics_df.sort_values('R²'))

# Visualisation graphique
plt.figure(figsize=(12, 6))
metrics_df['R²'].sort_values().plot(kind='barh', title='Comparaison du R2 entre modèles')
plt.xlabel('R² (plus grand = meilleur)')
plt.show()
