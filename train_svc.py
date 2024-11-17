import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from src.data.data_loader import load_data_preprocessed
from src.utils.config import OUTPUT_DIR, SVC_PARAM_GRID, N_SPLITS, BASE_DIR
from src.visualization.plots import plot_metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def main():
    # Charger les données prétraitées
    X, y, scaler, label_encoder = load_data_preprocessed()

    # Définir le modèle SVM
    svc = SVC(probability=False, random_state=42)

    # Définir la grille d'hyperparamètres pour GridSearchCV
    param_grid = SVC_PARAM_GRID

    # Initialiser GridSearchCV avec return_train_score=True
    grid_search = GridSearchCV(
        estimator=svc,
        param_grid=param_grid,
        cv=N_SPLITS,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    # Exécuter la recherche d'hyperparamètres avec validation croisée
    grid_search.fit(X, y)

    # Extraire les meilleurs hyperparamètres et le meilleur modèle
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_

    print(f"Meilleurs hyperparamètres: {best_params}")
    print(f"Meilleur score de validation croisée: {best_score:.4f}")

    # Sauvegarder le meilleur modèle et les objets de prétraitement
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    model_path = os.path.join(BASE_DIR, 'best_svc_model.pkl')
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, 'label_encoder.pkl'))

    print("Entraînement terminé et meilleur modèle SVM sauvegardé.")

    # Extraire les métriques d'accuracy
    results = pd.DataFrame(grid_search.cv_results_)

    # Créer un DataFrame similaire à celui du MLP
    processed_results = pd.DataFrame({
        'model_params': results['params'],
        'train_params': [{} for _ in range(len(results))],  # SVC n'a pas de paramètres d'entraînement spécifiques
        'average_train_accuracy': results['mean_train_score'],
        'average_val_accuracy': results['mean_test_score'],
        'average_train_loss': [None] * len(results),  # SVC ne calcule pas de perte par défaut
        'average_val_loss': [None] * len(results)     # SVC ne calcule pas de perte par défaut
    })

    # Sauvegarder les résultats dans data/results/svc
    model_results_dir = os.path.join(OUTPUT_DIR, 'svc')
    os.makedirs(model_results_dir, exist_ok=True)
    processed_results.to_csv(os.path.join(model_results_dir, 'cross_validation_results.csv'), index=False)
    print(f"Résultats de la cross-validation sauvegardés dans '{model_results_dir}/cross_validation_results.csv'")

    # Visualiser les métriques supplémentaires
    plot_metrics(processed_results, model_type='svc')

if __name__ == "__main__":
    main()

