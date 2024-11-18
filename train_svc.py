# train_svc.py

import os
import joblib
import pandas as pd
from src.data.data_loader import load_data_preprocessed
from src.utils.config import OUTPUT_DIR, SVC_PARAM_GRID, N_SPLITS
from src.utils.cross_valid import CrossValidator
from src.models.svc_model import SVCModel
from src.visualization.plots import plot_metrics

def main():
    # Charger les données prétraitées
    X, y, scaler, label_encoder = load_data_preprocessed()

    # Définir la grille d'hyperparamètres pour le SVC
    model_param_grid = SVC_PARAM_GRID
    train_param_grid = {}  # SVC n'a pas de paramètres d'entraînement spécifiques

    # Définir les paramètres fixes
    fixed_params = {
        'probability': False,
        'random_state': 42
    }

    # Initialiser et exécuter la validation croisée pour le SVC
    cross_validator = CrossValidator(
        model_class=SVCModel,
        model_param_grid=model_param_grid,
        train_param_grid=train_param_grid,
        X=X,
        y=y,
        n_splits=N_SPLITS,
        fixed_params=fixed_params
    )
    best_result = cross_validator.perform_cross_validation()

    # Définir le répertoire des résultats
    model_results_dir = os.path.join(OUTPUT_DIR, 'svc')
    os.makedirs(model_results_dir, exist_ok=True)

    # Sauvegarder les meilleurs hyperparamètres
    cross_validator.save_best_hyperparameters(best_result, 'best_hyperparameters.json', model_results_dir)
    cross_validator.save_train_metrics('train_metrics.json', model_results_dir)
    
    # Sauvegarder les résultats complets dans data/results/svc
    results_df = pd.DataFrame(cross_validator.results)
    results_df.to_csv(os.path.join(model_results_dir, 'cross_validation_results.csv'), index=False)
    print(f"Résultats de la cross-validation sauvegardés dans '{model_results_dir}/cross_validation_results.csv'")

    # Extraire les meilleurs hyperparamètres
    best_model_params = best_result['model_params']
    best_score = best_result['average_val_accuracy']

    # Réentraîner le meilleur modèle SVC sur l'ensemble des données d'entraînement
    print(f"Retraining the best SVC model with params: {best_model_params}")

    best_svc = SVCModel(**best_model_params, probability=False, random_state=42)
    best_svc.train_model(X, y)

    # Sauvegarder le meilleur modèle et les objets de prétraitement
    best_model_path = os.path.join(model_results_dir, 'best_svc_model.pkl')
    best_svc.save(best_model_path)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, 'label_encoder.pkl'))
    print("Entraînement terminé et meilleur modèle SVC sauvegardé.")

    # Visualiser les métriques supplémentaires
    plot_metrics(results_df, model_type='svc')

if __name__ == "__main__":
    main()