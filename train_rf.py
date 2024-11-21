# train_RF.py

import os
import pandas as pd
from src.data.data_loader import load_data_preprocessed
from src.utils.cross_valid import CrossValidator
from src.models.random_forest_model import RandomForestModel
from src.utils.config import RESULTS_DIR, RF_PARAM_GRID, N_SPLITS

def load_preprocessed_data():
    # Implémentez la méthode de chargement de vos données prétraitées
    # Par exemple :
    # X = pd.read_csv('data/preprocessed_X.csv')
    # y = pd.read_csv('data/preprocessed_y.csv')
    # return X.values, y.values.flatten()
    pass

def main():
    # Charger les données prétraitées
    X, y, scaler, label_encoder = load_data_preprocessed()

    # Définir les grilles de paramètres
    model_param_grid = RF_PARAM_GRID
    train_param_grid = {}
    fixed_params = {
        'random_state': 42
    }

    # Répertoire pour sauvegarder les résultats
    model_results_dir = os.path.join(RESULTS_DIR, 'random_forest')
    os.makedirs(model_results_dir, exist_ok=True)

    # Initialiser le CrossValidator
    cross_validator = CrossValidator(
        model_class=RandomForestModel,
        model_param_grid=model_param_grid,
        train_param_grid=train_param_grid,
        X=X,
        y=y,
        n_splits=N_SPLITS,
        fixed_params=fixed_params
    )

    # Effectuer la validation croisée
    best_result = cross_validator.perform_cross_validation()

    # Sauvegarder les meilleurs hyperparamètres
    cross_validator.save_best_hyperparameters(best_result, 'best_hyperparameters.json', model_results_dir)
    cross_validator.save_train_metrics('train_metrics.json', model_results_dir)

    # Sauvegarder les résultats complets
    results_df = pd.DataFrame(cross_validator.results)
    results_df.to_csv(os.path.join(model_results_dir, 'cross_validation_results.csv'), index=False)
    print(f"Résultats de la cross-validation sauvegardés dans '{model_results_dir}/cross_validation_results.csv'")

    # Extraire les meilleurs hyperparamètres
    best_model_params = best_result['model_params']
    best_score = best_result['average_val_accuracy']

    # Réentraîner le meilleur modèle Random Forest sur l'ensemble des données d'entraînement
    print(f"Retraining the best Random Forest model with params: {best_model_params}")

    best_rf = RandomForestModel(
        n_estimators=best_model_params.get('n_estimators', 100),
        max_depth=best_model_params.get('max_depth', None),
        random_state=best_model_params.get('random_state', 42)
    )

    best_rf.train_model(X_train=X, y_train=y)
    best_rf.save(os.path.join(model_results_dir, 'best_rf_model.pkl'))
    print("Training completed and best Random Forest model saved.")

    # Sauvegarder l'historique d'entraînement si nécessaire
    # Pour Random Forest, l'historique peut ne pas être particulièrement informatif
    # Vous pouvez choisir de ne pas le sauvegarder ou de sauvegarder d'autres métriques

if __name__ == "__main__":
    main()