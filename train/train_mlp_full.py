# train_mlp_full.py

import os
import joblib
import pandas as pd
import numpy as np
from src.data.data_loader import load_data_preprocessed_full
from src.utils.config import RESULTS_DIR, MLP_PARAMS, N_SPLITS, OPTIMIZER, LOSS_FUNCTION
from src.utils.cross_valid import CrossValidator
from src.models.mlp_model import MLPModel
from src.visualization.plots import plot_metrics

def main():
    # Charger les données prétraitées complètes
    X, y, scaler, label_encoder, text_encoder = load_data_preprocessed_full()

    input_size = X.shape[1]
    output_size = len(np.unique(y))

    # Définir la grille d'hyperparamètres pour le MLP
    model_param_grid = MLP_PARAMS['mlp_param_grid']
    train_param_grid = MLP_PARAMS['train_param_grid']

    # Définir les paramètres fixes
    fixed_params = {
        'input_size': input_size,
        'output_size': output_size,
        'optimizer_name': OPTIMIZER,
        'loss_name': LOSS_FUNCTION
    }

    # Initialiser et exécuter la validation croisée pour le MLP
    cross_validator = CrossValidator(
        model_class=MLPModel,
        model_param_grid=model_param_grid,
        train_param_grid=train_param_grid,
        X=X,
        y=y,
        n_splits=N_SPLITS,
        fixed_params=fixed_params 
    )
    best_result = cross_validator.perform_cross_validation()

    # Définir le répertoire des résultats
    model_results_dir = os.path.join(RESULTS_DIR, 'mlp_full')
    os.makedirs(model_results_dir, exist_ok=True)

    # Sauvegarder les meilleurs hyperparamètres
    cross_validator.save_best_hyperparameters(best_result, 'best_hyperparameters.json', model_results_dir)
    cross_validator.save_train_metrics('train_metrics.json', model_results_dir)

    # Sauvegarder les résultats complets de la cross-validation
    results_df = pd.DataFrame(cross_validator.results)
    results_df.to_csv(os.path.join(model_results_dir, 'cross_validation_results.csv'), index=False)
    print(f"Résultats de la cross-validation sauvegardés dans '{model_results_dir}/cross_validation_results.csv'")

    # Extraire les meilleurs hyperparamètres
    best_model_params = best_result['model_params']
    best_train_params = best_result['train_params']
    
    best_hidden_layers = best_model_params['hidden_layers']
    best_learning_rate = best_train_params['learning_rate']
    best_epochs = best_train_params.get('epochs', 10)
    batch_size = best_train_params.get('batch_size', 32)

    # Réentraîner le meilleur modèle MLP sur l'ensemble des données d'entraînement
    print(f"Réentraînement du meilleur modèle MLP avec taux d'apprentissage: {best_learning_rate}, couches cachées: {best_hidden_layers}, epochs: {best_epochs}, batch size: {batch_size}")

    model = MLPModel(
        hidden_layers=best_hidden_layers,
        input_size=input_size,
        output_size=output_size,
        learning_rate=best_learning_rate,
        optimizer_name=OPTIMIZER,
        loss_name=LOSS_FUNCTION
    )

    # Pas de validation ici
    history = model.train_model(
        X_train=X,
        y_train=y,
        X_val=None,
        y_val=None,
        epochs=best_epochs,
        batch_size=batch_size
    )
    model.save(os.path.join(model_results_dir, 'mlp_model_full.pth'))
    print("Entraînement terminé et modèle MLP sauvegardé.")

    # Sauvegarder l'historique d'entraînement
    train_history = pd.DataFrame(history)
    train_history.to_csv(os.path.join(model_results_dir, 'training_history.csv'), index=False)
    print(f"Historique d'entraînement sauvegardé dans '{model_results_dir}/training_history.csv'")

    # Visualiser les métriques de Cross-Validation
    plot_metrics(results_df, model_type='mlp_full')
    cross_validator.save_loss_plot(model_results_dir)

if __name__ == "__main__":
    main()
