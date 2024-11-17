import numpy as np
import os
import pandas as pd
from sklearn.model_selection import KFold, ParameterGrid
from src.utils.helpers import is_neural_network
from src.models.base_model import BaseModel
from src.utils.config import OUTPUT_DIR

class CrossValidator:
    def __init__(self, model_class, model_param_grid, train_param_grid, X, y, n_splits=5, fixed_params=None):
        self.model_class = model_class
        self.model_param_grid = model_param_grid
        self.train_param_grid = train_param_grid
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        self.results = []
        self.model_name = self.model_class.__name__.lower()
        self.fixed_params = fixed_params if fixed_params is not None else {}

    def perform_cross_validation(self):
        model_param_combinations = list(ParameterGrid(self.model_param_grid))
        train_param_combinations = list(ParameterGrid(self.train_param_grid))
        
        for model_params in model_param_combinations:
            for train_params in train_param_combinations:
                fold_metrics = {
                    'train_accuracy': [],
                    'val_accuracy': []
                }

                # Ajouter les métriques de loss si c'est un réseau de neurones
                if is_neural_network(self.model_class):
                    fold_metrics['train_loss'] = []
                    fold_metrics['val_loss'] = []

                for train_index, val_index in self.kf.split(self.X):
                    X_train, X_val = self.X[train_index], self.X[val_index]
                    y_train, y_val = self.y[train_index], self.y[val_index]

                    # Fusionner les paramètres fixes avec les hyperparamètres
                    full_model_params = {**self.fixed_params, **model_params}
                    print(f"Model parameters: {[name for name, _ in full_model_params.items()]}")
                    print(f"Training parameters: {[name for name, _ in train_params.items()]}")
                    
                    model = self.model_class(**full_model_params)
                    history = model.train(X_train, y_train, X_val, y_val, **train_params)

                    # Enregistrer l'accuracy
                    fold_metrics['train_accuracy'].append(history.history['accuracy'][-1])
                    fold_metrics['val_accuracy'].append(
                        history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else None
                    )

                    # Enregistrer la loss si applicable
                    if is_neural_network(self.model_class):
                        fold_metrics['train_loss'].append(history.history['loss'][-1])
                        fold_metrics['val_loss'].append(
                            history.history['val_loss'][-1] if 'val_loss' in history.history else None
                        )

                # Calculer les moyennes
                avg_metrics = {
                    'model_params': model_params,
                    'train_params': train_params,
                    'average_train_accuracy': np.mean(fold_metrics['train_accuracy']),
                    'average_val_accuracy': np.mean(fold_metrics['val_accuracy']) if any(fold_metrics['val_accuracy']) else None
                }

                if is_neural_network(self.model_class):
                    avg_metrics['average_train_loss'] = np.mean(fold_metrics['train_loss'])
                    avg_metrics['average_val_loss'] = np.mean(fold_metrics['val_loss']) if any(fold_metrics['val_loss']) else None

                self.results.append(avg_metrics)
                self._print_metrics(model_params, train_params, avg_metrics)

        best_result = self._select_best_model()
        self._save_results(best_result)
        return best_result

    def _print_metrics(self, model_params, train_params, avg_metrics):
        print(f"Model Configuration: {model_params}")
        print(f"Training Configuration: {train_params}")
        if is_neural_network(self.model_class):
            print(f"  Train Loss: {avg_metrics['average_train_loss']:.4f}")
            if avg_metrics['average_val_loss'] is not None:
                print(f"  Val Loss: {avg_metrics['average_val_loss']:.4f}")
        print(f"  Train Accuracy: {avg_metrics['average_train_accuracy']:.4f}")
        if avg_metrics['average_val_accuracy'] is not None:
            print(f"  Val Accuracy: {avg_metrics['average_val_accuracy']:.4f}")
        print()

    def _select_best_model(self):
        best_result = max(
            [res for res in self.results if res['average_val_accuracy'] is not None],
            key=lambda x: x['average_val_accuracy']
        )
        print(f"Meilleure Configuration Modèle: {best_result['model_params']}")
        print(f"Meilleure Configuration Entrainement: {best_result['train_params']}")
        print(f"Accuracy de Validation: {best_result['average_val_accuracy']:.4f}")
        return best_result

    def _save_results(self, best_result):
        model_results_dir = os.path.join(OUTPUT_DIR, self.model_name)
        os.makedirs(model_results_dir, exist_ok=True)

        # Sauvegarder les meilleurs hyperparamètres
        self.save_best_hyperparameters(best_result, 'best_hyperparameters.json', model_results_dir)

        # Sauvegarder les résultats complets
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(os.path.join(model_results_dir, 'cross_validation_results.csv'), index=False)
        print(f"Résultats de la cross-validation sauvegardés dans '{model_results_dir}/cross_validation_results.csv'")

    def save_best_hyperparameters(self, best_result, filename, directory):
        import json
        with open(os.path.join(directory, filename), 'w') as f:
            json.dump(best_result, f, indent=4)
        print(f"Meilleurs hyperparamètres sauvegardés dans '{directory}/{filename}'")