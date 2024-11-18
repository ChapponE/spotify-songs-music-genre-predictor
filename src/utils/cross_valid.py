# src/utils/cross_valid.py

import numpy as np
import os
import json
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
        self.result_history = []
        self.model_name = self.model_class.__name__.lower()
        self.fixed_params = fixed_params if fixed_params is not None else {}

    def perform_cross_validation(self):
        model_param_combinations = list(ParameterGrid(self.model_param_grid))
        train_param_combinations = list(ParameterGrid(self.train_param_grid))
        
        for model_params in model_param_combinations:
            for train_params in train_param_combinations:
                metrics = {
                    'train_accuracy': [],
                    'val_accuracy': []
                }

                # Ajouter les métriques de loss si c'est un réseau de neurones
                if is_neural_network(self.model_class):
                    metrics['train_loss'] = []
                    metrics['val_loss'] = []

                for train_index, val_index in self.kf.split(self.X):
                    X_train, X_val = self.X[train_index], self.X[val_index]
                    y_train, y_val = self.y[train_index], self.y[val_index]

                    # Fusionner les paramètres fixes avec les hyperparamètres
                    full_model_params = {**self.fixed_params, **model_params}

                    model = self.model_class(**full_model_params)
                    history = model.train_model(X_train, y_train, X_val, y_val, **train_params)
                    self.result_history.append(history)

                    # Enregistrer l'accuracy
                    metrics['train_accuracy'].append(history['accuracy'][-1])
                    metrics['val_accuracy'].append(
                        history['val_accuracy'][-1] if 'val_accuracy' in history else None
                    )

                    # Enregistrer la loss si applicable
                    if is_neural_network(self.model_class):
                        metrics['train_loss'].append(history.get('loss', [None])[-1])
                        metrics['val_loss'].append(
                            history.get('val_loss', [None])[-1] if history.get('val_loss', [None])[-1] is not None else None
                        )

                # Calculer les moyennes
                avg_metrics = {
                    'model_params': model_params,
                    'train_params': train_params,
                    'average_train_accuracy': np.mean(metrics['train_accuracy']),
                    'average_val_accuracy': np.mean([acc for acc in metrics['val_accuracy'] if acc is not None]) if any(metrics['val_accuracy']) else None
                }

                if is_neural_network(self.model_class):
                    avg_metrics['average_train_loss'] = np.mean(metrics['train_loss']) if metrics['train_loss'] else None
                    avg_metrics['average_val_loss'] = np.mean([loss for loss in metrics['val_loss'] if loss is not None]) if any(metrics['val_loss']) else None

                self.results.append(avg_metrics)
                self._print_metrics(model_params, train_params, avg_metrics)

        best_result = self._select_best_model()
        return best_result

    def _print_metrics(self, model_params, train_params, avg_metrics):
        print(f"Model Configuration: {model_params}")
        print(f"Training Configuration: {train_params}")
        if is_neural_network(self.model_class):
            print(f"  Train Loss: {avg_metrics.get('average_train_loss', None):.4f}")
            if avg_metrics.get('average_val_loss', None) is not None:
                print(f"  Val Loss: {avg_metrics.get('average_val_loss'):.4f}")
        print(f"  Train Accuracy: {avg_metrics['average_train_accuracy']:.4f}")
        if avg_metrics.get('average_val_accuracy', None) is not None:
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

    def save_best_hyperparameters(self, best_result, filename, directory):
        with open(os.path.join(directory, filename), 'w') as f:
            json.dump(best_result, f, indent=4)

    def save_train_metrics(self, filename, directory):
        with open(os.path.join(directory, filename), 'w') as f:
            json.dump(self.result_history, f, indent=4)

