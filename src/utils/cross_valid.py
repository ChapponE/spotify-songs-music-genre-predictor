# src/utils/cross_valid.py

import numpy as np
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, ParameterGrid
from src.utils.helpers import is_neural_network
from src.models.base_model import BaseModel
from src.utils.config import RESULTS_DIR

class CrossValidator:
    def __init__(self, model_class, model_param_grid, train_param_grid, X, y, n_splits=5, fixed_params=None):
        self.model_class = model_class
        model_param_combinations = list(ParameterGrid(model_param_grid))

        #initialisation d'une instance pour avoir accès à la propriété neural_network
        if model_param_combinations:
            selected_params = model_param_combinations[0]
            self.model_instance = model_class(**selected_params, **fixed_params)
        else:
            self.model_instance = model_class(**fixed_params)
        
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
                if self.model_instance.neural_network:
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
                    if self.model_instance.neural_network:
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

                if self.model_instance.neural_network:
                    avg_metrics['average_train_loss'] = np.mean(metrics['train_loss']) if metrics['train_loss'] else None
                    avg_metrics['average_val_loss'] = np.mean([loss for loss in metrics['val_loss'] if loss is not None]) if any(metrics['val_loss']) else None

                self.results.append(avg_metrics)
                self._print_metrics(model_params, train_params, avg_metrics)

        best_result = self._select_best_model()
        return best_result

    def _print_metrics(self, model_params, train_params, avg_metrics):
        print(f"Model Configuration: {model_params}")
        print(f"Training Configuration: {train_params}")
        if self.model_instance.neural_network:
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

    def save_loss_plot(self, directory, metrics_filename='train_metrics.json', results_csv='cross_validation_results.csv'):
        metrics_path = os.path.join(directory, metrics_filename)
        if not os.path.exists(metrics_path):
            print(f"Le fichier des métriques d'entraînement '{metrics_path}' n'existe pas.")
            return

        results_csv_path = os.path.join(directory, results_csv)
        if not os.path.exists(results_csv_path):
            print(f"Le fichier des résultats de validation croisée '{results_csv_path}' n'existe pas.")
            return

        with open(metrics_path, 'r') as f:
            train_metrics = json.load(f)

        results_df = pd.read_csv(results_csv_path)

        num_configs = len(train_metrics)
        if num_configs == 0:
            print("Aucune métrique de loss disponible pour le tracé.")
            return

        # Définir le nombre de configurations par colonne
        configs_per_column = self.n_splits * 2  # Nombre de folds * 2
        num_columns = int(np.ceil(num_configs / configs_per_column))
        num_rows = configs_per_column

        fig, axes = plt.subplots(num_rows, num_columns, figsize=(5 * num_columns, 5 * num_rows), squeeze=False)
        axes = axes.flatten(order='F')  # Aplatir par colonnes

        for i, config_metrics in enumerate(train_metrics):
            loss = config_metrics.get('loss', [])
            val_loss = config_metrics.get('val_loss', [])

            if not loss:
                print(f"Configuration {i+1} n'a pas de métriques de loss. Ignorée.")
                continue

            if i >= len(axes):
                print(f"Index {i} hors limites pour les axes disponibles. Ignorée.")
                continue

            ax = axes[i]
            epochs = range(1, len(loss) + 1)

            ax.plot(epochs, loss, label='Train Loss', marker='o')
            if val_loss:
                ax.plot(epochs, val_loss, label='Validation Loss', marker='x')

            ax.set_yscale('log')
            ax.set_xlabel('Époque')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)

            # Calculer les indices de ligne et de colonne
            row = i % num_rows
            col = i // num_rows

            # Définir le titre tous les 4 itérations
            if i % 4 == 0:
                csv_index = i // 4
                if csv_index < len(results_df):
                    model_params_str = results_df.loc[csv_index, 'model_params']
                    train_params_str = results_df.loc[csv_index, 'train_params']

                    try:
                        model_params = eval(model_params_str)
                        train_params = eval(train_params_str)
                    except Exception as e:
                        print(f"Erreur lors de l'évaluation des paramètres à l'index {csv_index}: {e}")
                        model_params = {}
                        train_params = {}

                    hidden_layers = model_params.get('hidden_layers', [])
                    batch_size = train_params.get('batch_size', '')
                    epochs_param = train_params.get('epochs', '')
                    learning_rate = train_params.get('learning_rate', '')

                    hidden_layers_str = ','.join(map(str, hidden_layers)) if isinstance(hidden_layers, list) else str(hidden_layers)
                    # Correction du titre sans les guillemets
                    title = f"hl[{hidden_layers_str}]_bs{batch_size}_e{epochs_param}_lr{learning_rate}"
                    ax.set_title(title, color='red', fontsize=20)
                else:
                    ax.set_title(f'Configuration {i+1}')

            # Ajouter le numéro de fold à gauche du plot le plus à gauche de chaque ligne
            if col == 0:
                fold_number = (row % 4) + 1
                # Ajouter le texte du fold à gauche du plot
                ax.annotate(f'Fold {fold_number}', xy=(-0.3, 0.5), xycoords='axes fraction',
                            ha='right', va='center', fontsize=20, rotation=90, color='red')

        # Cacher les sous-graphiques non utilisés
        for j in range(num_configs, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plot_path = os.path.join(directory, 'loss_plot.png')
        plt.savefig(plot_path)
        print(f"Les graphiques de loss ont été sauvegardés dans '{plot_path}'.")
        plt.close()
