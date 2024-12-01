import os
import pandas as pd
import ast
from src.utils import config
from itertools import product

# Définir le répertoire des résultats
RESULTS_DIR = config.RESULTS_DIR 

# Parcourir chaque dossier de modèle dans le répertoire des résultats
for model_name in os.listdir(RESULTS_DIR):
    model_dir = os.path.join(RESULTS_DIR, model_name)
    cross_val_file = os.path.join(model_dir, 'cross_validation_results.csv')

    # Vérifier si le fichier cross_validation_results.csv existe
    if os.path.isfile(cross_val_file):
        # Lire le fichier CSV
        df = pd.read_csv(cross_val_file)

        # Liste pour stocker les données d'analyse
        analysis_data = []

        # Fonction pour extraire et analyser les hyperparamètres
        def analyze_hyperparams(param_column, param_source):
            hyperparams_list = []
            for params_str in df[param_column]:
                params_dict = ast.literal_eval(params_str)
                processed_params_dict = {}
                for key, value in params_dict.items():
                    if isinstance(value, list):
                        processed_params_dict[key] = tuple(value)
                    else:
                        processed_params_dict[key] = value
                hyperparams_list.append(processed_params_dict)

            # Ajouter les hyperparamètres au DataFrame
            hyperparams_df = pd.DataFrame(hyperparams_list)
            temp_df = pd.concat([df.reset_index(drop=True), hyperparams_df.reset_index(drop=True)], axis=1)

            # Identifier les hyperparamètres avec au moins 2 valeurs différentes
            hyperparams = []
            for col in hyperparams_df.columns:
                # Convertir les listes en tuples pour les rendre hachables
                if hyperparams_df[col].apply(lambda x: isinstance(x, list)).any():
                    hyperparams_df[col] = hyperparams_df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)
                if hyperparams_df[col].nunique() >= 2:
                    hyperparams.append(col)

            # Préparer les données pour l'analyse
            for hyperparam in hyperparams:
                unique_values = hyperparams_df[hyperparam].unique()
                for value in unique_values:
                    subset = temp_df[hyperparams_df[hyperparam] == value]
                    if not subset.empty:
                        avg_train_acc = subset['average_train_accuracy'].mean()
                        avg_val_acc = subset['average_val_accuracy'].mean()
                        analysis_data.append({
                            'hyperparametre': f"{hyperparam}={value}",
                            'accuracy moyenne train': avg_train_acc,
                            'accuracy moyenne test': avg_val_acc
                        })

        # Analyser les hyperparamètres de 'model_params' pour tous les modèles
        analyze_hyperparams('model_params', 'model_params')

        # Pour 'mlp' et 'mlp_full', analyser également 'train_params' si non vide
        if model_name in ['mlp', 'mlp_full'] and not df['train_params'].isnull().all():
            # Vérifier que 'train_params' n'est pas vide
            if df['train_params'].apply(lambda x: ast.literal_eval(x) != {}).any():
                analyze_hyperparams('train_params', 'train_params')

        # Créer le DataFrame d'analyse
        analysis_df = pd.DataFrame(analysis_data)

        # Trier par 'accuracy moyenne test' décroissante
        analysis_df = analysis_df.sort_values(by='accuracy moyenne test', ascending=False)

        # Sauvegarder le résultat dans le fichier hyperparameters_analysis.csv
        output_file = os.path.join(model_dir, 'hyperparameters_analysis.csv')
        analysis_df.to_csv(output_file, index=False)
        print(f"Analyse des hyperparamètres pour le modèle '{model_name}' enregistrée dans '{output_file}'.")
    else:
        print(f"Aucun fichier 'cross_validation_results.csv' trouvé dans le dossier '{model_dir}'.")