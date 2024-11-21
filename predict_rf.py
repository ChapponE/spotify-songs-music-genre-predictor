# predict_rf.py

import os
import joblib
import pandas as pd
import json
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.data.data_loader import load_data, get_numerical_features
from src.models.random_forest_model import RandomForestModel
from src.utils.config import PROCESSED_DIR, RESULTS_DIR, TEST_CSV

def main():
    # Charger les données de test
    _, test_df = load_data()

    # Charger les objets de prétraitement
    scaler_path = os.path.join(PROCESSED_DIR, 'scaler.pkl')
    label_encoder_path = os.path.join(PROCESSED_DIR, 'label_encoder.pkl')
    
    # Vérifier si le scaler existe avant de le charger
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("Scaler chargé.")
    else:
        scaler = None
        print("Scaler non trouvé. Aucune normalisation appliquée.")

    # Charger le label encoder
    label_encoder = joblib.load(label_encoder_path)
    print("Label encoder chargé.")

    # Sélectionner les features numériques
    X_test = get_numerical_features(test_df)

    # Appliquer la normalisation si un scaler est disponible
    if scaler:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test.values  # Convertir en numpy array si nécessaire

    # Charger les meilleurs hyperparamètres
    best_hyperparams_path = os.path.join(RESULTS_DIR, 'random_forest', 'best_hyperparameters.json')
    with open(best_hyperparams_path, 'r') as f:
        best_hyperparams = json.load(f)

    # Extraire les hyperparamètres du modèle
    best_model_params = best_hyperparams['model_params']
    
    # Initialiser le modèle avec les hyperparamètres optimaux
    best_rf = RandomForestModel(
        n_estimators=best_model_params.get('n_estimators', 100),
        max_depth=best_model_params.get('max_depth', None),
        min_samples_split=best_model_params.get('min_samples_split', 2),
        min_samples_leaf=best_model_params.get('min_samples_leaf', 1),
        bootstrap=best_model_params.get('bootstrap', True),
        random_state=best_hyperparams['model_params'].get('random_state', 42)
    )

    # Charger le modèle entraîné
    model_rf_path = os.path.join(RESULTS_DIR, 'random_forest', 'best_rf_model.pkl')
    best_rf.load(model_rf_path)
    print(f"Modèle Random Forest chargé depuis '{model_rf_path}'.")

    # Effectuer les prédictions
    predictions_rf = best_rf.model.predict(X_test_scaled)

    # Convertir les prédictions encodées en labels originaux
    predicted_labels_rf = label_encoder.inverse_transform(predictions_rf)

    # Ajouter les prédictions au DataFrame de test
    test_df['predicted_genre_rf'] = predicted_labels_rf

    # Sauvegarder les prédictions
    prediction_path = os.path.join(RESULTS_DIR, 'random_forest', 'test_predictions_rf.csv')
    test_df.to_csv(prediction_path, index=False)
    print(f"Les prédictions Random Forest ont été sauvegardées dans '{prediction_path}'.")

    # Charger les vraies étiquettes si disponibles
    if os.path.exists(TEST_CSV):
        test_df_with_labels = pd.read_csv(TEST_CSV, sep='\t')  # Adaptez le séparateur si nécessaire
        true_labels = test_df_with_labels['playlist_genre']
        
        # Vérifier que les tailles correspondent
        if len(true_labels) != len(predicted_labels_rf):
            print("Le nombre de vraies étiquettes ne correspond pas au nombre de prédictions.")
            true_labels = None
        else:
            print("Le nombre de vraies étiquettes correspond au nombre de prédictions.")
    else:
        true_labels = None
        print(f"Fichier des vraies étiquettes '{TEST_CSV}' non trouvé. Accuracy et matrice de confusion non calculées.")

    if true_labels is not None:
        # Assurez-vous que true_labels et predicted_labels_rf sont du même type (strings)
        true_labels = true_labels.astype(str)
        predicted_labels_rf = predicted_labels_rf.astype(str)

        # Ajout de débogage pour vérifier les types et les premières valeurs
        print(f"Type de true_labels: {type(true_labels.iloc[0])}")
        print(f"Type de predicted_labels_rf: {type(predicted_labels_rf[0])}")
        print("Exemples de true_labels:", true_labels.head())
        print("Exemples de predicted_labels_rf:", pd.Series(predicted_labels_rf).head())

        # Calculer l'accuracy
        accuracy = accuracy_score(true_labels, predicted_labels_rf)
        print(f"\nAccuracy du modèle Random Forest: {accuracy:.2%}")

        # Calculer et afficher la matrice de confusion
        conf_matrix = confusion_matrix(true_labels, predicted_labels_rf)
        print("Matrice de confusion:")
        print(conf_matrix)
        
        # Créer une figure plus grande
        plt.figure(figsize=(12, 8))
        
        # Créer la heatmap
        sns.heatmap(conf_matrix, 
                    annot=True, 
                    fmt='d',
                    cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        
        plt.title('Matrice de Confusion - Random Forest')
        plt.xlabel('Prédictions')
        plt.ylabel('Vraies Étiquettes')
        
        # Rotation des labels pour une meilleure lisibilité
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45)
        
        # Ajuster le layout pour éviter que les labels soient coupés
        plt.tight_layout()
        
        # Sauvegarder la matrice de confusion
        confusion_matrix_path = os.path.join(RESULTS_DIR, 'random_forest', 'confusion_matrix_rf.png')
        plt.savefig(confusion_matrix_path)
        print(f"La matrice de confusion a été sauvegardée dans '{confusion_matrix_path}'")
        plt.close()
    else:
        print("Les vraies étiquettes de test ne sont pas disponibles ou ne correspondent pas aux prédictions. Accuracy et matrice de confusion non calculées.")

if __name__ == "__main__":
    main()