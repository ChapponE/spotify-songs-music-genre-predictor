# predict_mlp.py

import os
import joblib
import pandas as pd
import json
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.data.data_loader import load_data, get_numerical_features
from src.models.mlp_model import MLPModel
from src.utils.config import PROCESSED_DIR, RESULTS_DIR, TEST_CSV

def main():
    # Charger les données de test
    _, test_df = load_data()

    # Charger les objets de prétraitement
    scaler = joblib.load(os.path.join(PROCESSED_DIR, 'scaler.pkl'))
    label_encoder = joblib.load(os.path.join(PROCESSED_DIR, 'label_encoder.pkl'))

    # Sélectionner les features numériques
    X_test = get_numerical_features(test_df)

    # Appliquer la normalisation
    X_test_scaled = scaler.transform(X_test)

    # Charger les meilleurs hyperparamètres
    best_hyperparams_path = os.path.join(RESULTS_DIR, 'mlp', 'best_hyperparameters.json')
    with open(best_hyperparams_path, 'r') as f:
        best_hyperparams = json.load(f)

    hidden_layers = best_hyperparams['model_params']['hidden_layers']
    # Vous pouvez ajuster la méthode pour récupérer 'learning_rate', 'optimizer_name', 'loss_name' si nécessaire
    learning_rate = best_hyperparams['train_params'].get('learning_rate', 0.001)
    optimizer_name = 'adam'  # Vous pouvez également charger cela depuis le fichier de configuration si nécessaire
    loss_name = 'cross_entropy'  # Idem que pour 'optimizer_name'

    # Déterminer 'input_size' et 'output_size'
    input_size = X_test_scaled.shape[1]
    output_size = len(label_encoder.classes_)

    # Initialiser le modèle avec les hyperparamètres optimaux
    model_mlp = MLPModel(
        hidden_layers=hidden_layers,
        input_size=input_size,
        output_size=output_size,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        loss_name=loss_name
    )

    # Charger les poids du modèle sauvegardé
    model_mlp_path = os.path.join(RESULTS_DIR, 'mlp', 'mlp_model.pth')  # Correction du chemin et de l'extension
    model_mlp.load(model_mlp_path)
    model_mlp.eval()  # Mettre le modèle en mode évaluation

    # Effectuer les prédictions
    with torch.no_grad():
        inputs = torch.tensor(X_test_scaled, dtype=torch.float32)
        outputs = model_mlp(inputs)
        predictions_mlp = torch.argmax(outputs, dim=1).numpy()

    # Convertir les prédictions encodées en labels originaux
    predicted_labels_mlp = label_encoder.inverse_transform(predictions_mlp)

    # Ajouter les prédictions au DataFrame de test
    test_df['predicted_genre_mlp'] = predicted_labels_mlp

    # Sauvegarder les prédictions
    prediction_path = os.path.join(RESULTS_DIR, 'test_predictions_mlp.csv')
    test_df.to_csv(prediction_path, index=False)
    print(f"Les prédictions MLP ont été sauvegardées dans '{prediction_path}'.")

    # Charger les données de test avec les vraies étiquettes
    test_df_with_labels = pd.read_csv(TEST_CSV, sep='\t')
    true_labels = test_df_with_labels['playlist_genre']

    # Calculer l'accuracy
    accuracy = accuracy_score(true_labels, predicted_labels_mlp)
    print(f"\nAccuracy du modèle MLP: {accuracy:.2%}")

    # Calculer et afficher la matrice de confusion
    conf_matrix = confusion_matrix(true_labels, predicted_labels_mlp)
    
    # Créer une figure plus grande
    plt.figure(figsize=(12, 8))
    
    # Créer la heatmap
    sns.heatmap(conf_matrix, 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    
    plt.title('Matrice de Confusion - MLP')
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies Étiquettes')
    
    # Rotation des labels pour une meilleure lisibilité
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    
    # Ajuster le layout pour éviter que les labels soient coupés
    plt.tight_layout()
    
    # Sauvegarder la matrice de confusion
    confusion_matrix_path = os.path.join(RESULTS_DIR, 'mlp', 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    print(f"La matrice de confusion a été sauvegardée dans '{confusion_matrix_path}'")
    plt.close()

if __name__ == "__main__":
    main()