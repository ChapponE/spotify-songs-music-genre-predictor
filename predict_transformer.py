# predict_transformer.py

import os
import joblib
import pandas as pd
import torch
import json
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.data.data_loader import load_data, get_numerical_features
from src.models.transformer_model import TransformerModel
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
    X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)

    # Charger les meilleurs hyperparamètres
    best_hyperparams_path = os.path.join(RESULTS_DIR, 'transformer', 'best_hyperparameters.json')
    with open(best_hyperparams_path, 'r') as f:
        best_hyperparams = json.load(f)

    model_params = best_hyperparams['model_params']
    train_params = best_hyperparams['train_params']

    # Déterminer 'input_size' et 'output_size'
    input_size = X_test_scaled.shape[1]
    output_size = len(label_encoder.classes_)

    # Initialiser le modèle avec les meilleurs hyperparamètres
    model_transformer = TransformerModel(
        input_size=input_size,
        output_size=output_size,
        learning_rate=train_params.get('learning_rate', 0.001),
        optimizer_name='adam',
        loss_name='cross_entropy',
        **model_params
    )

    # Charger les poids du modèle sauvegardé
    model_path = os.path.join(RESULTS_DIR, 'transformer', 'transformer_model.pth')
    model_transformer.load(model_path)
    model_transformer.eval()

    # Effectuer les prédictions
    with torch.no_grad():
        outputs = model_transformer(X_test_scaled)
        predictions = torch.argmax(outputs, dim=1).numpy()

    # Convertir les prédictions encodées en labels originaux
    predicted_labels = label_encoder.inverse_transform(predictions)

    # Ajouter les prédictions au DataFrame de test
    test_df['predicted_genre_transformer'] = predicted_labels

    # Sauvegarder les prédictions
    prediction_path = os.path.join(RESULTS_DIR, 'test_predictions_transformer.csv')
    test_df.to_csv(prediction_path, index=False)
    print(f"Les prédictions Transformer ont été sauvegardées dans '{prediction_path}'.")

    # Charger les données de test avec les vraies étiquettes
    test_df_with_labels = pd.read_csv(TEST_CSV, sep='\t')
    true_labels = test_df_with_labels['playlist_genre']

    # Calculer l'accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"\nAccuracy du modèle Transformer: {accuracy:.2%}")

    # Calculer et afficher la matrice de confusion
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Créer une figure plus grande
    plt.figure(figsize=(12, 8))

    # Créer la heatmap
    sns.heatmap(conf_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)

    plt.title('Matrice de Confusion - Transformer')
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies Étiquettes')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()

    # Sauvegarder la matrice de confusion
    confusion_matrix_path = os.path.join(RESULTS_DIR, 'transformer', 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    print(f"La matrice de confusion a été sauvegardée dans '{confusion_matrix_path}'")
    plt.close()

if __name__ == "__main__":
    main()
