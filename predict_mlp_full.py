
# predict_mlp_full.py

import os
import joblib
import pandas as pd
import json
import torch
from src.data.data_loader import load_data, preprocess_data_full
from src.models.mlp_model import MLPModel
from src.utils.config import PROCESSED_DIR, RESULTS_DIR, TEST_FULL_CSV

def main():
    # Charger les données de test complètes
    _, test_df = load_data(all_data=True)
    
    # **Assurez-vous que 'playlist_subgenre' est déjà exclue par le préprocesseur**
    # Pas besoin de le supprimer ici si le préprocesseur l'a déjà fait
    # Si nécessaire, pour exclure 'playlist_name', assurez-vous que 'data_loader.py' l'exclut
    
    # Charger les objets de prétraitement
    data_path = os.path.join(PROCESSED_DIR, 'preprocessed_data_full.pkl')
    if not os.path.exists(data_path):
        data_path = os.path.join(PROCESSED_DIR, 'preprocessed_data_test_full.pkl')
    data = joblib.load(data_path)
    scaler = data['scaler']
    label_encoder = data['label_encoder']
    text_encoder = data['text_encoder']

    # Prétraiter les données de test
    X_test_processed, _, _, _, _ = preprocess_data_full(
        test_df,
        scaler=scaler,
        label_encoder=label_encoder,
        text_encoder=text_encoder,
        is_train=False
    )

    # Charger les meilleurs hyperparamètres
    best_hyperparams_path = os.path.join(RESULTS_DIR, 'mlp_full', 'best_hyperparameters.json')
    with open(best_hyperparams_path, 'r') as f:
        best_hyperparams = json.load(f)

    hidden_layers = best_hyperparams['model_params']['hidden_layers']
    learning_rate = best_hyperparams['train_params'].get('learning_rate', 0.001)
    optimizer_name = 'adam'  # Ajustez si nécessaire
    loss_name = 'cross_entropy'  # Ajustez si nécessaire

    # Déterminer 'input_size' et 'output_size'
    input_size = X_test_processed.shape[1]
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
    model_mlp_path = os.path.join(RESULTS_DIR, 'mlp_full', 'mlp_model_full.pth')
    model_mlp.load(model_mlp_path)
    model_mlp.eval()  # Mettre le modèle en mode évaluation

    # Effectuer les prédictions
    with torch.no_grad():
        inputs = torch.tensor(X_test_processed, dtype=torch.float32)
        outputs = model_mlp(inputs)
        predictions_mlp = torch.argmax(outputs, dim=1).numpy()

    # Convertir les prédictions encodées en labels originaux
    predicted_labels_mlp = label_encoder.inverse_transform(predictions_mlp)

    # Ajouter les prédictions dans la colonne 'playlist_genre'
    test_df['playlist_genre'] = predicted_labels_mlp

    # Définir le répertoire des prédictions
    predictions_dir = os.path.join('predictions')
    os.makedirs(predictions_dir, exist_ok=True)

    # Sauvegarder les prédictions avec le nom spécifié
    prediction_filename = 'CHAPPON_prediction_full.csv'
    prediction_path = os.path.join(predictions_dir, prediction_filename)
    test_df.to_csv(prediction_path, index=False)
    print(f"Les prédictions MLP ont été sauvegardées dans '{prediction_path}'.")

if __name__ == "__main__":
    main()