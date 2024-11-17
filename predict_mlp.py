import os
import joblib
import pandas as pd
from src.data.data_loader import load_data, get_numerical_features
from src.models.mlp_model import MLPModel
from src.utils.config import OUTPUT_DIR

def main():
    # Charger les données de test
    _, test_df = load_data()

    # Charger les objets de prétraitement
    scaler = joblib.load(os.path.join(OUTPUT_DIR, 'scaler.pkl'))
    label_encoder = joblib.load(os.path.join(OUTPUT_DIR, 'label_encoder.pkl'))

    # Sélectionner les features numériques
    X_test = get_numerical_features(test_df)

    # Appliquer la normalisation
    X_test_scaled = scaler.transform(X_test)

    # Charger le modèle MLP
    model_mlp = MLPModel()  # Aucun paramètre requis car le modèle est chargé
    model_mlp.load(os.path.join(OUTPUT_DIR, 'mlp_model.h5'))

    # Prédire les genres avec le MLP
    predictions_mlp = model_mlp.predict(X_test_scaled)
    predicted_labels_mlp = label_encoder.inverse_transform(predictions_mlp)

    # Ajouter les prédictions au DataFrame de test
    test_df['predicted_genre_mlp'] = predicted_labels_mlp

    # Sauvegarder les prédictions
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test_predictions_mlp.csv'), index=False)
    print("Les prédictions MLP ont été sauvegardées dans 'test_predictions_mlp.csv'.")

if __name__ == "__main__":
    main()