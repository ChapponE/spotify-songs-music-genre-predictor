import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.data.data_loader import load_data, get_numerical_features
from src.utils.config import OUTPUT_DIR, BASE_DIR

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

    # Charger le meilleur modèle SVM
    model_svc_path = os.path.join(BASE_DIR, 'best_svc_model.pkl')
    model_svc = joblib.load(model_svc_path)

    # Prédire les genres avec le SVM
    predicted_classes_svc = model_svc.predict(X_test_scaled)
    predicted_labels_svc = label_encoder.inverse_transform(predicted_classes_svc)

    # Ajouter les prédictions au DataFrame de test
    test_df['predicted_genre_svc'] = predicted_labels_svc

    # Sauvegarder les prédictions
    prediction_path = os.path.join(OUTPUT_DIR, 'test_predictions_svc.csv')
    test_df.to_csv(prediction_path, index=False)
    print(f"Les prédictions SVM ont été sauvegardées dans '{prediction_path}'.")

if __name__ == "__main__":
    main()