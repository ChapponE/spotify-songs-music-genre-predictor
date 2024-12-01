# preprocess_data_full.py

import os
import joblib
import pandas as pd
from src.data.data_loader import load_data, preprocess_data_full
from src.utils.config import PROCESSED_DIR, TEST

def main():
    # Charger les données brutes complètes
    train_df, _ = load_data()

    # Prétraiter les données avec la nouvelle fonction
    X_processed, y_encoded, scaler, label_encoder, text_encoder = preprocess_data_full(train_df, is_train=True)

    # Assurer que le répertoire de sortie existe
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    # Préparer les données à sauvegarder
    data_to_save = {
        'X_train_processed': X_processed,
        'y_train_encoded': y_encoded,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'text_encoder': text_encoder,
    }

    # Déterminer le nom du fichier en fonction de TEST
    filename = 'preprocessed_data_test_full.pkl' if TEST else 'preprocessed_data_full.pkl'
    data_path = os.path.join(PROCESSED_DIR, filename)

    # Sauvegarder les données prétraitées
    joblib.dump(data_to_save, data_path)

    print(f"Prétraitement terminé et données sauvegardées dans '{data_path}'.")

if __name__ == "__main__":
    main()