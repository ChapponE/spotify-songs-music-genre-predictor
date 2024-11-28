import os
import joblib
import pandas as pd
from src.data.data_loader import load_data, preprocess_data
from src.utils.config import PROCESSED_DIR, TEST

def main():
    # Charger les données brutes
    train_df, _ = load_data()
    
    # Prétraiter les données
    X_scaled, y_encoded, scaler, label_encoder = preprocess_data(train_df, is_train=True)
    
    # Assurer que le répertoire de sortie existe
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
    
    # Préparer les données à sauvegarder
    data_to_save = {
        'X_train_scaled': X_scaled,
        'y_train_encoded': y_encoded,
        'scaler': scaler,
        'label_encoder': label_encoder
    }
    
    # Déterminer le nom du fichier en fonction de TEST
    filename = 'preprocessed_data_test.pkl' if TEST else 'preprocessed_data.pkl'
    data_path = os.path.join(PROCESSED_DIR, filename)
    
    # Sauvegarder les données prétraitées
    joblib.dump(data_to_save, data_path)
    
    print(f"Prétraitement terminé et données sauvegardées dans '{data_path}'.")

if __name__ == "__main__":
    main()
