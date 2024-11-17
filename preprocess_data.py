import os
import joblib
import pandas as pd
from src.data.data_loader import load_data, preprocess_data
from src.utils.config import PROCESSED_DIR

def main():
    # Charger les données brutes
    train_df, _ = load_data()
    
    # Prétraiter les données
    X_scaled, y_encoded, scaler, label_encoder = preprocess_data(train_df, is_train=True)
    
    # Assurer que le répertoire de sortie existe
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
    
    # Sauvegarder les données prétraitées
    joblib.dump(X_scaled, os.path.join(PROCESSED_DIR, 'X_train_scaled.pkl'))
    joblib.dump(y_encoded, os.path.join(PROCESSED_DIR, 'y_train_encoded.pkl'))
    
    # Sauvegarder les objets de prétraitement
    joblib.dump(scaler, os.path.join(PROCESSED_DIR, 'scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(PROCESSED_DIR, 'label_encoder.pkl'))
    
    print("Prétraitement terminé et données sauvegardées dans 'processed/'.")

if __name__ == "__main__":
    main()