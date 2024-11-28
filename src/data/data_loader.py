import os
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.utils.config import TRAIN_CSV, TEST_CSV, RESULTS_DIR, PROCESSED_DIR, TEST

def load_data(all_data=False):
    # Lire les fichiers CSV avec séparateur tabulation
    train_df = pd.read_csv(TRAIN_CSV, sep='\t')
    test_df = pd.read_csv(TEST_CSV, sep='\t')
    
    if TEST and not all_data:
        train_df = train_df.head(100)
    
    if not all_data:
        train_df = train_df.head(1000)
        
    return train_df, test_df

def load_data_preprocessed():
    """
    Charge les données prétraitées et les objets de prétraitement depuis un fichier unique dans le répertoire processed.
    
    Retourne:
        X_train_scaled (array): Features d'entraînement prétraités
        y_train_encoded (array): Labels d'entraînement encodés
        scaler (StandardScaler): Objet de scalage
        label_encoder (LabelEncoder): Objet d'encodage des labels
    """
    # Déterminer le nom du fichier en fonction de TEST
    filename = 'preprocessed_data_test.pkl' if TEST else 'preprocessed_data.pkl'
    data_path = os.path.join(PROCESSED_DIR, filename)
    
    # Charger les données
    data = joblib.load(data_path)
    X_train_scaled = data['X_train_scaled']
    y_train_encoded = data['y_train_encoded']
    scaler = data['scaler']
    label_encoder = data['label_encoder']
    
    return X_train_scaled, y_train_encoded, scaler, label_encoder

def get_numerical_features(df):
    # Sélectionner uniquement les colonnes numériques
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'playlist_genre' in numerical_cols:
        numerical_cols.remove('playlist_genre')
        numerical_cols.remove('liveness')
        numerical_cols.remove('energy')
        numerical_cols.remove('mode')
    X_numeric = df[numerical_cols]
    return X_numeric

def get_categorical_features(df):
    # Sélectionner uniquement les colonnes catégorielles
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'playlist_genre' in categorical_cols:
        categorical_cols.remove('playlist_genre')
    X_categorical = df[categorical_cols]
    return X_categorical    

def get_class_counts(df):
    # Obtenir le compte de chaque classe
    class_counts = df['playlist_genre'].value_counts()
    return class_counts

def preprocess_data(df, scaler=None, label_encoder=None, is_train=True, use_text=False):
    
    if 'playlist_genre' in df.columns:
        y = df['playlist_genre'] 
        df.drop(columns=['playlist_genre'], inplace=True)
    else:
        raise ValueError("La colonne 'playlist_genre' n'est pas présente dans le dataframe.")
    
    # Sélectionner les features
    if not use_text:
        X = get_numerical_features(df)
    else:
        # Utiliser toutes les colonnes sauf 'playlist_genre'
        X = df

    # Normaliser les features numériques et encoder les labels
    if is_train:
        scaler = StandardScaler()
        label_encoder = LabelEncoder()
        X_scaled = scaler.fit_transform(X)
        y_encoded = label_encoder.fit_transform(y)
    else:
        assert scaler is not None and label_encoder is not None, "scaler and label_encoder must be provided for test data"
        X_scaled = scaler.transform(X)
        y_encoded = label_encoder.transform(y)
    
    return X_scaled, y_encoded, scaler, label_encoder