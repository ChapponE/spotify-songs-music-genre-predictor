import os
import pandas as pd
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.utils.config import TRAIN_CSV, TEST_CSV, TRAIN_FULL_CSV, TEST_FULL_CSV, RESULTS_DIR, PROCESSED_DIR, TEST

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

def load_data_preprocessed_full():
    """
    Charge les données prétraitées et les objets de prétraitement depuis un fichier unique dans le répertoire processed.
    
    Retourne:
        X_train_scaled (array): Features d'entraînement prétraités
        y_train_encoded (array): Labels d'entraînement encodés
        scaler (StandardScaler): Objet de scalage
        label_encoder (LabelEncoder): Objet d'encodage des labels
    """
    # Déterminer le nom du fichier en fonction de TEST
    filename = 'preprocessed_data_test_full.pkl' if TEST else 'preprocessed_data_full.pkl'
    data_path = os.path.join(PROCESSED_DIR, filename)
    
    # Charger les données
    data = joblib.load(data_path)
    X_train_processed = data['X_train_processed']
    y_train_encoded = data['y_train_encoded']
    scaler = data['scaler']
    label_encoder = data['label_encoder']
    text_encoder = data['text_encoder']
    
    return X_train_processed, y_train_encoded, scaler, label_encoder, text_encoder

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
    exclude_cols = ['playlist_genre']
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    X_categorical = df[categorical_cols]
    return X_categorical     

def get_class_counts(df):
    # Obtenir le compte de chaque classe
    class_counts = df['playlist_genre'].value_counts()
    return class_counts

def preprocess_data(df, scaler=None, label_encoder=None, is_train=True):
    
    if 'playlist_genre' in df.columns:
        y = df['playlist_genre'] 
        df.drop(columns=['playlist_genre'], inplace=True)
    else:
        raise ValueError("La colonne 'playlist_genre' n'est pas présente dans le dataframe.")
    
    # Sélectionner les features
    X = get_numerical_features(df)

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

def preprocess_data_full(df, scaler=None, label_encoder=None, text_encoder=None, is_train=True):
    """
    Pré-traite les données en excluant la colonne 'playlist_subgenre', en manipulant les caractéristiques numériques,
    en encodant les caractéristiques catégorielles avec SentenceTransformer, et en normalisant les données.
    
    Args:
        df (pd.DataFrame): DataFrame d'entrée
        scaler (StandardScaler): Scaler pour les caractéristiques numériques
        label_encoder (LabelEncoder): Encodeur pour les labels cibles
        text_encoder (SentenceTransformer): Encodeur textuel utilisé
        is_train (bool): Indique si les données sont d'entraînement

    Retourne:
        X_processed (np.array): Caractéristiques pré-traitées
        y_encoded (np.array): Labels cibles encodés
        scaler (StandardScaler): Scaler utilisé
        label_encoder (LabelEncoder): LabelEncoder utilisé
        text_encoder (SentenceTransformer): Encodeur textuel utilisé
    """

    # Extraire la variable cible
    if 'playlist_genre' in df.columns:
        y = df['playlist_genre']
        df = df.drop(columns=['playlist_genre'])
    elif 'playlist_subgenre' in df.columns:
        y = df['playlist_subgenre']
        df = df.drop(columns=['playlist_subgenre'])
    else:
        raise ValueError("La colonne 'playlist_genre' n'est pas présente dans le dataframe.")

    # Obtenir les caractéristiques numériques
    X_numeric = get_numerical_features(df)

    # Obtenir les caractéristiques catégorielles
    X_categorical = get_categorical_features(df)

    # **Exclure explicitement la colonne 'playlist_subgenre'**
    # (Cela est déjà géré dans get_numerical_features et get_categorical_features)

    # Encodage des autres caractéristiques textuelles avec SentenceTransformer
    if X_categorical.shape[1] > 0:
        # Combiner les colonnes textuelles en une seule chaîne par ligne
        text_data = X_categorical.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        if is_train:
            text_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            text_embeddings = text_encoder.encode(text_data.tolist())
        else:
            assert text_encoder is not None, "text_encoder doit être fourni pour les données de test"
            text_embeddings = text_encoder.encode(text_data.tolist())
    else:
        text_embeddings = None

    # Mise à l'échelle des caractéristiques numériques
    if is_train:
        scaler = StandardScaler()
        X_numeric_scaled = scaler.fit_transform(X_numeric)
    else:
        assert scaler is not None, "scaler doit être fourni pour les données de test"
        X_numeric_scaled = scaler.transform(X_numeric)

    # Préparation de la liste des tableaux de caractéristiques
    feature_arrays = []

    # Ajouter les caractéristiques numériques
    feature_arrays.append(X_numeric_scaled)

    # Ajouter les embeddings textuels si disponibles
    if text_embeddings is not None:
        feature_arrays.append(text_embeddings)

    # Concaténer toutes les caractéristiques
    X_processed = np.hstack(feature_arrays)

    # Encodage des labels cibles
    if is_train:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
    else:
        assert label_encoder is not None, "label_encoder doit être fourni pour les données de test"
        y_encoded = label_encoder.transform(y)

    return X_processed, y_encoded, scaler, label_encoder, text_encoder