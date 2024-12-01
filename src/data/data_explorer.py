import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from src.utils.config import DATA_DIR
from src.data.data_loader import (
    get_categorical_features,
    get_numerical_features,
    load_data)

def explore_data():
    # Charger les données
    df, _ = load_data()
    
    if df['playlist_genre'].isnull().any():
        raise ValueError("La colonne 'playlist_genre' contient des valeurs manquantes.")
    
    # Créer le répertoire data_explorer s'il n'existe pas
    data_explorer_dir = os.path.join(DATA_DIR, 'data_explorer')
    os.makedirs(data_explorer_dir, exist_ok=True)
    
    # Obtenir les features numériques et catégorielles
    categorical_features = get_categorical_features(df)
    numerical_features = get_numerical_features(df)
    
    # Tableau avec le nombre d'éléments uniques par feature catégorielle
    unique_counts = categorical_features.nunique()
    unique_counts_df = unique_counts.to_frame(name='Unique Values').reset_index()
    unique_counts_df.rename(columns={'index': 'Feature'}, inplace=True)
    unique_counts_df.to_csv(os.path.join(data_explorer_dir, 'categorical_unique_counts.csv'), index=False)
    
    # Définir la variable cible
    y = df['playlist_genre'].astype('category').cat.codes
    
    # Calculer les corrélations entre les features numériques et la variable cible
    correlation_list = []
    for col in numerical_features.columns:
        corr = numerical_features[col].corr(y, method='pearson')
        correlation_list.append({'Feature': col, 'Correlation with Genre': corr})

    # Créer un DataFrame combiné
    correlation_combined_df = pd.DataFrame(correlation_list)
    
    # Sauvegarder le tableau combiné
    correlation_combined_df.to_csv(os.path.join(data_explorer_dir, 'correlation_combined_features.csv'), index=False)
    
    # Visualiser les corrélations combinées
    plt.figure(figsize=(10, max(6, len(correlation_combined_df) * 0.4)))  # Ajuster la taille en fonction du nombre de features
    sns.barplot(
        x='Correlation with Genre',
        y='Feature',
        data=correlation_combined_df,
        palette='coolwarm',
        orient='h'
    )
    plt.title("Corrélation des Features avec 'playlist_genre'")
    plt.xlabel("Coefficient de Corrélation de Pearson")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(data_explorer_dir, 'correlation_combined_features.png'))
    plt.close()

    # Optionnel : Visualiser le tableau unique des counts
    print("\nNombre d'éléments uniques par feature catégorielle :")
    print(unique_counts_df)
