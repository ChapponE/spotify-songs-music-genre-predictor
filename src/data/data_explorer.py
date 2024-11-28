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
    
    # Vérifier que 'playlist_subgenre' est une seule colonne catégorielle
    if 'playlist_subgenre' not in df.columns:
        raise ValueError("La colonne 'playlist_subgenre' n'existe pas dans les données.")
    
    # Vérifier s'il y a des valeurs manquantes dans 'playlist_subgenre' ou 'playlist_genre'
    if df['playlist_subgenre'].isnull().any():
        raise ValueError("La colonne 'playlist_subgenre' contient des valeurs manquantes.")
    if df['playlist_genre'].isnull().any():
        raise ValueError("La colonne 'playlist_genre' contient des valeurs manquantes.")
    
    # Créer le répertoire data_explorer s'il n'existe pas
    data_explorer_dir = os.path.join(DATA_DIR, 'data_explorer')
    os.makedirs(data_explorer_dir, exist_ok=True)
    
    # Obtenir les features numériques et catégorielles
    X_numeric = get_numerical_features(df)
    X_categorical = get_categorical_features(df)
    
    # Tableau avec le nombre d'éléments uniques par feature catégorielle
    unique_counts = X_categorical.nunique()
    unique_counts_df = unique_counts.to_frame(name='Unique Values').reset_index()
    unique_counts_df.rename(columns={'index': 'Feature'}, inplace=True)
    unique_counts_df.to_csv(os.path.join(data_explorer_dir, 'categorical_unique_counts.csv'), index=False)
    
    # Vérifier que 'playlist_subgenre' est bien présent dans les features catégorielles
    if 'playlist_subgenre' not in X_categorical.columns:
        raise ValueError("La colonne 'playlist_subgenre' n'est pas présente dans les features catégorielles.")
    
    # Encoder la feature catégorielle 'playlist_subgenre' avec Label Encoding
    label_encoder = LabelEncoder()
    X_categorical_encoded = label_encoder.fit_transform(X_categorical['playlist_subgenre'])
    X_categorical_encoded_df = pd.DataFrame(
        X_categorical_encoded,
        columns=['playlist_subgenre_encoded']
    )
    
    # Combiner les features numériques et encodées
    X = pd.concat([X_numeric.reset_index(drop=True), X_categorical_encoded_df.reset_index(drop=True)], axis=1)
    
    # Définir la variable cible
    y = df['playlist_genre'].astype('category').cat.codes
    
    # Liste pour stocker les corrélations
    correlation_list = []
    
    # 1. Corrélation pour la feature catégorielle encodée
    feature_cat = 'playlist_subgenre_encoded'
    correlation_cat = X[feature_cat].corr(y)
    correlation_list.append({
        'Feature': 'playlist_subgenre',
        'Correlation with Genre': correlation_cat
    })
    
    # 2. Corrélations pour les features numériques
    correlation_numeric = X_numeric.corrwith(y)
    for feature, corr in correlation_numeric.items():
        correlation_list.append({
            'Feature': feature,
            'Correlation with Genre': corr
        })
    
    # Créer un DataFrame combiné
    correlation_combined_df = pd.DataFrame(correlation_list)
    
    # Trier par valeur absolue de corrélation décroissante
    correlation_combined_df['Abs_Correlation'] = correlation_combined_df['Correlation with Genre'].abs()
    correlation_combined_df.sort_values(by='Abs_Correlation', ascending=False, inplace=True)
    correlation_combined_df.drop(columns='Abs_Correlation', inplace=True)
    
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
    
    # Optionnel : Afficher le tableau combiné dans la console
    print("\nCorrélations des Features avec 'playlist_genre' (classées par valeur absolue) :")
    print(correlation_combined_df)

    # Optionnel : Visualiser le tableau unique des counts
    print("\nNombre d'éléments uniques par feature catégorielle :")
    print(unique_counts_df)
    