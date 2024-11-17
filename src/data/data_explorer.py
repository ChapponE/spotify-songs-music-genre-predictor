import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.data_loader import load_data, get_numerical_features, get_class_counts

def explore_data():
    # Charger les données
    df, _ = load_data()

    # Obtenir les features numériques
    X_numeric = get_numerical_features(df)

    # Résumé statistique
    print("Résumé Statistique :")
    print(X_numeric.describe())

    # Matrice de corrélation
    plt.figure(figsize=(12, 10))
    sns.heatmap(X_numeric.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Matrice de Corrélation des Features')
    plt.show()

    # Nombre d'éléments par classe
    class_counts = get_class_counts(df)
    print("\nNombre d'éléments par classe :")
    print(class_counts)

    # Diagramme en barres du nombre d'éléments par classe
    plt.figure(figsize=(8,6))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
    plt.title("Nombre d'éléments par classe")
    plt.xlabel('Genre')
    plt.ylabel('Nombre')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Histogrammes des features numériques
    X_numeric.hist(bins=20, figsize=(15,10))
    plt.tight_layout()
    plt.show()


