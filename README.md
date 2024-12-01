# Spotify Genre Classifier

## Introduction

Le but est de classer les playlists Spotify en différents genres en utilisant des modèles de machine learning. Ce projet exploite diverses techniques de prétraitement des données et des algorithmes d'apprentissage automatique (features numériques ou numérique et textuelles) (MLP, SVC, RF). Le rapport associé au projet est à la racine du repository 'Rapport.pdf'.

## Structure du Projet

data
├── brut_data
├── data_explorer
├── processed
└── results
    ├── mlp
    ├── mlp_full
    ├── random_forest
    └── svc
predictions
src
├── data
│   ├── data_explorer.py
│   ├── data_loader.py
│   └── __init__.py
├── models
│   ├── base_model.py
│   ├── mlp_model.py
│   ├── mlp_model_full.py
│   ├── random_forest_model.py
│   ├── svc_model.py
│   └── __init__.py
├── utils
│   ├── config.py
│   ├── cross_valid.py
│   ├── helpers.py
│   └── __init__.py
└── visualization
    ├── plots.py
    └── __init__.py
train
├── train_mlp.py
├── train_mlp_full.py
├── train_rf.py
├── train_svc.py
└── __init__.py
explore_data.py
hyperparameters_analysis.py
predict_mlp.py
predict_mlp_full.py
preprocess_data.py
preprocess_data_full.py
train.py


## Installation

1. **Cloner le dépôt :**

   ```bash
   git clone https://github.com/votre-utilisateur/spotify_genre_classifier.git
   ```

2. **Installer les dépendances :**

   ```bash
   pip install -r requirements.txt
   ```

## Prérequis

- **Python 3.7+**
- **Bibliothèques Python :**
  - Data Processing : `numpy`, `pandas`, `joblib`, `scikit-learn`
  - Deep Learning : `tensorflow`, `keras`, `torch`, `sentence-transformers`
  - Visualisation : `matplotlib`, `seaborn`

## Utilisation

### Configuration

Les configurations globales du projet sont définies dans `src/utils/config.py`. Vous pouvez ajuster les paramètres tels que les chemins des données, les hyperparamètres des modèles, le nombre de splits pour la validation croisée, etc.

### Exploration des Données

Pour explorer les données et visualiser des informations telles que la corrélation des features avec les genres de playlists :
```bash
python explore_data.py
```
Les résultats seront sauvegardés dans `data/data_explorer/`.

### Prétraitement des Données

Avant d'entraîner les modèles, il est nécessaire de prétraiter les données brutes.

**Exécution :**
Pour prétraiter les données numériques et textuelles : 
```bash
python preprocess_data.py
```
```bash
python preprocess_data_full.py
```

### Entraînement des Modèles

Le projet supporte l'entraînement de plusieurs modèles de machine learning :

- **Multi-Layer Perceptron (MLP)**
- **Support Vector Classifier (SVC)**
- **Random Forest (RF)**
- **MLP Complet**

**Exécution Générale :**
```bash
python train.py --model <nom_du_modèle>
```
Les résultats de l'entraînement, y compris les meilleurs hyperparamètres et les métriques de validation croisée, seront sauvegardés dans le répertoire `data/results/<nom_du_modèle>/`.

### Prédiction

Après avoir entraîné un modèle, vous pouvez l'utiliser pour effectuer des prédictions sur de nouvelles données.

**Prédiction avec MLP :**
```bash
python predict_mlp.py
```
```bash
python predict_mlp_full.py
```
Les prédictions seront sauvegardées dans le répertoire `predictions/` avec des noms de fichiers spécifiés.

### Analyse des Hyperparamètres

Pour analyser les hyperparamètres utilisés lors de l'entraînement des modèles :
```bash
python hyperparameters_analysis.py
```
Les analyses seront sauvegardées dans `data/results/<nom_du_modèle>/hyperparameters_analysis.csv`.
