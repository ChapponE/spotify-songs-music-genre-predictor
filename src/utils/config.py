import os

# Mode Test
TEST = False  # Set to True to process only 100 data

# Chemins vers les fichiers de données
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
TRAIN_CSV = os.path.join(DATA_DIR, 'brut_data', 'train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'brut_data', 'test.csv')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')

# Paramètres d'optimisation généraux
N_SPLITS = 4
OPTIMIZER = 'adam'  # Options : 'adam', 'sgd'
LOSS_FUNCTION = 'cross_entropy'
METRICS = ['accuracy']
EPOCHS = 150

# Hyperparamètres optimisés du modèle Transformer pour des données tabulaires de petite dimension
TRANSFORMER_PARAMS = {
    'train_param_grid': {
        'learning_rate': [0.0005, 0.001, 0.005],
        'batch_size': [16, 32],
        'epochs': [150]
    },
    'transformer_param_grid': {
        'num_layers': [1, 2],                # Nombre de couches Transformer réduites pour éviter le surapprentissage
        'd_model': [16, 32],                  # Dimensionnalité des embeddings ajustée pour petites entrées
        'nhead': [2, 4],                      # Nombre de têtes d'attention
        'dim_feedforward': [10, 20],         # Dimension du réseau feedforward
        'dropout': [0.2]                  # Taux de dropout pour régularisation
    }
}

# Hyperparamètres du modèle MLP
MLP_PARAMS = {   'train_param_grid': {    
        'learning_rate': [0.1, 0.01],
        'batch_size': [64, 256],
        'epochs': [150]},
        

    'mlp_param_grid': {
        'hidden_layers': [ [32],
            [64],
            [8, 8],
            [16, 16],
            [32, 32],          
            [16, 16, 16],
            [32, 32, 32]
            ]}
    }

# Hyperparamètres du modèle SVM
SVC_PARAM_GRID = {
    'C': [0.1, 0.2, 0.4, 0.8,1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'kernel': ['linear', 'poly', 'rbf']
}

# Hyperparamètres du modèle Random Forest
RF_PARAM_GRID = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 20],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [4, 8, 16],
    'bootstrap': [True]
}

# Hyperparamètres du modèle SVM
SVC_REG_PARAMETER = [0.1, 1, 10, 100]  # Liste des valeurs de C à tester pour SVM

