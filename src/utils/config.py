import os

# Mode Test
TEST = False  # Set to True to process only 100 data

# Chemins vers les fichiers de données
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
PROCESSED_DIR_FULL = os.path.join(DATA_DIR, 'processed_full')
TRAIN_CSV = os.path.join(DATA_DIR, 'brut_data', 'train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'brut_data', 'test.csv')
TRAIN_FULL_CSV = os.path.join(DATA_DIR, 'brut_data', 'train_full.csv')
TEST_FULL_CSV = os.path.join(DATA_DIR, 'brut_data', 'test_full.csv')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')

# Paramètres d'optimisation généraux
N_SPLITS = 4
OPTIMIZER = 'adam'  # Options : 'adam', 'sgd'
LOSS_FUNCTION = 'cross_entropy'
METRICS = ['accuracy']
EPOCHS = 150

# Hyperparamètres du modèle MLP
MLP_PARAMS = {   'train_param_grid': {    
        'learning_rate': [0.05, 0.005],
        'batch_size': [256, 512],
        'epochs': [150]},
        

    'mlp_param_grid': {
        'hidden_layers': [ [2], [4], [8], [16], [32], [64], [2, 2], [4, 4] ]}
    }

# Hyperparamètres du modèle MLP complet
MLP_FULL_PARAMS = {
    'train_param_grid': {
        'learning_rate': [0.05, 0.005],
        'batch_size': [256, 512],
        'epochs': [150]},
    'mlp_full_param_grid': {
        'hidden_layers': [ [2], [4], [8], [16], [32], [64], [2, 2], [4, 4] ]}
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

