import os

# Mode Test
TEST = True  # Set to True to process only 100 data

# Chemins vers les fichiers de données
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
TRAIN_CSV = os.path.join(DATA_DIR, 'raw', 'train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'raw', 'test.csv')
OUTPUT_DIR = os.path.join(DATA_DIR, 'results')

# Paramètres d'optimisation généraux
N_SPLITS = 5
OPTIMIZER = 'adam'  # Options : 'adam', 'sgd'
LOSS_FUNCTION = 'cross_entropy'
METRICS = ['accuracy']
EPOCHS = 10

# Paramètres de traitement de texte (pour la partie 3)
EMBEDDING_DIM = 768  # Dimension des embeddings de phrases

# Hyperparamètres du modèle MLP
MLP_PARAMS = {   'train_param_grid': {    
        'learning_rate': [0.01],
        'epochs': [70],
        'batch_size': [32]},

    'mlp_param_grid': {
        'hidden_layers': [
            [64, 32],
            [128, 64]
        ]}
    }

# Hyperparamètres du modèle SVM
SVC_PARAM_GRID = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear']  # Vous pouvez ajouter d'autres kernels si nécessaire
}

# Hyperparamètres du modèle SVM
SVC_REG_PARAMETER = [0.1, 1, 10, 100]  # Liste des valeurs de C à tester pour SVM

