import torch.nn as nn
import torch

# Classe de base pour les modèles utilisant PyTorch
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, X):
        """
        Forward pass du modèle.

        Paramètres:
        - X: Features d'entrée

        Retourne:
        - Prédictions du modèle
        """
        raise NotImplementedError("La méthode forward doit être implémentée par les sous-classes.")

    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """
        Entraîne le modèle avec les données fournies.

        Paramètres:
        - X_train: Features d'entraînement
        - y_train: Labels d'entraînement
        - X_val: Features de validation (optionnel)
        - y_val: Labels de validation (optionnel)
        """
        raise NotImplementedError("La méthode train_model doit être implémentée par les sous-classes.")

    def evaluate(self, X_test, y_test):
        """
        Évalue le modèle avec les données de test.

        Paramètres:
        - X_test: Features de test
        - y_test: Labels de test
        """
        raise NotImplementedError("La méthode evaluate doit être implémentée par les sous-classes.")

    def save(self, filepath):
        """
        Sauvegarde le modèle sur le disque.

        Paramètres:
        - filepath: Chemin du fichier pour sauvegarder le modèle
        """
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        """
        Charge le modèle depuis le disque.

        Paramètres:
        - filepath: Chemin du fichier pour charger le modèle
        """
        self.load_state_dict(torch.load(filepath))