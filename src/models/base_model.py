# Classe de base pour les modèles
class BaseModel:
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Entraîne le modèle avec les données fournies.

        Paramètres:
        - X_train: Features d'entraînement
        - y_train: Labels d'entraînement
        - X_val: Features de validation (optionnel)
        - y_val: Labels de validation (optionnel)
        """
        raise NotImplementedError("La méthode train doit être implémentée par les sous-classes.")

    def evaluate(self, X_test, y_test):
        """
        Évalue le modèle avec les données de test.

        Paramètres:
        - X_test: Features de test
        - y_test: Labels de test
        """
        raise NotImplementedError("La méthode evaluate doit être implémentée par les sous-classes.")

    def predict(self, X):
        """
        Prédit les labels pour les features fournies.

        Paramètres:
        - X: Features d'entrée

        Retourne:
        - Prédictions du modèle
        """
        raise NotImplementedError("La méthode predict doit être implémentée par les sous-classes.")

    def save(self, filepath):
        """
        Sauvegarde le modèle sur le disque.

        Paramètres:
        - filepath: Chemin du fichier pour sauvegarder le modèle
        """
        raise NotImplementedError("La méthode save doit être implémentée par les sous-classes.")

    def load(self, filepath):
        """
        Charge le modèle depuis le disque.

        Paramètres:
        - filepath: Chemin du fichier pour charger le modèle
        """
        raise NotImplementedError("La méthode load doit être implémentée par les sous-classes.")
