# src/models/random_forest_model.py

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42, **kwargs):
        self.neural_network = False  # Indique que ce n'est pas un modèle neuronal
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, **train_params):
        self.model.fit(X_train, y_train)
        train_accuracy = self.model.score(X_train, y_train)
        val_accuracy = self.model.score(X_val, y_val) if X_val is not None and y_val is not None else None

        history = {
            'accuracy': [train_accuracy],
            'val_accuracy': [val_accuracy] if val_accuracy is not None else [None],
            'loss': [None],                
            'val_loss': [None]             
        }
        print(f"Training Accuracy: {train_accuracy:.4f}")
        if val_accuracy is not None:
            print(f"Validation Accuracy: {val_accuracy:.4f}")

        return history
    
    def evaluate(self, X_val, y_val):
        accuracy = self.model.score(X_val, y_val)
        return {
            'accuracy': accuracy
        }
    
    def save(self, filepath):
        joblib.dump(self.model, filepath)
    
    def load(self, filepath):
        self.model = joblib.load(filepath)