# src/models/random_forest_model.py
from sklearn.ensemble import RandomForestClassifier
from src.models.base_model import BaseModel
import joblib

class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        super(RandomForestModel, self).__init__()
        self.neural_network = False
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

    def train_model(self, X_train, y_train, X_val=None, y_val=None, **train_params):
        self.model.fit(X_train, y_train)
        train_accuracy = self.model.score(X_train, y_train)
        val_accuracy = self.model.score(X_val, y_val) if X_val is not None and y_val is not None else None

        history = {
            'accuracy': [train_accuracy],
            'val_accuracy': [val_accuracy],
            'loss': [None],
            'val_loss': [None]
        }
        print(f"Training Accuracy: {train_accuracy:.4f}")
        if val_accuracy is not None:
            print(f"Validation Accuracy: {val_accuracy:.4f}")

        return history

    def evaluate(self, X_test, y_test):
        test_accuracy = self.model.score(X_test, y_test)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        return test_accuracy

    def save(self, filepath):
        joblib.dump(self.model, filepath)

    def load(self, filepath):
        self.model = joblib.load(filepath)