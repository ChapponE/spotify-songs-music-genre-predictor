# src/models/mlp_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from src.models.base_model import BaseModel
from src.utils.helpers import get_optimizer, get_loss_function

class MLPModel(BaseModel):
    def __init__(self, hidden_layers, input_size, output_size, 
                 learning_rate=0.001, optimizer_name='adam', loss_name='cross_entropy'):
        super(MLPModel, self).__init__()
        self.neural_network = True
        self.hidden_layers = hidden_layers
        self.input_size = input_size
        self.output_size = output_size
        self.optimizer_name = optimizer_name
        self.loss_name = loss_name

        # Définir l'architecture du réseau avec Batch Normalization
        layers = []
        in_features = input_size
        for units in hidden_layers:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.BatchNorm1d(units))
            layers.append(nn.ReLU())
            in_features = units
        layers.append(nn.Linear(in_features, output_size))
        layers.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*layers)

        # Définir le loss function
        self.criterion = get_loss_function(self.loss_name)

    def forward(self, X):
        return self.model(X)

    def train_model(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32, learning_rate=0.001):
        # Définir l'optimizer
        self.optimizer = get_optimizer(self.optimizer_name, self.parameters(), learning_rate)
        self.train()  # Mettre le modèle en mode entraînement
        history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                # Convertir en tenseurs torch
                X_batch = torch.tensor(X_batch, dtype=torch.float32)
                y_batch = torch.tensor(y_batch, dtype=torch.long)

                # Zero les gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.forward(X_batch)
                loss = self.criterion(outputs, y_batch)

                # Backward pass et optimisation
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                # Calculer la précision
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

            epoch_accuracy = correct / total
            epoch_loss /= len(X_train) / batch_size

            history['accuracy'].append(epoch_accuracy)
            history['loss'].append(epoch_loss)

            if X_val is not None and y_val is not None:
                val_accuracy, val_loss = self.evaluate(X_val, y_val)
                history['val_accuracy'].append(val_accuracy)
                history['val_loss'].append(val_loss)
            else:
                history['val_accuracy'].append(None)
                history['val_loss'].append(None)

        return history

    def evaluate(self, X_test, y_test):
        self.eval()  # Mettre le modèle en mode évaluation
        correct = 0
        total = 0
        loss_total = 0.0

        with torch.no_grad():
            for i in range(0, len(X_test), 32):
                X_batch = X_test[i:i+32]
                y_batch = y_test[i:i+32]

                X_batch = torch.tensor(X_batch, dtype=torch.float32)
                y_batch = torch.tensor(y_batch, dtype=torch.long)

                outputs = self.forward(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss_total += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        accuracy = correct / total
        average_loss = loss_total / (len(X_test) / 32)

        return accuracy, average_loss

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))
        self.eval()