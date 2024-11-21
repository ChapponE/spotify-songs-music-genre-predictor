# src/models/transformer_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import math
from src.models.base_model import BaseModel
from src.utils.helpers import get_optimizer, get_loss_function

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        self.neural_network = True
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(1)  # Shape: (max_seq_length, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_length, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(BaseModel):
    def __init__(self, input_size, output_size, num_layers=2, d_model=128, nhead=4, dim_feedforward=512, dropout=0.1,
                 learning_rate=0.001, optimizer_name='adam', loss_name='cross_entropy'):
        super(TransformerModel, self).__init__()
        self.neural_network = True
        self.input_size = input_size
        self.output_size = output_size
        self.d_model = d_model
        self.optimizer_name = optimizer_name
        self.loss_name = loss_name

        # Embedding layer to project input features to d_model dimensions
        self.input_embedding = nn.Linear(1, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_length=input_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.fc_out = nn.Linear(d_model, output_size)
        self.softmax = nn.Softmax(dim=1)

        # Loss function
        self.criterion = get_loss_function(self.loss_name)

    def forward(self, src):
        # src shape: (batch_size, input_size)
        # Reshape to (input_size, batch_size, 1)
        src = src.transpose(0, 1).unsqueeze(2)
        src = self.input_embedding(src)  # Shape: (input_size, batch_size, d_model)
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)
        # Mean over the sequence dimension
        output = output.mean(dim=0)  # Shape: (batch_size, d_model)
        output = self.fc_out(output)
        output = self.softmax(output)
        return output

    def train_model(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32, learning_rate=0.001):
        # Define optimizer
        self.optimizer = get_optimizer(self.optimizer_name, self.parameters(), learning_rate)
        self.train()
        history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            permutation = torch.randperm(X_train.size()[0])

            for i in range(0, X_train.size()[0], batch_size):
                indices = permutation[i:i+batch_size]
                X_batch, y_batch = X_train[indices], y_train[indices]

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.forward(X_batch)
                loss = self.criterion(outputs, y_batch)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

            epoch_accuracy = correct / total
            epoch_loss /= (X_train.size()[0] / batch_size)

            history['accuracy'].append(epoch_accuracy)
            history['loss'].append(epoch_loss)

            if X_val is not None and y_val is not None:
                val_accuracy, val_loss = self.evaluate(X_val, y_val)
                history['val_accuracy'].append(val_accuracy)
                history['val_loss'].append(val_loss)
            else:
                history['val_accuracy'].append(None)
                history['val_loss'].append(None)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        return history

    def evaluate(self, X_test, y_test):
        self.eval()
        correct = 0
        total = 0
        loss_total = 0.0

        with torch.no_grad():
            for i in range(0, X_test.size()[0], 32):
                X_batch = X_test[i:i+32]
                y_batch = y_test[i:i+32]

                outputs = self.forward(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss_total += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        accuracy = correct / total
        average_loss = loss_total / (X_test.size()[0] / 32)

        return accuracy, average_loss

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))
        self.eval()
