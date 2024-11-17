from src.models.base_model import BaseModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class MLPModel(BaseModel):
    def __init__(self, hidden_layers, input_size, output_size, learning_rate=0.001):
        self.hidden_layers = hidden_layers
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.model = None
        self.history = None

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32, learning_rate=0.001):
        self.model = Sequential()
        self.model.add(Dense(self.hidden_layers[0], activation='relu', input_shape=(self.input_size,)))
        for units in self.hidden_layers[1:]:
            self.model.add(Dense(units, activation='relu'))
        self.model.add(Dense(self.output_size, activation='softmax'))

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        return self.history

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)