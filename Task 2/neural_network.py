import numpy as np


class BackpropagationNN:
    def __init__(
        self,
        n_features,
        n_hidden_layers,
        n_neurons_per_hidden,
        n_classes,
        learning_rate=0.01,
        n_epochs=1000,
        bias=False,
        activation_function="sigmoid",
    ):
        self.n_features = n_features
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons_per_hidden = n_neurons_per_hidden
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.bias = bias
        self.activation_function = activation_function

        layer_sizes = [n_features] + n_neurons_per_hidden + [n_classes]

        self.weights = []
        for i in range(len(layer_sizes) - 1):
            w = (
                np.random.randn(layer_sizes[i + 1], layer_sizes[i] + (1 if bias else 0))
                * 0.1
            )
            self.weights.append(w)

    def _activate(self, x):
        if self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == "tanh":
            return np.tanh(x)

    def _activate_derivative(self, k):
        if self.activation_function == "sigmoid":
            return k * (1 - k)
        elif self.activation_function == "tanh":
            return (1 - k) * (1 + k)

    def forward(self, x):
        f = [x]
        inputs = x
        for w in self.weights:
            if self.bias:
                inputs = np.append(inputs, 1)
            z = np.dot(w, inputs)
            a = self._activate(z)
            f.append(a)
            inputs = a
        self.f = f
        return f[-1]

    def backward(self, y_true):
        deltas = []

        delta = (y_true - self.f[-1]) * self._activate_derivative(self.f[-1])
        deltas.append(delta)

        for i in reversed(range(len(self.weights) - 1)):
            w_next = self.weights[i + 1]

            w_no_bias = w_next[:, :-1] if self.bias else w_next

            delta = np.dot(w_no_bias.T, deltas[-1]) * self._activate_derivative(
                self.f[i + 1]
            )
            deltas.append(delta)

        deltas.reverse()

        for i in range(len(self.weights)):
            a = self.f[i]
            if self.bias:
                a = np.append(a, 1)
            self.weights[i] += self.learning_rate * np.outer(deltas[i], a)

    def fit(self, X, Y):
        for epoch in range(self.n_epochs):
            total_error = 0
            for x, y in zip(X, Y):
                output = self.forward(x)
                self.backward(y)
                total_error += np.sum((y - output) ** 2)
            Eav = total_error / len(X)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Average Error: {Eav:.6f}")

    def predict(self, x):
        output = self.forward(x)
        return np.argmax(output)
