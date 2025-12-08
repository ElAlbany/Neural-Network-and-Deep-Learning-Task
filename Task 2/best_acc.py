import numpy as np
from data_processor import DataProcessor
from neural_network import BackpropagationNN
from sklearn.metrics import accuracy_score

# Load data
processor = DataProcessor()
processor.load_data("Task 2\\penguins.csv")
X_train, X_test, Y_train, y_test = processor.prepare_data()
y_train_true = np.argmax(Y_train, axis=1)

# Define hyperparameter search space
learning_rates = [0.0001, 0.001, 0.005, 0.01, 0.05]
epoch_list = [200, 500, 800, 1000, 1500]
neuron_configs = [
    [3],
    [4],
    [5],
    [8],
    [10],
    [5, 3],
    [8, 4],
    [10, 5]
]

activations = ["sigmoid", "tanh"]
bias_options = [True, False]

results = []

print("Starting grid search...")

for activation in activations:
    for bias in bias_options:
        best_result = None

        for lr in learning_rates:
            for epochs in epoch_list:
                for neurons in neuron_configs:

                    model = BackpropagationNN(
                        n_features=5,
                        n_hidden_layers=len(neurons),
                        n_neurons_per_hidden=neurons,
                        n_classes=3,
                        learning_rate=lr,
                        n_epochs=epochs,
                        bias=bias,
                        activation_function=activation
                    )

                    # Train
                    for epoch in range(epochs):
                        for x, y in zip(X_train, Y_train):
                            model.forward(x)
                            model.backward(y)

                    # Compute accuracies
                    y_train_pred = np.array([model.predict(x) for x in X_train])
                    train_acc = accuracy_score(y_train_true, y_train_pred)

                    y_test_pred = np.array([model.predict(x) for x in X_test])
                    test_acc = accuracy_score(y_test, y_test_pred)

                    # Track best
                    if (best_result is None) or (test_acc > best_result["test_acc"]):
                        best_result = {
                            "activation": activation,
                            "bias": bias,
                            "lr": lr,
                            "epochs": epochs,
                            "neurons": neurons,
                            "train_acc": train_acc,
                            "test_acc": test_acc
                        }

                print(f"✓ Finished: {activation}, bias={bias}, lr={lr}")

        results.append(best_result)

print("\n===== BEST RESULTS =====")
for r in results:
    print(
        f"{r['activation'].upper()} | Bias={r['bias']} | LR={r['lr']} | "
        f"Epochs={r['epochs']} | Neurons={r['neurons']} | "
        f"TrainAcc={r['train_acc']*100:.2f}% | TestAcc={r['test_acc']*100:.2f}%"
    )
