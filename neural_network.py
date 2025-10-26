import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000, add_bias=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.add_bias = add_bias
        self.weights = None
        self.errors = []
        self.converged = False
        
    def add_bias_term(self, X):
        """Add bias term to features"""
        return np.c_[np.ones(X.shape[0]), X]
    
    def initialize_weights(self, n_features):
        """Initialize weights with small random numbers"""
        if self.add_bias:
            self.weights = np.random.uniform(-0.1, 0.1, n_features + 1)
        else:
            self.weights = np.random.uniform(-0.1, 0.1, n_features)
    
    def predict(self, X):
        """Predict class labels"""
        if self.add_bias:
            X = self.add_bias_term(X)
        activation = np.dot(X, self.weights)
        return np.where(activation >= 0, 1, -1)  # Return -1/1
    
    def fit(self, X, y):
        """Train the perceptron with proper convergence checking"""
        if self.add_bias:
            X = self.add_bias_term(X)
        
        n_samples, n_features = X.shape
        self.initialize_weights(n_features - 1 if self.add_bias else n_features)
        
        for epoch in range(self.n_iterations):
            errors = 0
            for idx, x_i in enumerate(X):
                activation = np.dot(x_i, self.weights)
                prediction = 1 if activation >= 0 else -1
                if prediction != y[idx]:
                    self.weights += self.learning_rate * y[idx] * x_i
                    errors += 1
            
            self.errors.append(errors)
            
            # Check for convergence (no errors for 5 consecutive epochs)
            if errors == 0:
                if epoch > 5 and all(e == 0 for e in self.errors[-5:]):
                    self.converged = True
                    break
            else:
                self.converged = False
    
    def get_decision_boundary(self, X):
        """Get decision boundary coordinates for plotting (works with or without bias)."""
        weights = self.weights
        x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5

        if len(weights) == 3 and self.add_bias:
            w0, w1, w2 = weights
            x2_values = (-w0 - w1 * np.array([x1_min, x1_max])) / w2

        elif len(weights) == 2 and not self.add_bias:
            w1, w2 = weights
            x2_values = -(w1 / w2) * np.array([x1_min, x1_max])

        return [x1_min, x1_max], x2_values


class Adaline:
    def __init__(self, learning_rate=0.01, n_iterations=1000, mse_threshold=0.001, add_bias=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.mse_threshold = mse_threshold
        self.add_bias = add_bias
        self.weights = None
        self.mse_history = []
        self.converged = False
        
    def add_bias_term(self, X):
        """Add bias term to features"""
        return np.c_[np.ones(X.shape[0]), X]
    
    def initialize_weights(self, n_features):
        """Initialize weights with small random numbers"""
        if self.add_bias:
            self.weights = np.random.uniform(-0.1, 0.1, n_features + 1)
        else:
            self.weights = np.random.uniform(-0.1, 0.1, n_features)
    
    def predict(self, X):
        """Predict class labels"""
        if self.add_bias:
            X = self.add_bias_term(X)
        activation = np.dot(X, self.weights)
        return np.where(activation >= 0, 1, -1)  # Return -1/1
    
    def fit(self, X, y):
        """Train the Adaline using standard gradient descent (no momentum)."""
        if self.add_bias:
            X = self.add_bias_term(X)

        n_samples, n_features = X.shape
        self.initialize_weights(n_features - 1 if self.add_bias else n_features)

        for iteration in range(self.n_iterations):
            net_input = np.dot(X, self.weights)
            errors = y - net_input
            gradient = -2 * X.T.dot(errors) / n_samples
            self.weights -= self.learning_rate * gradient
            mse = np.mean(errors ** 2)
            self.mse_history.append(mse)

            if mse <= self.mse_threshold:
                self.converged = True
                break
            elif iteration > 10 and abs(self.mse_history[-1] - self.mse_history[-2]) < 1e-8:
                self.converged = True
                break

    
    def get_decision_boundary(self, X):
        """Get decision boundary coordinates for plotting (works with or without bias)."""
        weights = self.weights
        x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5

        if len(weights) == 3 and self.add_bias:
            w0, w1, w2 = weights
            x2_values = (-w0 - w1 * np.array([x1_min, x1_max])) / w2

        elif len(weights) == 2 and not self.add_bias:
            w1, w2 = weights
            x2_values = -(w1 / w2) * np.array([x1_min, x1_max])

        return [x1_min, x1_max], x2_values

class NeuralNetworkManager:
    def __init__(self):
        self.perceptron = None
        self.adaline = None
        self.current_model = None
        
    def create_confusion_matrix(self, y_true, y_pred):
        """Create confusion matrix without using sklearn"""
        # Convert from -1/1 to 0/1 for confusion matrix
        y_true_binary = np.where(y_true == -1, 0, 1)
        y_pred_binary = np.where(y_pred == -1, 0, 1)
        
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        confusion_matrix = np.array([[tn, fp], [fn, tp]])
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        return confusion_matrix, accuracy
    
    def plot_decision_boundary(self, X, y, model, feature_names, class_names):
        """Plot decision boundary and data points"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: Decision boundary
        # Separate classes
        class0_mask = (y == -1)
        class1_mask = (y == 1)
        
        # Plot all data points
        ax1.scatter(X[class0_mask, 0], X[class0_mask, 1], 
                   color='red', marker='o', label=f'{class_names[0]}', alpha=0.7, s=60)
        ax1.scatter(X[class1_mask, 0], X[class1_mask, 1], 
                   color='blue', marker='s', label=f'{class_names[1]}', alpha=0.7, s=60)
        
        # Plot decision boundary
        x_boundary, y_boundary = model.get_decision_boundary(X)
        if x_boundary is not None and y_boundary is not None:
            ax1.plot(x_boundary, y_boundary, color='green', linewidth=3, 
                    label='Decision Boundary', linestyle='--')
        
        ax1.set_xlabel(feature_names[0])
        ax1.set_ylabel(feature_names[1])
        ax1.set_title(f'Decision Boundary: {feature_names[0]} vs {feature_names[1]}\n{class_names[0]} vs {class_names[1]}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training history
        if isinstance(model, Perceptron) and model.errors:
            ax2.plot(range(1, len(model.errors) + 1), model.errors, marker='o', color='purple', linewidth=2)
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Number of Errors')
            ax2.set_title('Perceptron Training - Errors per Epoch')
            ax2.grid(True, alpha=0.3)
            
            # Add convergence info
            final_errors = model.errors[-1]
            convergence_status = "Converged" if model.converged else "Not Converged"
            ax2.annotate(f'Final Errors: {final_errors}\n{convergence_status}', 
                        xy=(len(model.errors), final_errors),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
        elif isinstance(model, Adaline) and model.mse_history:
            ax2.plot(range(1, len(model.mse_history) + 1), model.mse_history, 
                    marker='o', color='orange', linewidth=2)
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('MSE')
            ax2.set_title('Adaline Training - MSE per Epoch')
            ax2.grid(True, alpha=0.3)
            
            # Add convergence info
            final_mse = model.mse_history[-1]
            convergence_status = "Converged" if model.converged else "Not Converged"
            ax2.annotate(f'Final MSE: {final_mse:.6f}\n{convergence_status}', 
                        xy=(len(model.mse_history), final_mse),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        
        return fig