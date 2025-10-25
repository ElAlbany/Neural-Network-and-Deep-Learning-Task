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
        return np.where(activation >= 0, 1, 0)
    
    def fit(self, X, y):
        """Train the perceptron"""
        if self.add_bias:
            X = self.add_bias_term(X)
        
        n_samples, n_features = X.shape
        self.initialize_weights(n_features - 1 if self.add_bias else n_features)
        
        for epoch in range(self.n_iterations):
            errors = 0
            for idx, x_i in enumerate(X):
                activation = np.dot(x_i, self.weights)
                prediction = 1 if activation >= 0 else 0
                update = self.learning_rate * (y[idx] - prediction)
                self.weights += update * x_i
                errors += int(update != 0.0)
            
            self.errors.append(errors)
            if errors == 0:
                break
    
    def get_decision_boundary(self, X):
        """Get decision boundary coordinates for plotting"""
        if self.add_bias and self.weights is not None and len(self.weights) == 3:
            w0, w1, w2 = self.weights
            if abs(w2) > 1e-10:  # Avoid division by zero with tolerance
                x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                x2_values = (-w0 - w1 * np.array([x1_min, x1_max])) / w2
                return [x1_min, x1_max], x2_values
        return None, None

class Adaline:
    def __init__(self, learning_rate=0.01, n_iterations=1000, mse_threshold=0.001, add_bias=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.mse_threshold = mse_threshold
        self.add_bias = add_bias
        self.weights = None
        self.mse_history = []
        
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
        return np.where(activation >= 0, 1, 0)
    
    def fit(self, X, y):
        """Train the Adaline using MSE with proper gradient descent"""
        if self.add_bias:
            X = self.add_bias_term(X)
        
        n_samples, n_features = X.shape
        self.initialize_weights(n_features - 1 if self.add_bias else n_features)
        
        for iteration in range(self.n_iterations):
            # Calculate net input and errors
            net_input = np.dot(X, self.weights)
            errors = y - net_input
            
            # Update weights using gradient descent
            # Gradient = -X.T.dot(errors) / n_samples
            gradient = -X.T.dot(errors) / n_samples
            self.weights -= self.learning_rate * gradient
            
            # Calculate MSE
            mse = np.mean(errors ** 2)
            self.mse_history.append(mse)
            
            # Check convergence
            if mse <= self.mse_threshold:
                break
    
    def get_decision_boundary(self, X):
        """Get decision boundary coordinates for plotting"""
        if self.add_bias and self.weights is not None and len(self.weights) == 3:
            w0, w1, w2 = self.weights
            if abs(w2) > 1e-10:  # Avoid division by zero with tolerance
                x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                x2_values = (-w0 - w1 * np.array([x1_min, x1_max])) / w2
                return [x1_min, x1_max], x2_values
        return None, None

class NeuralNetworkManager:
    def __init__(self):
        self.perceptron = None
        self.adaline = None
        self.current_model = None
        
    def create_confusion_matrix(self, y_true, y_pred):
        """Create confusion matrix without using sklearn"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        confusion_matrix = np.array([[tn, fp], [fn, tp]])
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        return confusion_matrix, accuracy
    
    def plot_decision_boundary(self, X, y, model, feature_names, class_names):
        """Plot decision boundary and data points"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: Decision boundary with training and test data
        # Separate classes
        class0_mask = (y == 0)
        class1_mask = (y == 1)
        
        # Training data (first 60 points as training)
        train_size = len(X) // 3 * 2  # 2/3 for training
        ax1.scatter(X[class0_mask][:train_size//2, 0], X[class0_mask][:train_size//2, 1], 
                   color='red', marker='o', label=f'{class_names[0]} (Train)', alpha=0.7, s=60)
        ax1.scatter(X[class1_mask][:train_size//2, 0], X[class1_mask][:train_size//2, 1], 
                   color='blue', marker='s', label=f'{class_names[1]} (Train)', alpha=0.7, s=60)
        
        # Test data (remaining points)
        ax1.scatter(X[class0_mask][train_size//2:, 0], X[class0_mask][train_size//2:, 1], 
                   color='red', marker='o', facecolors='none', edgecolors='red',
                   label=f'{class_names[0]} (Test)', s=80, linewidth=2)
        ax1.scatter(X[class1_mask][train_size//2:, 0], X[class1_mask][train_size//2:, 1], 
                   color='blue', marker='s', facecolors='none', edgecolors='blue',
                   label=f'{class_names[1]} (Test)', s=80, linewidth=2)
        
        # Plot decision boundary
        if model.add_bias:
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
            
            # Add final error count annotation
            final_errors = model.errors[-1]
            ax2.annotate(f'Final Errors: {final_errors}', 
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
            
            # Add final MSE annotation
            final_mse = model.mse_history[-1]
            ax2.annotate(f'Final MSE: {final_mse:.6f}', 
                        xy=(len(model.mse_history), final_mse),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        
        return fig