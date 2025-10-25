import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from data_processor import DataProcessor
from neural_network import Perceptron, Adaline, NeuralNetworkManager

class PenguinsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Penguins Classification - Neural Networks")
        self.root.geometry("1600x1000")
        self.root.minsize(1400, 800)
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.nn_manager = NeuralNetworkManager()
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None
        self.current_features = None
        self.current_classes = None
        self.current_figure = None
        self.canvas = None
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the main GUI layout with left-right split"""
        # Create main paned window for resizable left-right split
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left frame for controls
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Right frame for results and visualization
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)
        
        # Configure left frame (controls)
        self.setup_left_frame(left_frame)
        
        # Configure right frame (results and visualization)
        self.setup_right_frame(right_frame)
        
    def setup_left_frame(self, parent):
        """Setup the left frame with all controls"""
        # Create a canvas with scrollbar for left frame
        left_canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=left_canvas.yview)
        scrollable_frame = ttk.Frame(left_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        )
        
        left_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        left_canvas.configure(yscrollcommand=scrollbar.set)
        
        left_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Title
        title_label = ttk.Label(scrollable_frame, text="Penguins Classification Neural Network", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Data loading section
        self.setup_data_section(scrollable_frame)
        
        # Algorithm selection
        self.setup_algorithm_section(scrollable_frame)
        
        # Feature and class selection
        self.setup_selection_section(scrollable_frame)
        
        # Parameters section
        self.setup_parameters_section(scrollable_frame)
        
        # Control buttons
        self.setup_control_section(scrollable_frame)
        
    def setup_right_frame(self, parent):
        """Setup the right frame with results and visualization"""
        # Create notebook for tabs
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Results tab
        results_tab = ttk.Frame(notebook)
        notebook.add(results_tab, text="Results")
        self.setup_results_tab(results_tab)
        
        # Visualization tab
        visualization_tab = ttk.Frame(notebook)
        notebook.add(visualization_tab, text="Visualization")
        self.setup_visualization_tab(visualization_tab)
        
    def setup_data_section(self, parent):
        """Setup data loading section"""
        data_frame = ttk.LabelFrame(parent, text="Data Management", padding="10")
        data_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(data_frame, text="Load Penguins Dataset", 
                  command=self.load_dataset).pack(side=tk.LEFT, padx=(0, 10))
        
        self.data_status = ttk.Label(data_frame, text="No data loaded", foreground="red")
        self.data_status.pack(side=tk.LEFT)
        
    def setup_algorithm_section(self, parent):
        """Setup algorithm selection section"""
        algo_frame = ttk.LabelFrame(parent, text="Algorithm Selection", padding="10")
        algo_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.algorithm_var = tk.StringVar(value="perceptron")
        
        ttk.Radiobutton(algo_frame, text="Perceptron", 
                       variable=self.algorithm_var, value="perceptron").pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(algo_frame, text="Adaline", 
                       variable=self.algorithm_var, value="adaline").pack(side=tk.LEFT)
        
    def setup_selection_section(self, parent):
        """Setup feature and class selection section"""
        selection_frame = ttk.LabelFrame(parent, text="Feature and Class Selection", padding="10")
        selection_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Feature selection
        feature_frame = ttk.Frame(selection_frame)
        feature_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(feature_frame, text="Feature 1:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.feature1_var = tk.StringVar()
        self.feature1_combo = ttk.Combobox(feature_frame, textvariable=self.feature1_var, state="readonly", width=20)
        self.feature1_combo.grid(row=0, column=1, padx=(0, 20), sticky=(tk.W, tk.E))
        
        ttk.Label(feature_frame, text="Feature 2:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.feature2_var = tk.StringVar()
        self.feature2_combo = ttk.Combobox(feature_frame, textvariable=self.feature2_var, state="readonly", width=20)
        self.feature2_combo.grid(row=0, column=3, sticky=(tk.W, tk.E))
        
        feature_frame.columnconfigure(1, weight=1)
        feature_frame.columnconfigure(3, weight=1)
        
        # Class selection
        class_frame = ttk.Frame(selection_frame)
        class_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(class_frame, text="Class 1:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.class1_var = tk.StringVar()
        self.class1_combo = ttk.Combobox(class_frame, textvariable=self.class1_var, state="readonly", width=20)
        self.class1_combo.grid(row=0, column=1, padx=(0, 20), sticky=(tk.W, tk.E))
        
        ttk.Label(class_frame, text="Class 2:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.class2_var = tk.StringVar()
        self.class2_combo = ttk.Combobox(class_frame, textvariable=self.class2_var, state="readonly", width=20)
        self.class2_combo.grid(row=0, column=3, sticky=(tk.W, tk.E))
        
        class_frame.columnconfigure(1, weight=1)
        class_frame.columnconfigure(3, weight=1)
        
    def setup_parameters_section(self, parent):
        """Setup hyperparameters section"""
        param_frame = ttk.LabelFrame(parent, text="Hyperparameters", padding="10")
        param_frame.pack(fill=tk.X, pady=(0, 10))
        
        # First row of parameters
        param_row1 = ttk.Frame(param_frame)
        param_row1.pack(fill=tk.X, pady=5)
        
        ttk.Label(param_row1, text="Learning Rate (η):").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.learning_rate_var = tk.DoubleVar(value=0.01)
        ttk.Entry(param_row1, textvariable=self.learning_rate_var, width=10).grid(row=0, column=1, padx=(0, 20))
        
        ttk.Label(param_row1, text="Epochs:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.epochs_var = tk.IntVar(value=1000)
        ttk.Entry(param_row1, textvariable=self.epochs_var, width=10).grid(row=0, column=3, padx=(0, 20))
        
        # Second row of parameters
        param_row2 = ttk.Frame(param_frame)
        param_row2.pack(fill=tk.X, pady=5)
        
        ttk.Label(param_row2, text="MSE Threshold:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.mse_threshold_var = tk.DoubleVar(value=0.001)
        ttk.Entry(param_row2, textvariable=self.mse_threshold_var, width=10).grid(row=0, column=1, padx=(0, 20))
        
        # Bias checkbox
        self.bias_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_row2, text="Add Bias", variable=self.bias_var).grid(row=0, column=2, padx=(20, 0))
        
    def setup_control_section(self, parent):
        """Setup control buttons section"""
        control_frame = ttk.LabelFrame(parent, text="Model Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # First row of buttons
        button_row1 = ttk.Frame(control_frame)
        button_row1.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_row1, text="Train Model", 
                  command=self.train_model).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_row1, text="Test Model", 
                  command=self.test_model).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_row1, text="Single Prediction", 
                  command=self.single_prediction).pack(side=tk.LEFT, padx=(0, 10))
        
        # Second row of buttons
        button_row2 = ttk.Frame(control_frame)
        button_row2.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_row2, text="Plot Results", 
                  command=self.plot_results).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_row2, text="Clear Plot", 
                  command=self.clear_plot).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_row2, text="Clear Results", 
                  command=self.clear_results).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_row2, text="Clear All", 
                  command=self.clear_all).pack(side=tk.LEFT)
        
    def setup_results_tab(self, parent):
        """Setup results tab with scrolling"""
        # Create frame for text and scrollbars
        text_frame = ttk.Frame(parent)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Text widget for results
        self.results_text = tk.Text(text_frame, wrap=tk.WORD, font=('Consolas', 10))
        
        # Vertical scrollbar
        v_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=v_scrollbar.set)
        
        # Horizontal scrollbar
        h_scrollbar = ttk.Scrollbar(text_frame, orient="horizontal", command=self.results_text.xview)
        self.results_text.configure(xscrollcommand=h_scrollbar.set)
        
        # Pack everything
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def setup_visualization_tab(self, parent):
        """Setup visualization tab with horizontal scrolling"""
        # Create main frame for visualization tab
        viz_main_frame = ttk.Frame(parent)
        viz_main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a frame for the plot with scrollbars
        plot_frame = ttk.Frame(viz_main_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create canvas and scrollbars for the plot
        self.viz_canvas = tk.Canvas(plot_frame, bg='white')
        
        # Horizontal scrollbar
        h_scrollbar = ttk.Scrollbar(plot_frame, orient="horizontal", command=self.viz_canvas.xview)
        self.viz_canvas.configure(xscrollcommand=h_scrollbar.set)
        
        # Vertical scrollbar  
        v_scrollbar = ttk.Scrollbar(plot_frame, orient="vertical", command=self.viz_canvas.yview)
        self.viz_canvas.configure(yscrollcommand=v_scrollbar.set)
        
        # Pack scrollbars and canvas
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.viz_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create frame inside canvas for the plot
        self.plot_container = ttk.Frame(self.viz_canvas)
        self.viz_canvas.create_window((0, 0), window=self.plot_container, anchor="nw")
        
        # Update scroll region when plot container changes size
        self.plot_container.bind("<Configure>", self._on_plot_configure)
        
        # Status label for plot
        self.plot_status = ttk.Label(viz_main_frame, text="No plot generated", foreground="gray")
        self.plot_status.pack(side=tk.BOTTOM, pady=(5, 0))
        
    def _on_plot_configure(self, event):
        """Update scroll region when plot container is configured"""
        self.viz_canvas.configure(scrollregion=self.viz_canvas.bbox("all"))
        
    def load_dataset(self):
        """Load the penguins dataset"""
        try:
            # For now, we'll use the provided CSV file
            success = self.data_processor.load_data("penguins.csv")
            if success:
                self.data_status.config(text="Data loaded successfully", foreground="green")
                self.update_selection_combos()
                self.log_result("Penguins dataset loaded successfully!")
                self.log_result(f"Dataset shape: {self.data_processor.data.shape}")
                self.log_result(f"Available classes: {', '.join(self.data_processor.classes)}")
            else:
                messagebox.showerror("Error", "Failed to load dataset")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
    
    def update_selection_combos(self):
        """Update combo boxes with available features and classes"""
        # Update feature combinations
        features = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'OriginLocation', 'BodyMass']
        self.feature1_combo['values'] = features
        self.feature2_combo['values'] = features
        
        # Update class combinations
        classes = ['Adelie', 'Chinstrap', 'Gentoo']
        self.class1_combo['values'] = classes
        self.class2_combo['values'] = classes
        
        # Set default values
        if features:
            self.feature1_var.set(features[0])
            self.feature2_var.set(features[1])
        if classes:
            self.class1_var.set(classes[0])
            self.class2_var.set(classes[1])
    
    def train_model(self):
        """Train the selected neural network model"""
        try:
            # Validate inputs
            if not self.validate_inputs():
                return
            
            # Prepare data
            self.X_train, self.X_test, self.y_train, self.y_test, filtered_data = self.data_processor.prepare_data(
                self.feature1_var.get(), self.feature2_var.get(),
                self.class1_var.get(), self.class2_var.get()
            )
            
            # Get parameters
            learning_rate = self.learning_rate_var.get()
            epochs = self.epochs_var.get()
            add_bias = self.bias_var.get()
            
            algorithm = self.algorithm_var.get()
            
            if algorithm == "perceptron":
                self.current_model = Perceptron(
                    learning_rate=learning_rate,
                    n_iterations=epochs,
                    add_bias=add_bias
                )
            else:  # adaline
                mse_threshold = self.mse_threshold_var.get()
                self.current_model = Adaline(
                    learning_rate=learning_rate,
                    n_iterations=epochs,
                    mse_threshold=mse_threshold,
                    add_bias=add_bias
                )
            
            # Train model
            self.current_model.fit(self.X_train, self.y_train)
            
            self.current_features = (self.feature1_var.get(), self.feature2_var.get())
            self.current_classes = (self.class1_var.get(), self.class2_var.get())
            
            self.log_result(f"\n{'='*60}")
            self.log_result(f"{algorithm.upper()} TRAINING COMPLETED")
            self.log_result(f"{'='*60}")
            self.log_result(f"Features: {self.feature1_var.get()}, {self.feature2_var.get()}")
            self.log_result(f"Classes: {self.class1_var.get()} (0) vs {self.class2_var.get()} (1)")
            self.log_result(f"Training samples: {len(self.X_train)}")
            self.log_result(f"Testing samples: {len(self.X_test)}")
            self.log_result(f"Learning rate: {learning_rate}")
            self.log_result(f"Epochs: {epochs}")
            self.log_result(f"Bias: {'Yes' if add_bias else 'No'}")
            
            if algorithm == "perceptron":
                self.log_result(f"Final training errors: {self.current_model.errors[-1]}")
                self.log_result(f"Total epochs used: {len(self.current_model.errors)}")
            else:
                self.log_result(f"Final MSE: {self.current_model.mse_history[-1]:.6f}")
                self.log_result(f"Total epochs used: {len(self.current_model.mse_history)}")
            
        except Exception as e:
            messagebox.showerror("Training Error", f"Failed to train model: {str(e)}")
    
    def test_model(self):
        """Test the trained model"""
        try:
            if self.current_model is None:
                messagebox.showwarning("Warning", "Please train a model first!")
                return
            
            # Make predictions
            y_pred = self.current_model.predict(self.X_test)
            
            # Calculate confusion matrix and accuracy
            confusion_matrix, accuracy = self.nn_manager.create_confusion_matrix(self.y_test, y_pred)
            
            # Display results
            self.log_result(f"\n{'='*60}")
            self.log_result("TEST RESULTS")
            self.log_result(f"{'='*60}")
            self.log_result(f"Test samples: {len(self.X_test)}")
            self.log_result("\nConfusion Matrix:")
            self.log_result(f"                Predicted {self.current_classes[0]:<12} | Predicted {self.current_classes[1]:<12}")
            self.log_result(f"Actual {self.current_classes[0]:<10} {confusion_matrix[0, 0]:>10} | {confusion_matrix[0, 1]:>10}")
            self.log_result(f"Actual {self.current_classes[1]:<10} {confusion_matrix[1, 0]:>10} | {confusion_matrix[1, 1]:>10}")
            
            self.log_result(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Calculate additional metrics
            precision = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1]) if (confusion_matrix[1, 1] + confusion_matrix[0, 1]) > 0 else 0
            recall = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0]) if (confusion_matrix[1, 1] + confusion_matrix[1, 0]) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            self.log_result(f"Precision: {precision:.4f}")
            self.log_result(f"Recall:    {recall:.4f}")
            self.log_result(f"F1-Score:  {f1:.4f}")
            
        except Exception as e:
            messagebox.showerror("Testing Error", f"Failed to test model: {str(e)}")
    
    def plot_results(self):
        """Plot decision boundary and training history"""
        try:
            if self.current_model is None:
                messagebox.showwarning("Warning", "Please train a model first!")
                return
            
            # Clear previous plot
            self.clear_plot()
            
            # Create plot
            fig = self.nn_manager.plot_decision_boundary(
                np.vstack([self.X_train, self.X_test]), 
                np.hstack([self.y_train, self.y_test]), 
                self.current_model,
                self.current_features, 
                self.current_classes
            )
            
            # Embed plot in GUI
            self.current_figure = fig
            self.canvas = FigureCanvasTkAgg(fig, self.plot_container)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Update scroll region
            self._on_plot_configure(None)
            
            self.plot_status.config(text="Plot generated successfully - Use scrollbars to navigate", foreground="green")
            self.log_result("Plot generated successfully! Check the Visualization tab.")
            
        except Exception as e:
            messagebox.showerror("Plotting Error", f"Failed to create plot: {str(e)}")
            self.plot_status.config(text="Plot generation failed", foreground="red")
    
    def single_prediction(self):
        """Make prediction for a single sample"""
        try:
            if self.current_model is None:
                messagebox.showwarning("Warning", "Please train a model first!")
                return
            
            # Create prediction dialog
            self.create_prediction_dialog()
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Failed to make prediction: {str(e)}")
    
    def create_prediction_dialog(self):
        """Create dialog for single prediction with correctness checking"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Single Prediction")
        dialog.geometry("450x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
        
        ttk.Label(dialog, text=f"Single Sample Prediction", 
                 font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Feature inputs
        input_frame = ttk.LabelFrame(dialog, text="Input Features", padding="10")
        input_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Feature 1 input
        feature1_frame = ttk.Frame(input_frame)
        feature1_frame.pack(fill=tk.X, pady=5)
        ttk.Label(feature1_frame, text=f"{self.current_features[0]}:").pack(side=tk.LEFT)
        feature1_var = tk.DoubleVar()
        feature1_entry = ttk.Entry(feature1_frame, textvariable=feature1_var)
        feature1_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Feature 2 input
        feature2_frame = ttk.Frame(input_frame)
        feature2_frame.pack(fill=tk.X, pady=5)
        ttk.Label(feature2_frame, text=f"{self.current_features[1]}:").pack(side=tk.LEFT)
        feature2_var = tk.DoubleVar()
        feature2_entry = ttk.Entry(feature2_frame, textvariable=feature2_var)
        feature2_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # True class selection (for correctness checking)
        true_class_frame = ttk.LabelFrame(dialog, text="Verification", padding="10")
        true_class_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(true_class_frame, text="True Class (optional):").pack(side=tk.LEFT)
        true_class_var = tk.StringVar(value="Unknown")
        true_class_combo = ttk.Combobox(true_class_frame, textvariable=true_class_var, state="readonly", width=15)
        true_class_combo['values'] = ("Unknown", self.current_classes[0], self.current_classes[1])
        true_class_combo.pack(side=tk.RIGHT)
        
        # Result display
        result_frame = ttk.LabelFrame(dialog, text="Prediction Result", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        result_text = tk.Text(result_frame, height=6, wrap=tk.WORD, font=('Arial', 10))
        result_scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=result_text.yview)
        result_text.configure(yscrollcommand=result_scrollbar.set, state=tk.DISABLED)
        result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        result_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        def predict():
            try:
                sample = np.array([[feature1_var.get(), feature2_var.get()]])
                prediction = self.current_model.predict(sample)[0]
                
                predicted_class = self.current_classes[1] if prediction == 1 else self.current_classes[0]
                true_class = true_class_var.get()
                
                # Prepare result message
                result_message = f"Input Features:\n"
                result_message += f"  • {self.current_features[0]}: {feature1_var.get()}\n"
                result_message += f"  • {self.current_features[1]}: {feature2_var.get()}\n\n"
                result_message += f"Predicted Class: {predicted_class}\n"
                
                # Check correctness if true class is provided
                if true_class != "Unknown":
                    is_correct = (predicted_class == true_class)
                    result_message += f"True Class: {true_class}\n"
                    result_message += f"Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}"
                    
                    # Color coding
                    if is_correct:
                        result_text.tag_configure("correct", foreground="green", font=('Arial', 10, 'bold'))
                    else:
                        result_text.tag_configure("wrong", foreground="red", font=('Arial', 10, 'bold'))
                
                # Update result text
                result_text.configure(state=tk.NORMAL)
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, result_message)
                
                # Apply color tags
                if true_class != "Unknown":
                    start_idx = "5.0" if true_class != "Unknown" else "4.0"
                    if is_correct:
                        result_text.tag_add("correct", f"{start_idx}+2lines", f"{start_idx}+2lines+7chars")
                    else:
                        result_text.tag_add("wrong", f"{start_idx}+2lines", f"{start_idx}+2lines+6chars")
                
                result_text.configure(state=tk.DISABLED)
                
                # Log to main results
                self.log_result(f"\nSingle Prediction:")
                self.log_result(f"  Features: {self.current_features[0]}={feature1_var.get()}, "
                              f"{self.current_features[1]}={feature2_var.get()}")
                self.log_result(f"  Predicted: {predicted_class}")
                if true_class != "Unknown":
                    self.log_result(f"  True Class: {true_class}")
                    self.log_result(f"  Result: {'CORRECT' if is_correct else 'WRONG'}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Prediction failed: {str(e)}")
        
        def clear_inputs():
            feature1_var.set(0)
            feature2_var.set(0)
            true_class_var.set("Unknown")
            result_text.configure(state=tk.NORMAL)
            result_text.delete(1.0, tk.END)
            result_text.configure(state=tk.DISABLED)
            feature1_entry.focus()
        
        # Button frame
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Button(button_frame, text="Predict", command=predict).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Clear", command=clear_inputs).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Close", command=dialog.destroy).pack(side=tk.RIGHT)
        
        # Set focus to first entry and bind Enter key
        feature1_entry.focus()
        dialog.bind('<Return>', lambda e: predict())
    
    def clear_plot(self):
        """Clear the plot area"""
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.current_figure:
            plt.close(self.current_figure)
            self.current_figure = None
            
        # Clear any existing widgets in plot container
        for widget in self.plot_container.winfo_children():
            widget.destroy()
            
        self.plot_status.config(text="Plot cleared", foreground="gray")
        self.log_result("Plot cleared")
    
    def clear_results(self):
        """Clear the results text area"""
        self.results_text.delete(1.0, tk.END)
        self.log_result("Results cleared - ready for new operations")
    
    def clear_all(self):
        """Clear everything: results, plot, and reset model"""
        self.clear_plot()
        self.clear_results()
        
        # Reset model and data
        self.current_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Reset status
        self.data_status.config(text="Data loaded - Model reset", foreground="orange")
        self.plot_status.config(text="No plot generated", foreground="gray")
        
        self.log_result("="*60)
        self.log_result("ALL CLEARED - System reset")
        self.log_result("="*60)
        self.log_result("Ready for new training session")
    
    def validate_inputs(self):
        """Validate user inputs"""
        if not self.feature1_var.get() or not self.feature2_var.get():
            messagebox.showerror("Error", "Please select both features!")
            return False
        
        if not self.class1_var.get() or not self.class2_var.get():
            messagebox.showerror("Error", "Please select both classes!")
            return False
        
        if self.feature1_var.get() == self.feature2_var.get():
            messagebox.showerror("Error", "Please select two different features!")
            return False
        
        if self.class1_var.get() == self.class2_var.get():
            messagebox.showerror("Error", "Please select two different classes!")
            return False
        
        return True
    
    def log_result(self, message):
        """Add message to results text widget"""
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)