import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self):
        self.data = None
        self.features = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'OriginLocation', 'BodyMass']
        self.classes = ['Adelie', 'Chinstrap', 'Gentoo']
        self.scaler = StandardScaler()
        
    def load_data(self, file_path):
        """Load and preprocess the penguins dataset"""
        try:
            self.data = pd.read_csv(file_path)
            self._preprocess_data()
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def _preprocess_data(self):
        """Preprocess the data: handle missing values and encode categorical variables"""
        # Handle missing values by forward fill
        self.data.fillna(method='ffill', inplace=True)
        
        # Encode OriginLocation
        location_mapping = {'Torgersen': 0, 'Biscoe': 1, 'Dream': 2}
        self.data['OriginLocation'] = self.data['OriginLocation'].map(location_mapping)
        
        # Encode Species
        species_mapping = {species: idx for idx, species in enumerate(self.classes)}
        self.data['Species'] = self.data['Species'].map(species_mapping)
    
    def get_feature_combinations(self):
        """Get all possible feature combinations"""
        numeric_features = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'OriginLocation', 'BodyMass']
        combinations = []
        for i in range(len(numeric_features)):
            for j in range(i + 1, len(numeric_features)):
                combinations.append((numeric_features[i], numeric_features[j]))
        return combinations
    
    def get_class_combinations(self):
        """Get all possible class combinations"""
        combinations = []
        for i in range(len(self.classes)):
            for j in range(i + 1, len(self.classes)):
                combinations.append((self.classes[i], self.classes[j]))
        return combinations
    
    def prepare_data(self, feature1, feature2, class1, class2, test_size=0.4, random_state=None):
        """Prepare training and testing data for the selected features and classes"""
        # Filter data for selected classes
        class1_idx = self.classes.index(class1)
        class2_idx = self.classes.index(class2)
        
        filtered_data = self.data[self.data['Species'].isin([class1_idx, class2_idx])].copy()
        
        # Extract features and labels
        X = filtered_data[[feature1, feature2]].values
        y = filtered_data['Species'].values
        
        # Convert to binary classification (0 for class1, 1 for class2)
        y_binary = np.where(y == class1_idx, -1, 1)  # Use -1/1 for better convergence
        
        # Split data (30 training, 20 testing per class as required)
        # Manual split to ensure exactly 30 train and 20 test per class
        class0_indices = np.where(y_binary == -1)[0]
        class1_indices = np.where(y_binary == 1)[0]
        
        # Shuffle indices
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(class0_indices)
        np.random.shuffle(class1_indices)
        
        # Take exactly 30 for training and 20 for testing from each class
        class0_train = class0_indices[:30]
        class0_test = class0_indices[30:50]
        class1_train = class1_indices[:30]
        class1_test = class1_indices[30:50]
        
        # Combine indices
        train_indices = np.concatenate([class0_train, class1_train])
        test_indices = np.concatenate([class0_test, class1_test])
        
        # Create splits
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y_binary[train_indices]
        y_test = y_binary[test_indices]
        
        # Feature scaling (CRITICAL for neural networks)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, filtered_data
    
    def scale_single_sample(self, sample):
        """Scale a single sample for prediction"""
        return self.scaler.transform(sample.reshape(1, -1))[0]