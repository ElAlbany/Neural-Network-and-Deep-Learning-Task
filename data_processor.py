import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self):
        self.data = None
        self.features = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'OriginLocation', 'BodyMass']
        self.classes = ['Adelie', 'Chinstrap', 'Gentoo']
        
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
    
    def prepare_data(self, feature1, feature2, class1, class2, test_size=0.4):
        """Prepare training and testing data for the selected features and classes"""
        # Filter data for selected classes
        class1_idx = self.classes.index(class1)
        class2_idx = self.classes.index(class2)
        
        filtered_data = self.data[self.data['Species'].isin([class1_idx, class2_idx])].copy()
        
        # Extract features and labels
        X = filtered_data[[feature1, feature2]].values
        y = filtered_data['Species'].values
        
        # Convert to binary classification (0 for class1, 1 for class2)
        y_binary = np.where(y == class1_idx, 0, 1)
        
        # Split data (30 training, 20 testing per class as required)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=test_size, stratify=y_binary, random_state=42
        )
        
        return X_train, X_test, y_train, y_test, filtered_data