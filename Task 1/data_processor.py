import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self):
        self.data = None
        self.features = [
            "CulmenLength",
            "CulmenDepth",
            "FlipperLength",
            "OriginLocation",
            "BodyMass",
        ]
        self.classes = ["Adelie", "Chinstrap", "Gentoo"]
        self.scaler = StandardScaler()

    def load_data(self, file_path):
        """Load and preprocess the penguins dataset"""
        self.data = pd.read_csv(file_path)
        self._preprocess_data()
        return True

    def _preprocess_data(self):
        """Preprocess the data: handle missing values and encode categorical variables"""

        self.data.fillna(method="ffill", inplace=True)

        location_mapping = {"Torgersen": 0, "Biscoe": 1, "Dream": 2}
        self.data["OriginLocation"] = self.data["OriginLocation"].map(location_mapping)

        species_mapping = {species: idx for idx, species in enumerate(self.classes)}
        self.data["Species"] = self.data["Species"].map(species_mapping)

    def prepare_data(self, feature1, feature2, class1, class2, random_state=None):
        """Prepare training and testing data for the selected features and classes"""

        class1_idx = self.classes.index(class1)
        class2_idx = self.classes.index(class2)

        filtered_data = self.data[
            self.data["Species"].isin([class1_idx, class2_idx])
        ].copy()

        X = filtered_data[[feature1, feature2]].values
        y = filtered_data["Species"].values

        y_binary = np.where(y == class1_idx, -1, 1)

        class0_indices = np.where(y_binary == -1)[0]
        class1_indices = np.where(y_binary == 1)[0]

        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(class0_indices)
        np.random.shuffle(class1_indices)

        class0_train = class0_indices[:30]
        class0_test = class0_indices[30:50]
        class1_train = class1_indices[:30]
        class1_test = class1_indices[30:50]

        train_indices = np.concatenate([class0_train, class1_train])
        test_indices = np.concatenate([class0_test, class1_test])

        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y_binary[train_indices]
        y_test = y_binary[test_indices]

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, filtered_data
