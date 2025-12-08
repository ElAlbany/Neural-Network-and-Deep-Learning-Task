import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    def __init__(self):
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
        self.data = pd.read_csv(file_path)
        self.data.bfill(inplace=True)

        location_mapping = {"Torgersen": 0, "Biscoe": 1, "Dream": 2}
        self.data["OriginLocation"] = self.data["OriginLocation"].map(location_mapping)
        species_mapping = {species: idx for idx, species in enumerate(self.classes)}
        self.data["Species"] = self.data["Species"].map(species_mapping)
        return True

    def prepare_data(self):
        """Split first 30 samples per class for training, remaining 20 for testing"""
        X_train, y_train, X_test, y_test = [], [], [], []

        for c in range(len(self.classes)):
            class_data = self.data[self.data["Species"] == c]
            X_class = class_data[self.features].values
            y_class = class_data["Species"].values

            X_train.append(X_class[:30])
            y_train.append(y_class[:30])
            X_test.append(X_class[30:50])
            y_test.append(y_class[30:50])

        X_train = np.vstack(X_train)
        y_train = np.hstack(y_train)
        X_test = np.vstack(X_test)
        y_test = np.hstack(y_test)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        Y_train_onehot = np.eye(len(self.classes))[y_train]

        return X_train_scaled, X_test_scaled, Y_train_onehot, y_test
