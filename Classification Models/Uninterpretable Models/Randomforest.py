import sys
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

cwd = Path(__file__).parent
parent_cwd = Path(__file__).parents[1]

sys.path
sys.path.append(str(parent_cwd))

import Classification_models

class RandomForestTrainer(Classification_models.ClassificationModels):
    """
    A class to train, evaluate, and manage a Random Forest model for classification tasks.

    This class extends the `ClassificationModels` class and provides specific methods to train, evaluate,
    save, and load a Random Forest classifier model. It uses the `RandomForestClassifier` from `sklearn.ensemble`
    and integrates the functionality of the parent class to perform the training and evaluation steps.
    Attributes:
        X_train (pandas.DataFrame): The features of the training dataset.
        y_train (pandas.Series): The labels of the training dataset.
        X_test (pandas.DataFrame): The features of the testing dataset.
        y_test (pandas.Series): The labels of the testing dataset.
        model_name (str): The name of the model being used (set to 'RandomForestClassifier').
        model (RandomForestClassifier): The Random Forest classifier model instance.
        model_path (str, optional): The path where the model is saved or loaded from (default is None).
    Methods:
        train_random_forest(): 
            Trains the Random Forest model using the `train_model` method from the parent class.
        evaluate_random_forest(): 
            Evaluates the Random Forest model using the `evaluate_model` method from the parent class.
        save_random_forest(): 
            Saves the trained Random Forest model to the specified path using the `save_model` method from the parent class.
        load_random_forest(): 
            Loads a pre-trained Random Forest model from the specified path using the `load_model` method from the parent class.
    """
    def __init__(self, X_train, y_train, X_test, y_test, model_path=None, evaluation_results=None):
        modelName = 'RandomForestClassifier'
        super().__init__(X_train, y_train, X_test, y_test, modelName, model_path, evaluation_results)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train_random_forest(self):
        """
        Train the Random Forest model using the train_model method from the parent class.
        """
        self.model = self.train_model(self.model)

    def evaluate_random_forest(self):
        """
        Evaluate the Random Forest model using the evaluate_model method from the parent class.
        """
        self.evaluate_model(self.model)

    def save_random_forest(self):
        """
        Save the trained Random Forest model using the save_model method from the parent class.
        """
        self.save_model(self.model)

    def load_random_forest(self):
        """
        Load a pre-trained Random Forest model using the load_model method from the parent class.
        """
        self.model = self.load_model(self.model)

    def get_model(self):
        """Return the trained Random Forest model."""
        return self.model