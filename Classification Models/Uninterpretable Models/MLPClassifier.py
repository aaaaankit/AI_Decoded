import sys
from sklearn.neural_network import MLPClassifier

from pathlib import Path

cwd = Path(__file__).parent
parent_cwd = Path(__file__).parents[1]

sys.path
sys.path.append(str(parent_cwd))

import Classification_models

class NeuralNetworkTrainer(Classification_models.ClassificationModels):
    """
    A class to train, evaluate, and manage a Neural Network model for classification tasks.

    This class extends the `ClassificationModels` class and provides specific methods to train, evaluate,
    save, and load an MLPClassifier model. It integrates the functionality of the parent class to perform
    the training and evaluation steps.

    Attributes:
        X_train (pandas.DataFrame): The features of the training dataset.
        y_train (pandas.Series): The labels of the training dataset.
        X_test (pandas.DataFrame): The features of the testing dataset.
        y_test (pandas.Series): The labels of the testing dataset.
        model_name (str): The name of the model being used (set to 'MLPClassifier').
        model (MLPClassifier): The Neural Network classifier model instance.
        model_path (str, optional): The path where the model is saved or loaded from (default is None).
    
    Methods:
        train_neural_network(): 
            Trains the Neural Network model using the `train_model` method from the parent class.
        evaluate_neural_network(): 
            Evaluates the Neural Network model using the `evaluate_model` method from the parent class.
        save_neural_network(): 
            Saves the trained Neural Network model using the `save_model` method from the parent class.
        load_neural_network(): 
            Loads a pre-trained Neural Network model using the `load_model` method from the parent class.
    """
    def __init__(self, X_train, y_train, X_test, y_test, model_path, evaluation_results="Classification Models/Evaluation Results"):
        modelName = 'MLPClassifier'
        super().__init__(X_train, y_train, X_test, y_test, modelName, model_path, evaluation_results)
        
        self.model = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam',             
                           random_state=42, max_iter=2000, learning_rate_init=0.001, alpha=0.0001)             


    def train_neural_network(self):
        """
        Train the Neural Network model using the train_model method from the parent class.
        """
        self.model = self.train_model(self.model)

    def evaluate_neural_network(self):
        """
        Evaluate the Neural Network model using the evaluate_model method from the parent class.
        """
        self.evaluate_model(self.model)

    def save_neural_network(self):
        """
        Save the trained Neural Network model using the save_model method from the parent class.
        """
        self.save_model(self.model)

    def load_neural_network(self):
        """
        Load a pre-trained Neural Network model using the load_model method from the parent class.
        """
        self.model = self.load_model(self.model)

    def get_model(self):
        """Return the trained Neural Network model."""
        return self.model
