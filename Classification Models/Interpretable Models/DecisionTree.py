import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import shap
# from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
#import graphviz
from pathlib import Path
import sys

cwd = Path(__file__).parent
# parent_cwd = Path(__file__).parents[1]

sys.path
# sys.path.append(str(parent_cwd))

import Classification_models

# make class with Classification_models as parent class
class DecisionTreeTrainer(Classification_models.ClassificationModels):
    def __init__(self, X_train, y_train, X_test, y_test, model_path, evaluation_results="Classification Models/Evaluation Results"):
        modelName = 'DecisionTreeClassifier'
        super().__init__(X_train, y_train, X_test, y_test, modelName, model_path, evaluation_results)

        self.model = DecisionTreeClassifier(max_depth=10, random_state=42)
    
    def train_decision_tree(self):
        """
        Train the Decision Tree model using the train_model method from the parent class.
        """
        self.model = self.train_model(self.model)

    def evaluate_decision_tree(self):
        """
        Evaluate the Decision Tree model using the evaluate_model method from the parent class.
        """
        self.evaluate_model(self.model)

    def save_decision_tree(self):
        """
        Save the trained Decision Tree model using the save_model method from the parent class.
        """
        self.save_model(self.model)

    def load_decision_tree(self):
        """
        Load a pre-trained Decision Tree model using the load_model method from the parent class.
        """
        self.model = self.load_model(self.model)
    
    def get_model(self):
        """Return the trained Random Forest model."""
        return self.model

