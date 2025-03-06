import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import shap
from interpret.glassbox import ExplainableBoostingClassifier
from pathlib import Path
import sys

cwd = Path(__file__).parent
sys.path

import Classification_models

class ExplainableBoostingTrainer(Classification_models.ClassificationModels):
    def __init__(self, X_train, y_train, X_test, y_test, model_path, evaluation_results="AI_Decoded/Classification Models/Evaluation Results"):
        modelName = 'ExplainableBoostingClassifier'
        super().__init__(X_train, y_train, X_test, y_test, modelName, model_path, evaluation_results)

        self.model = ExplainableBoostingClassifier(random_state=42)
    
    def train_ebm(self):
        """
        Train the Explainable Boosting Machine model using the train_model method from the parent class.
        """
        self.model = self.train_model(self.model)

    def evaluate_ebm(self):
        """
        Evaluate the Explainable Boosting Machine model using the evaluate_model method from the parent class.
        """
        self.evaluate_model(self.model)

    def save_ebm(self):
        """
        Save the trained Explainable Boosting Machine model using the save_model method from the parent class.
        """
        self.save_model(self.model)

    def load_ebm(self):
        """
        Load a pre-trained Explainable Boosting Machine model using the load_model method from the parent class.
        """
        self.model = self.load_model(self.model)
    
    def get_model(self):
        """Return the trained Explainable Boosting Machine model."""
        return self.model
