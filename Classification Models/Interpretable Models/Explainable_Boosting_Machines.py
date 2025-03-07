import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from interpret.glassbox import ExplainableBoostingClassifier
from pathlib import Path
import sys
from interpret import show
import plotly.io as pio
#pio.renderers.default = "vscode"


cwd = Path(__file__).parent
sys.path

import Classification_models

class ExplainableBoostingTrainer(Classification_models.ClassificationModels):
    def __init__(self, X_train, y_train, X_test, y_test, feature_names, model_path, evaluation_results="AI_Decoded/Classification Models/Evaluation Results"):
        modelName = 'ExplainableBoostingClassifier'
        super().__init__(X_train, y_train, X_test, y_test, modelName, model_path, evaluation_results)

        self.model = ExplainableBoostingClassifier(random_state=42, feature_names=feature_names)
    
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
    
    def local_explanation(self,instance):
        """
        Generate local explanations for the first 4 test samples and display them using Plotly.
        """
        try:
            # Get local explanation from EBM
            ebm_local = self.model.explain_local([instance], self.y_test[0], name='EBM')
            plotly_fig = ebm_local.visualize(0)  # Get Plotly figure
            #plotly_fig.write_image(f"local_images/fig_{i}.png")
            pio.show(plotly_fig)  # Display the Plotly figure

        except AttributeError:
            print("Model does not support explain_local.")

    def global_explanation(self):
        """
        Generate local explanations for the first 4 test samples and display them using Plotly.
        """
        ebm_global = self.model.explain_global()
        for index, value in enumerate(self.model.feature_bounds_):
            plotly_fig = ebm_global.visualize(index)
            #plotly_fig.write_image(f"global_images/fig_{index}.png")
            pio.show(plotly_fig)  # Display the Plotly figure


    #def global_explanation(self):
