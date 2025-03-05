import matplotlib.pyplot as plt
import os
import pandas as pd
import graphviz
from sklearn.tree import export_graphviz, export_text
from lime.lime_tabular import LimeTabularExplainer

class VizTree:
    def __init__(self, model, X_train, X_test):
        """
        Initialize with the trained model and test/train data
        :param model: Trained decision tree model
        :param X_train: Training feature data
        :param X_test: Test feature data
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test

    def visualize_tree(self, output_path="Inherent_eval_result/tree_visualization"):
        """
        Visualize the decision tree as a PNG image using Graphviz
        :param output_path: Path to save the PNG image (default is "tree_visualization.png")
        """
        dot_data = export_graphviz(self.model, out_file=None, feature_names=self.X_test.columns, filled=True)
        graph = graphviz.Source(dot_data, format="png")
        graph.render(output_path)  # Save as PNG file
        # print(f"Decision tree visualized and saved to {output_path}")
        return output_path
    
    def feature_importance(self):
        """
        Compute and display feature importance
        """
        importances = self.model.feature_importances_
        features = self.X_train.columns
        
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        })
        
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        print(importance_df)
        return importance_df
    
    def explain_instance(self, instance_index=0):
        """
        Explain a single instance using LIME
        :param instance_index: Index of the instance to explain (default is 0)
        """
        explainer = LimeTabularExplainer(
            training_data=self.X_train.values, 
            feature_names=self.X_train.columns, 
            class_names=['class_0', 'class_1'], 
            discretize_continuous=True
        )
        
        instance = self.X_test.iloc[instance_index].values
        explanation = explainer.explain_instance(instance, self.model.predict_proba)
        # print(explanation.as_list())
        return explanation
    
    def export_rules(self):
        """
        Export decision tree rules in text format.
        """
        rules = export_text(self.model, feature_names=self.X_train.columns)
        # print(rules)
        return rules