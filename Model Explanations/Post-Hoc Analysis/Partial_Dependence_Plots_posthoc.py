import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay
import pandas as pd
import json

class PartialDependencePlotsAnalysis:
    def __init__(self, model, X_train, X_test, y_test, feature_names):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names

    def plot_partial_dependence(self, features, target=0):
        # Plot partial dependence for selected features
        display = PartialDependenceDisplay.from_estimator(self.model, self.X_train, features, feature_names=self.feature_names, target=target)
        display.plot()
        plt.show()

    def plot_partial_dependence_2d(self, features, target=0):
        # Plot 2D partial dependence for selected pairs of features
        display = PartialDependenceDisplay.from_estimator(self.model, self.X_train, features, feature_names=self.feature_names, kind='both', target=target)
        display.plot()
        plt.show()

    def explain_instance(self, transformed_data, feature_names, target=0):
        # Convert feature names to indices
        feature_indices = [self.feature_names.index(name) for name in feature_names]
        
        # Convert target to class index
        target_index = self.model.classes_.tolist().index(target)
        
        # Plot partial dependence for selected features using the transformed data
        display = PartialDependenceDisplay.from_estimator(self.model, self.X_train, feature_indices, feature_names=self.feature_names, target=target_index)
        display.plot()
        plt.show()

# def read_json(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#     return data

# Example usage
# if __name__ == "__main__":
#     # Load example data
#     data = load_iris()
#     X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
#     feature_names = data.feature_names

#     # Train a RandomForest model
#     model = RandomForestClassifier(random_state=42)
#     model.fit(X_train, y_train)

#     # Perform partial dependence plots analysis
#     pdp_analysis = PartialDependencePlotsAnalysis(model, X_train, X_test, y_test, feature_names)
    
#     # Plot partial dependence for selected features
#     pdp_analysis.plot_partial_dependence([0, 1, 2], target=0)  # Indices of features to plot, target class 0

#     # Plot 2D partial dependence for selected pairs of features
#     pdp_analysis.plot_partial_dependence_2d([(0, 1), (1, 2)], target=0)  # Pairs of feature indices to plot, target class 0

#     # Load and transform the data instance
#     data_point = read_json('dataPoint.txt')
#     new_data_df = pd.DataFrame([data_point])
#     # Assuming 'processor' is defined and has a 'pipeline' attribute with a 'transform' method
#     transformed_data = processor.pipeline.transform(new_data_df)
#     transformed_data = transformed_data.reshape(-1)

#     # Explain the specific instance
#     pdp_analysis.explain_instance(transformed_data, [0, 1, 2], target=0)  # Indices of features to plot, target class 0