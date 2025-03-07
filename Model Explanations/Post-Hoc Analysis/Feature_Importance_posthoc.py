import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json

class FeatureImportanceAnalysis:
    def __init__(self, model, X_train, X_test, y_test, feature_names):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names

    def plot_feature_importance(self):
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances")
        plt.bar(range(self.X_train.shape[1]), importances[indices], align="center")
        plt.xticks(range(self.X_train.shape[1]), [self.feature_names[i] for i in indices], rotation=90)
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.show()

    def get_top_n_features(self, n=10):
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_n_features = [(self.feature_names[i], importances[i]) for i in indices[:n]]
        return top_n_features

    def print_top_n_features(self, n=10):
        top_n_features = self.get_top_n_features(n)
        print(f"Top {n} Features:")
        for feature, importance in top_n_features:
            print(f"{feature}: {importance:.4f}")

    def explain_instance(self, transformed_data):
        prediction = self.model.predict(transformed_data.reshape(1, -1))
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_contributions = transformed_data * importances

            print(f"Prediction for the instance: {prediction[0]}")
            print("Feature contributions:")
            for feature, contribution in zip(self.feature_names, feature_contributions):
                print(f"{feature}: {contribution:.4f}")

            # Plot feature contributions
            plt.figure(figsize=(12, 8))
            plt.title("Feature Contributions for the Instance")
            plt.bar(range(len(self.feature_names)), feature_contributions, align="center")
            plt.xticks(range(len(self.feature_names)), self.feature_names, rotation=90)
            plt.xlabel("Feature")
            plt.ylabel("Contribution")
            plt.show()
        else:
            print("The model does not have feature importances.")

# def read_json(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#     return data

# Example usage
# if __name__ == "__main__":
#     # Assuming you have a trained model and data
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.datasets import load_iris
#     from sklearn.model_selection import train_test_split

#     # Load example data
#     data = load_iris()
#     X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
#     feature_names = data.feature_names

#     # Train a RandomForest model
#     model = RandomForestClassifier(random_state=42)
#     model.fit(X_train, y_train)

#     # Perform feature importance analysis
#     feature_importance_analysis = FeatureImportanceAnalysis(model, X_train, X_test, y_test, feature_names)
#     feature_importance_analysis.plot_feature_importance()
#     feature_importance_analysis.print_top_n_features(n=5)

#     # Load and transform the data instance
#     data_point = read_json('dataPoint.txt')
#     new_data_df = pd.DataFrame([data_point])
#     # Assuming 'processor' is defined and has a 'pipeline' attribute with a 'transform' method
#     transformed_data = processor.pipeline.transform(new_data_df)
#     transformed_data = transformed_data.reshape(-1)

#     # Explain the specific instance
#     feature_importance_analysis.explain_instance(transformed_data)