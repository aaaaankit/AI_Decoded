import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split


    
class SHAPAnalysis:
    """
    A class to perform SHAP (SHapley Additive exPlanations) analysis on a given machine learning model.

    This class is designed to generate various SHAP plots such as summary plots, feature importance plots,
    dependence plots, and force plots to interpret and visualize the contribution of each feature to model predictions.
    Attributes:
        model (sklearn estimator or similar): A trained machine learning model.
        X_train (pandas.DataFrame): The training dataset features.
        X_test (pandas.DataFrame): The test dataset features for which SHAP values will be calculated.
        shap_results_path (str): The path where SHAP plots will be saved.
    Methods:
        perform_shap_analysis(): 
            Perform SHAP analysis and generate SHAP plots (summary, feature importance, dependence, force).
    """
    def __init__(self, model, X_train, X_test, y_test, feature_names, shap_results_path):
        """
        Initializes the SHAPAnalysis class with the given model, training and test datasets, and results path.
        
        Parameters:
            model (sklearn estimator or similar): A trained machine learning model for which SHAP values will be computed.
            X_train (pandas.DataFrame): The training data.
            X_test (pandas.DataFrame): The test data used for SHAP value calculation.
            shap_results_path (str): The directory where SHAP plot images will be saved.
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.shap_results_path = shap_results_path

    def perform_shap_analysis(self, subset_percentage=0.1):
        """
        Perform SHAP analysis on a stratified subset of the test data, generating multiple SHAP plots.
        A random subset of the test data is selected using stratified sampling to increase speed.

        Args:
            subset_percentage (float): The percentage of the test set to use for SHAP analysis (0.1 = 10%).

        Returns:
            None: The function saves the plots to the given directory.
        """

        # Stratified Sampling to create a subset
        X_subset, _, y_subset, _ = train_test_split(self.X_test, self.y_test, test_size=(1 - subset_percentage), stratify=self.y_test, random_state=42)

        

        # Initialize SHAP explainer for the model
        explainer = shap.TreeExplainer(self.model)

        # Calculate SHAP values for the stratified subset
        shap_values = explainer.shap_values(X_subset)

        # Summary Plot
        os.makedirs(f"{self.shap_results_path}", exist_ok=True)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values[1], X_subset, feature_names=self.feature_names)
        plt.savefig(f"evaluation_results/shap_summary_plot.png")
        plt.show()
        plt.close()

        # Feature Importance Bar Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[1], X_subset, plot_type="bar", feature_names=self.feature_names)
        plt.savefig(f"evaluation_results/shap_feature_importance.png")
        plt.show()
        plt.close()

        # Dependence Plot for a selected feature (first feature in this case)
        feature_to_analyze = self.feature_names[0]
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(feature_to_analyze, shap_values[1], X_subset, feature_names=self.feature_names)
        plt.savefig(f"evaluation_results/shap_dependence_{feature_to_analyze}.png")
        plt.show()
        plt.close()

        # Force Plot for the first test instance
        shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], X_subset.iloc[0, :], feature_names=self.feature_names, matplotlib=True)
        plt.savefig(f"evaluation_results/shap_force_plot.png")
        plt.show()
        plt.close()