import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os


    
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
    def __init__(self, model, X_train, X_test, shap_results_path):
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
        self.shap_results_path = shap_results_path

    def perform_shap_analysis(self):
        """
        Perform SHAP analysis on the given model and data, generating multiple SHAP plots such as:
        - Summary plot
        - Feature importance bar plot
        - Dependence plot for a specified feature
        - Force plot for the first test instance
        
        The generated plots are saved as image files in the specified `shap_results_path` directory.
        
        Steps:
            1. Initialize a SHAP explainer using the TreeExplainer for tree-based models.
            2. Generate SHAP values for the test data.
            3. Create and save a summary plot showing the global impact of features.
            4. Create and save a feature importance plot (bar plot).
            5. Create and save a dependence plot for a specified feature.
            6. Create and save a force plot for the first instance in the test data.

        Returns:
            None: The function saves the plots to the given directory.
        """

        feature_names = self.X_test.columns
        
        # Initialize SHAP explainer for the model
        explainer = shap.TreeExplainer(self.model)
        
        # Calculate SHAP values for the test data
        shap_values = explainer.shap_values(self.X_test)

        # Summary Plot
        os.makedirs(f"{self.shap_results_path}", exist_ok=True)  
        plt.figure(figsize=(10, 8))  
        shap.summary_plot(shap_values[1], self.X_test, feature_names=feature_names)  
        plt.savefig(f"{self.shap_results_path}/shap_summary_plot.png") 
        plt.close()  

        # Feature Importance Bar Plot
        plt.figure(figsize=(10, 6))  
        shap.summary_plot(shap_values[1], self.X_test, plot_type="bar", feature_names=feature_names)  
        plt.savefig(f"{self.shap_results_path}/shap_feature_importance.png")  
        plt.close()  

        # Dependence Plot for a selected feature (first feature in this case)
        feature_to_analyze = feature_names[0]  
        plt.figure(figsize=(8, 6)) 
        shap.dependence_plot(feature_to_analyze, shap_values[1], self.X_test, feature_names=feature_names) 
        plt.savefig(f"{self.shap_results_path}/shap_dependence_{feature_to_analyze}.png") 
        plt.close()  

        # Force Plot for the first test instance
        shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], self.X_test.iloc[0, :], feature_names=feature_names, matplotlib=True)  
        plt.savefig(f"{self.shap_results_path}/shap_force_plot.png")  
        plt.close()  
