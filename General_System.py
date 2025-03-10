import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import math
import numpy as np
import re

# Define root and current working directory
root = Path(__file__).parent
cwd = Path(__file__).parent

# Define paths for different model components
path_Data_processing = cwd / 'Data Processing'
path_Random_forest = cwd / 'Classification Models' / 'Uninterpretable Models'
path_Decision_tree = cwd / 'Classification Models' / 'Interpretable Models'
path_post_hoc = cwd / 'Model Explanations' / 'Post-Hoc Analysis'
path_inherent = cwd / 'Model Explanations' / 'Inherently Interpretable Analysis'

# Add these paths to the system path for module access
sys.path.append(str(path_Data_processing))
sys.path.append(str(path_Random_forest))
sys.path.append(str(path_Decision_tree))
sys.path.append(str(path_post_hoc))
sys.path.append(str(path_inherent))


class General:
    """
    A class to handle the training, evaluation, and explanation of classification models.

    This class provides functionality to train machine learning models, evaluate their performance using various metrics,
    and save/load models. It supports multi-class classification with evaluation metrics such as precision, recall, F1-score,
    AUC, and ROC curves. It also enables cross-validation and saving evaluation results and plots.

    Attributes:
        classification_model (str): The name of the classification model being used.
        global_explainer (str): The explanation technique used for model interpretability.
        inherent_models (list): A list of inherently interpretable models.

    Methods:
        performance_evaluation():
            Returns paths to evaluation results including confusion matrix, ROC curve, and evaluation metrics file.
        global_explanation():
            Returns paths to explanation results based on the chosen global explainer.
    """
    def __init__(self, classification_model, global_explainer):
        """
        Initializes the General class with the selected classification model and explanation method.

        Args:
        - classification_model (str): Name of the classification model.
        - global_explainer (str): Explanation technique used (e.g., SHAP, Inherent Interpretability).
        """
        self.classification_model = classification_model  
        self.global_explainer = global_explainer     
        self.inherent_models = ["Decision Tree", "Explainable Boosting Machine"]

    def performance_evaluation(self):
        """
        Returns paths to evaluation results including confusion matrix, ROC curve, and evaluation metrics file.

        Returns:
        - list: A list containing paths to evaluation result files.
        """
        base_path = "Classification Models/Evaluation Results"
        model_suffixes = {
            "Multi-layered Perceptron": "MLPClassifier",
            "RandomForest": "RandomForestClassifier",
            "Explainable Boosting Machine": "ExplainableBoostingClassifier",
            "Decision Tree": "DecisionTreeClassifier",
        }

        if self.classification_model in model_suffixes:
            suffix = model_suffixes[self.classification_model]
        
            # Define file paths for evaluation results
            confusion_matrix_path = f"{base_path}/confusion_matrix_{suffix}.png"
            roc_curve_path = f"{base_path}/roc_curve_{suffix}.png"
            evaluation_file_path = f"{base_path}/evaluation_{suffix}.txt"
            
            # Plot the ROC curve
            self.plot_images([roc_curve_path,confusion_matrix_path])
            return self.parse_performance_metrics(evaluation_file_path)


    def global_explanation(self):
        """
        Returns the path to explanation results based on the chosen global explainer.

        Returns:
        - str: The path to the explanation results.
        """
        explanation_paths = {
            "SHAP": {
                "Multi-layered Perceptron": [
                    "Model Explanations/Post-Hoc Analysis/Post-Hoc Analysis Results/shap_summary_plot_MLP NN.png",
                    "Model Explanations/Post-Hoc Analysis/Post-Hoc Analysis Results/shap_feature_importance_MLP NN.png"
                ],
                "RandomForest": [
                    "Model Explanations/Post-Hoc Analysis/Post-Hoc Analysis Results/shap_summary_plot_RandomForest.png",
                    "Model Explanations/Post-Hoc Analysis/Post-Hoc Analysis Results/shap_feature_importance_RandomForest.png"
                ],
                "Explainable Boosting Machine": [
                    "Model Explanations/Post-Hoc Analysis/Post-Hoc Analysis Results/shap_summary_plot_Explainable Boosted Machine.png",
                    "Model Explanations/Post-Hoc Analysis/Post-Hoc Analysis Results/shap_feature_importance_Explainable Boosted Machine.png"
                ],
                "Decision Tree": [
                    "Model Explanations/Post-Hoc Analysis/Post-Hoc Analysis Results/shap_summary_plot_Decision Tree.png",
                    "Model Explanations/Post-Hoc Analysis/Post-Hoc Analysis Results/shap_feature_importance_Decision Tree.png"
                ],
            },
            "Textual": {
                "Explainable Boosting Machine": [
                    "Model Explanations/Inherently Interpretable Analysis/Interpretable Model Results/ebm/local_explanation_text.png",
                ],
                "Decision Tree": [
                    "Model Explanations/Inherently Interpretable Analysis/Interpretable Model Results/tree/decision_path_tree.png",
                ],
            },
            "Visualize": {
                "Explainable Boosting Machine": [
                    "Model Explanations/Inherently Interpretable Analysis/Interpretable Model Results/ebm/global/global_explanation_visualization.png",
                ],
                "Decision Tree": [
                    "Model Explanations/Inherently Interpretable Analysis/Interpretable Model Results/tree/global/tree_visualization.png",
                ],
            },
            "Feature Importance": {
                "Explainable Boosting Machine": [
                    "Model Explanations/Inherently Interpretable Analysis/Interpretable Model Results/ebm/global/global_feature_importance.png",
                ],
                "Decision Tree": [
                    "Model Explanations/Inherently Interpretable Analysis/Interpretable Model Results/tree/global/feature_importance_plot.png",
                ],
            },
            
        }

        self.plot_images(explanation_paths.get(self.global_explainer, {}).get(self.classification_model, Exception))



    def plot_images(self, image_paths, max_cols=3, img_size=5):
        """
        Plots multiple images from a list of file paths in a grid.

        Parameters:
        - image_paths (list): List of image file paths.
        - max_cols (int): Maximum number of columns in the grid (default: 3).
        - img_size (int): Size multiplier for figure dimensions (default: 5).
        """
        if not image_paths:
            print("No images to display.")
            return

        num_images = len(image_paths)
        cols = min(max_cols, num_images)  # Limit columns to `max_cols` or fewer if needed
        rows = math.ceil(num_images / cols)  # Calculate required rows

        fig, axes = plt.subplots(rows, cols, figsize=(cols * img_size, rows * img_size))

        # Ensure axes is always a 2D array
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1 or cols == 1:
            axes = np.expand_dims(axes, axis=0 if rows == 1 else 1)

        for i, image_path in enumerate(image_paths):
            try:
                img = mpimg.imread(image_path)
                axes.flatten()[i].imshow(img)
                axes.flatten()[i].axis('off')
                if i == 1:
                    axes.flatten()[i].set_title("Feature Importance")
                else:
                    axes.flatten()[i].set_title("Summary Plot")
            except FileNotFoundError:
                axes.flatten()[i].set_title("Image Not Found")
                axes.flatten()[i].axis('off')

        # Hide any unused subplots
        for j in range(i + 1, len(axes.flatten())):
            axes.flatten()[j].axis('off')

        plt.tight_layout()
        plt.show()

    def plot_metrics(self, metrics):
        """
        Plots performance metrics in a readable format.

        Args:
        - metrics (dict): The evaluation metrics to display.
        """
        # Set up the grid layout to display metrics alongside images
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5])

        # Plot the metrics on the second subplot
        ax = fig.add_subplot(gs[1])
        ax.axis('off')

        # Display the metrics
        metric_text = "\n".join([f"{key}: {value}" for key, value in metrics.items()])
        ax.text(0.5, 0.5, metric_text, ha='center', va='center', fontsize=12, wrap=True)

        plt.show()


    def parse_performance_metrics(self, file_path):
        with open(file_path, 'r') as file:
            content = file.read()

        # Regex to extract the metrics
        precision = list(map(float, re.findall(r"Precision for each class:\s*\[(.*?)\]", content)[0].split()))
        recall = list(map(float, re.findall(r"Recall for each class:\s*\[(.*?)\]", content)[0].split()))
        f1_score = list(map(float, re.findall(r"F1-score for each class:\s*\[(.*?)\]", content)[0].split()))

        avg_precision = float(re.findall(r"Average Precision:\s*([0-9.]+)", content)[0])
        avg_recall = float(re.findall(r"Average Recall:\s*([0-9.]+)", content)[0])
        avg_f1_score = float(re.findall(r"Average F1-score:\s*([0-9.]+)", content)[0])

        auc = float(re.findall(r"AUC:\s*([0-9.]+)", content)[0])
        cv_scores = list(map(float, re.findall(r"Cross-Validation Scores:\s*\[(.*?)\]", content)[0].split()))
        avg_cv_score = float(re.findall(r"Average Cross-Validation Score:\s*([0-9.]+)", content)[0])

        # Creating the dictionary
        performance_metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_f1_score": avg_f1_score,
            "auc": auc,
            "cross_validation_scores": cv_scores,
            "average_cross_validation_score": avg_cv_score
        }

        return performance_metrics