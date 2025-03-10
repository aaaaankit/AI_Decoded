import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

class VizEBM:
    def __init__(self, model, X_train, X_test, y_test, feature_names, class_names):
        """
        Initialize with the trained EBM model and test/train data
        :param model: Trained EBM model
        :param X_train: Training feature data
        :param X_test: Test feature data
        :param y_test: Test target data
        :param feature_names: List of feature names
        :param class_names: List of class names
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.class_names = class_names

    # def explain_global(self):
    #     """
    #     Provide global explanations by showing the overall importance and effect of each feature on the model's predictions.
    #     """
    #     global_explanation = self.model.explain_global()
    #     print(global_explanation)
    #     # save to Model Explanations\Inherently Interpretable Analysis\Interpretable Model Results
        # global_explanation.save("Model Explanations/Inherently Interpretable Analysis/Interpretable Model Results/ebm/global/global_explanation")
    #     return global_explanation
    
    def explain_global(self):
        """
        Provide global explanations by showing the overall importance and effect of each feature on the model's predictions.
        """
        self.plot_feature_effects()
        self.plot_global_feature_importance()
        self.global_explanation_visual()

    def explain_local(self, instance):
        """
        Provide local explanations by showing the contribution of each feature to the prediction of a specific instance.
        :param instance: The instance to explain
        """
        local_explanation = self.model.explain_local(instance.reshape(1, -1))
        print(local_explanation)
        # save to Model Explanations\Inherently Interpretable Analysis\Interpretable Model Results
        local_explanation.save("Model Explanations/Inherently Interpretable Analysis/Interpretable Model Results/ebm/local_explanation")
        return local_explanation

    def plot_feature_effects(self):
        """
        Visualize the effect of each feature on the model's predictions in a combined plot.
        """
        global_explanation = self.model.explain_global()
        num_features = len(global_explanation.data())

        # Define the number of rows and columns for subplots
        cols = 3  # Arrange plots in 3 columns
        rows = (num_features // cols) + (num_features % cols > 0)  # Compute required rows

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))  # Adjust size dynamically
        axes = axes.flatten()  # Flatten in case of a single row

        # Ensure save directory exists
        save_dir = "Model Explanations/Inherently Interpretable Analysis/Interpretable Model Results/ebm/global"
        os.makedirs(save_dir, exist_ok=True)

        # Iterate through features and plot
        for i, feature_scores in enumerate(global_explanation.data()):
            feature_name = global_explanation.feature_names[i]

            axes[i].plot(feature_scores, marker='o', linestyle='-')
            axes[i].set_title(feature_name)
            axes[i].set_xlabel('Feature Value')
            axes[i].set_ylabel('Score')

        # Hide any extra subplots if the number of features is not a multiple of 3
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout and save the combined plot
        plt.tight_layout()
        plt.savefig(f"{save_dir}/combined_feature_effects.png")
        plt.show()
    
    def plot_global_feature_importance(self):
        """
        Visualize the global feature importance.
        """
        global_explanation = self.model.explain_global()
        feature_importances = global_explanation.data()

        # Debugging: Print the structure of feature_importances
        print("Feature importances structure:", feature_importances)

        # Extract feature names and their importances
        feature_names = feature_importances['names']
        importances = feature_importances['scores']

        # Create a DataFrame for easier plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })

        # Sort the DataFrame by importance
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Plot the feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Global Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')

        # Ensure save directory exists
        save_dir = "Model Explanations/Inherently Interpretable Analysis/Interpretable Model Results/ebm/global"
        os.makedirs(save_dir, exist_ok=True)

        # Save the plot
        plt.tight_layout()
        plt.savefig(f"{save_dir}/global_feature_importance.png")
        plt.show()

    def global_explanation_visual(self):
        """
        Provide a detailed global explanation for all features using visualization.
        """
        global_explanation = self.model.explain_global()
        feature_names = global_explanation.data()['names']
        scores = np.array(global_explanation.data()['scores'])  # Convert to NumPy array
        x = np.arange(len(feature_names))  # X-axis positions
        width = 0.25  # Bar width

        plt.figure(figsize=(15, 10))
        plt.title("Global Explanation for All Features")
        plt.bar(x, scores, width=width)

        plt.xlabel('Feature')
        plt.ylabel('Score')
        plt.xticks(x, feature_names, rotation=90)
        plt.tight_layout()
        # save to Model Explanations\Inherently Interpretable Analysis\Interpretable Model Results
        save_dir = "Model Explanations/Inherently Interpretable Analysis/Interpretable Model Results/ebm/global"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/global_explanation_visualization.png")
        plt.show()

    def local_explanation_text(self, instance):
        """
        Provide a detailed local explanation for a specific instance using textual representation and plot the results.
        :param instance: The instance to explain
        """
        local_explanation = self.model.explain_local(instance.reshape(1, -1))
        feature_scores = local_explanation.data(0)['scores']
        
        explanation = ["Local explanation for the instance:"]
        
        # Ensure the output directory exists
        save_dir = "Model Explanations/Inherently Interpretable Analysis/Interpretable Model Results/ebm"
        os.makedirs(save_dir, exist_ok=True)

        # Format text output
        for feature, value in zip(self.feature_names, feature_scores):
            formatted_values = ", ".join([f"{v:.4f}" for v in value])  # Format each value
            explanation.append(f"{feature}: [{formatted_values}]")

        # Print and save text output
        explanation_text = "\n".join(explanation)
        print(explanation_text)
        with open(os.path.join(save_dir, "feature_contributions.txt"), "w") as file:
            file.write(explanation_text)

        # **Plot the feature contributions**
        feature_scores = np.array(feature_scores)  # Convert to NumPy array
        num_classes = feature_scores.shape[1]  # Number of classes

        plt.figure(figsize=(10, 6))
        bar_width = 0.2  # Adjust bar width for visibility

        # Plot bars for each class
        for class_idx in range(num_classes):
            plt.bar(
                np.arange(len(self.feature_names)) + (class_idx * bar_width),
                feature_scores[:, class_idx],
                width=bar_width,
                label=f'Class {class_idx}'
            )

        plt.xticks(np.arange(len(self.feature_names)) + (bar_width * (num_classes - 1) / 2), 
                self.feature_names, rotation=90)
        plt.xlabel("Feature")
        plt.ylabel("Score")
        plt.title("Local Explanation - Feature Contributions")
        plt.legend()
        plt.tight_layout()
        # Save the plot
        plt.savefig(os.path.join(save_dir, "local_explanation_text.png"))
        plt.show()

        return explanation_text
    
    def local_explanation_visual(self, instance):
        """
        Provide a detailed local explanation for a specific instance using visualization
        :param instance: The instance to explain
        """
        local_explanation = self.model.explain_local(instance.reshape(1, -1))
        scores = np.array(local_explanation.data(0)['scores'])  # Convert to NumPy array
        x = np.arange(len(self.feature_names))  # X-axis positions
        width = 0.25  # Bar width

        plt.figure()
        plt.title("Local Explanation for All Classes")
        for i in range(scores.shape[1]):  # Loop over classes
            plt.bar(x + i * width, scores[:, i], width=width, label=f'Class {i}')

        plt.xlabel('Feature')
        plt.ylabel('Score')
        plt.xticks(x + width, self.feature_names, rotation=90)
        plt.legend()
        # save to Model Explanations\Inherently Interpretable Analysis\Interpretable Model Results
        plt.savefig("Model Explanations/Inherently Interpretable Analysis/Interpretable Model Results/ebm/local_explanation_visualization.png")
        plt.show()

    def local_explanation_importance(self, instance):
        """
        Provide a detailed local explanation for a specific instance using feature importance and plot the results.
        :param instance: The instance to explain
        """
        local_explanation = self.model.explain_local(instance.reshape(1, -1))
        feature_contributions = local_explanation.data(0)['scores']

        explanation = ["Feature contributions for the instance:"]

        # Ensure the output directory exists
        save_dir = "Model Explanations/Inherently Interpretable Analysis/Interpretable Model Results/ebm"
        os.makedirs(save_dir, exist_ok=True)

        # Format text output
        for feature, contribution in zip(self.feature_names, feature_contributions):
            explanation.append(f"{feature}: {list(contribution)}")

        # Print and save text output
        explanation_text = "\n".join(explanation)
        print(explanation_text)
        with open(os.path.join(save_dir, "feature_contributions_importance.txt"), "w") as file:
            file.write(explanation_text)

        # **Plot the feature contributions**
        feature_contributions = np.array(feature_contributions)

        plt.figure(figsize=(10, 6))
        bar_width = 0.2  # Adjust bar width for visibility

        # Plot bars for each class
        num_classes = feature_contributions.shape[1]
        for class_idx in range(num_classes):
            plt.bar(
                np.arange(len(self.feature_names)) + (class_idx * bar_width),
                feature_contributions[:, class_idx],
                width=bar_width,
                label=f'Class {class_idx}'
            )

        plt.xticks(np.arange(len(self.feature_names)) + (bar_width * (num_classes - 1) / 2), 
                self.feature_names, rotation=90)
        plt.xlabel("Feature")
        plt.ylabel("Importance Score")
        plt.title("Feature Importance Contributions")
        plt.legend()
        plt.tight_layout()
        # Save the plot
        plt.savefig(os.path.join(save_dir, "local_explanation_importance.png"))
        plt.show()

        return explanation_text