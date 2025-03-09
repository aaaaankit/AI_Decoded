import matplotlib.pyplot as plt
import pandas as pd
import graphviz
from sklearn.tree import export_graphviz, export_text, plot_tree
import os
import matplotlib.image as mpimg


class VizTree:
    def __init__(self, model, X_train, X_test, y_test, feature_names, class_names):
        """
        Initialize with the trained model and test/train data
        :param model: Trained decision tree model
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

    def explain_global(self):
        """
        Provide global explanations by showing the overall importance and effect of each feature on the model's predictions.
        """
        self.visualize_tree()
        self.feature_importance()
        self.export_rules()

    def visualize_tree(self, output_path="Model Explanations/Inherently Interpretable Analysis/Interpretable Model Results/tree/global/tree_visualization"):
        """
        Visualize the decision tree as a PNG image using Graphviz
        :param output_path: Path to save the PNG image (default is "tree_visualization.png")
        """
        dot_data = export_graphviz(self.model, out_file=None, feature_names=self.feature_names, class_names=self.class_names, filled=True)
        graph = graphviz.Source(dot_data, format="png")
        graph.render(output_path)  # Save as PNG file
        print(f"Decision tree visualized and saved to {output_path}")
        return output_path

    def feature_importance(self):
        """
        Compute and display feature importance
        """
        importances = self.model.feature_importances_
        features = self.feature_names
        
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        })
        
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        print(importance_df)
        # save to Model Explanations\Inherently Interpretable Analysis\Interpretable Model Results
        save_dir = "Model Explanations/Inherently Interpretable Analysis/Interpretable Model Results/tree/global"
        os.makedirs(save_dir, exist_ok=True)
        importance_df.to_csv(f"{save_dir}/feature_importance.csv", index=False)
        
        # Plot the feature importance as a bar plot
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/feature_importance_plot.png")
        plt.show()
        
        return importance_df

    def export_rules(self):
        """
        Export decision tree rules in text format.
        """
        rules = export_text(self.model, feature_names=self.feature_names)
        print("Decision Tree Rules:")
        print(rules)
        # save to Model Explanations\Inherently Interpretable Analysis\Interpretable Model Results
        with open("Model Explanations/Inherently Interpretable Analysis/Interpretable Model Results/tree/global/decision_tree_rules.txt", "w") as file:
            file.write(rules)
        return rules

    def decision_path(self, instance):
        """
        Get the decision path for a specific instance
        :param instance: The instance to explain
        """
        node_indicator = self.model.decision_path(instance.reshape(1, -1))
        leaf_id = self.model.apply(instance.reshape(1, -1))

        feature = self.model.tree_.feature
        threshold = self.model.tree_.threshold

        print("Decision path for the instance:")
        for node_id in node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]:
            if leaf_id[0] == node_id:
                print(f"Leaf node {node_id} reached.")
            else:
                print(f"Node {node_id}: (instance[{feature[node_id]}] <= {threshold[node_id]})")

    def local_explanation_text(self, instance):
        """
        Provide a detailed local explanation for a specific instance using textual representation and a tree diagram.
        :param instance: The instance to explain
        """
        node_indicator = self.model.decision_path(instance.reshape(1, -1))
        leaf_id = self.model.apply(instance.reshape(1, -1))

        feature = self.model.tree_.feature
        threshold = self.model.tree_.threshold

        explanation = ["Decision path for the instance:"]
        decision_nodes = []  # Stores node descriptions for visualization

        dot = graphviz.Digraph(format="png")  # Create a decision tree diagram

        # Iterate through decision path nodes
        for node_id in node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]:
            if leaf_id[0] == node_id:
                explanation.append(f"Leaf node {node_id} reached.")
                decision_nodes.append((node_id, "Leaf node", None, None))
                dot.node(str(node_id), f"Leaf node {node_id}", shape="rectangle", style="filled", fillcolor="lightgreen")
            else:
                feature_name = self.feature_names[feature[node_id]]
                feature_value = instance[feature[node_id]]
                threshold_value = threshold[node_id]
                threshold_sign = "<=" if feature_value <= threshold_value else ">"
                decision_text = f"{feature_name} {threshold_sign} {threshold_value:.4f}"

                explanation.append(f"Node {node_id}: (feature '{feature_name}' {threshold_sign} {threshold_value:.4f})")
                decision_nodes.append((node_id, feature_name, threshold_value, threshold_sign))

                dot.node(str(node_id), decision_text, shape="ellipse", style="filled", fillcolor="lightblue")

        # Connect nodes in sequence
        for i in range(len(decision_nodes) - 1):
            dot.edge(str(decision_nodes[i][0]), str(decision_nodes[i + 1][0]))

        explanation_text = "\n".join(explanation)
        print(explanation_text)

        # Ensure save directory exists
        save_dir = "Model Explanations/Inherently Interpretable Analysis/Interpretable Model Results/tree"
        os.makedirs(save_dir, exist_ok=True)

        # Save textual explanation
        with open(f"{save_dir}/decision_path.txt", "w") as file:
            file.write(explanation_text)

        # Save and render decision path as a tree diagram
        tree_path_img = f"{save_dir}/decision_path_tree"
        dot.render(tree_path_img)  # Saves as PNG and other formats
        print(f"Decision path diagram saved as {tree_path_img}.png")

        img = mpimg.imread(tree_path_img + ".png")

        # Plot the image
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis("off")  # Hide axes for better visualization
        plt.show()


        return explanation_text

    def local_explanation_visual(self, instance):
        """
        Provide a detailed local explanation for a specific instance using visualization
        :param instance: The instance to explain
        """
        plt.figure(figsize=(20, 10))
        plot_tree(self.model, feature_names=self.feature_names, class_names=self.class_names, filled=True, rounded=True)
        # save to Model Explanations\Local Explanations\Decision Tree
        plt.savefig("Model Explanations/Inherently Interpretable Analysis/Interpretable Model Results/tree/decision_tree_visualization.png")
        plt.show()

    def local_explanation_importance(self, instance):
        """
        Provide a detailed local explanation for a specific instance using feature importance
        :param instance: The instance to explain
        """
        importances = self.model.feature_importances_
        feature_contributions = instance * importances

        explanation = []
        explanation.append("Feature contributions for the instance:")
        for feature, contribution in zip(self.feature_names, feature_contributions):
            explanation.append(f"{feature}: {contribution:.4f}")

        explanation_text = "\n".join(explanation)
        print(explanation_text)
        # save to Model Explanations\Local Explanations\Decision Tree
        save_dir = "Model Explanations/Inherently Interpretable Analysis/Interpretable Model Results/tree"
        with open(f"{save_dir}/feature_contributions.txt", "w") as file:
            file.write(explanation_text)
        
        # Plot the feature contributions as a bar plot
        plt.figure(figsize=(10, 6))
        plt.barh(self.feature_names, feature_contributions)
        plt.xlabel('Contribution')
        plt.ylabel('Feature')
        plt.title('Feature Contributions for the Instance')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/feature_contributions_plot.png")
        plt.show()

        return explanation_text