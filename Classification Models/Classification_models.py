from sklearn.calibration import label_binarize
import joblib
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import os


class ClassificationModels:
    """
        A class to handle the training, evaluation, and management of classification models.

        This class provides functionality to train machine learning models, evaluate their performance using various metrics, 
        and save/load models. It supports multi-class classification with metrics such as precision, recall, F1-score, AUC, 
        and ROC curves. It also allows cross-validation and saving of evaluation results and plots.

        Attributes:
            X_train (pandas.DataFrame): The features of the training dataset.
            y_train (pandas.Series): The labels of the training dataset.
            X_test (pandas.DataFrame): The features of the testing dataset.
            y_test (pandas.Series): The labels of the testing dataset.
            model_name (str): The name of the model being used.
            model_path (str, optional): The path where the model is saved or loaded from (default is None).
            evaluation_results (str, optional): The path to save evaluation results (default is None).

        Methods:
            train_model(model): 
                Trains the model using the provided training data.
            save_model(model): 
                Saves the trained model to the specified path using joblib.
            load_model(model): 
                Loads a pre-trained model from the specified path using joblib.
            evaluate_model(model): 
                Evaluates the model using metrics such as precision, recall, F1-score, AUC, and cross-validation scores.
            plot_roc_curve_multiclass(model): 
                Plots and saves the ROC curve for multi-class classification.
    """
    def __init__(self, X_train, y_train, X_test, y_test, model_name, model_path=None, evaluation_results=None):
        """
        Initializes the ClassificationModels class.

        Args:
        - X_train: Features of the training data.
        - y_train: Labels for the training data.
        - X_test: Features of the test data.
        - y_test: Labels for the test data.
        - model_name: Name of the model.
        - model_path: Optional path to save/load the model (default is None).
        - evaluation_results: Optional path to save evaluation results (default is None).
        """
        self.X_train = X_train  
        self.y_train = y_train  
        self.X_test = X_test    
        self.y_test = y_test    
        self.model_path = model_path  
        self.model_name = model_name  
        self.evaluation_results = evaluation_results 

    def train_model(self, model):
        """
        Train the model on the training data.

        Args:
        - model: The model to be trained.

        Returns:
        - The trained model.
        """
        model.fit(self.X_train, self.y_train)  
        return model  

    def save_model(self, model):
        """
        Save the trained model to the specified path using joblib.

        Args:
        - model: The trained model to save.

        Raises:
        - ValueError: If model path is not provided.
        """
        if not self.model_path:  
            raise ValueError("File path is not provided. Please specify a file path to save the model.")
        
        try:
            joblib.dump(model, self.model_path)  
            print(f"Model saved successfully to {self.model_path}.")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, model):
        """
        Load a pre-trained model from the specified path using joblib.

        Args:
        - model: The model instance to load into.

        Returns:
        - The loaded model.
        
        Raises:
        - ValueError: If model path is not provided.
        """
        if not self.model_path:  
            raise ValueError("File path is not provided.")
        
        try:
            model = joblib.load(self.model_path)  
            print(f"Model loaded successfully from {self.model_path}.")
        except Exception as e:
            print(f"Error loading model: {e}")

        return model  
    

    def evaluate_model(self, model):
        """
        Evaluate the model using various evaluation metrics (precision, recall, F1 score, etc.).

        Args:
        - model: The trained model to evaluate.
        """
        y_pred = model.predict(self.X_test) 
        
        # Calculate precision, recall, and F1 score for each class (None means for each class separately)
        precision = precision_score(self.y_test, y_pred, average=None, zero_division=1)
        recall = recall_score(self.y_test, y_pred, average=None, zero_division=1)
        f1 = f1_score(self.y_test, y_pred, average=None, zero_division=1)
    
        # Calculate weighted average precision, recall, and F1 score
        precision_avg = precision_score(self.y_test, y_pred, average='weighted', zero_division=1)
        recall_avg = recall_score(self.y_test, y_pred, average='weighted', zero_division=1)
        f1_avg = f1_score(self.y_test, y_pred, average='weighted', zero_division=1)
    
        # Generate a classification report
        classification_rep = classification_report(self.y_test, y_pred)
    
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
    
        # ROC Curve and AUC score (only if model supports predict_proba)
        try:
            auc = roc_auc_score(self.y_test, model.predict_proba(self.X_test), multi_class='ovr')  
        except AttributeError:
            auc = "Model does not support probability prediction"  
    
        # Cross-validation scores 
        cross_val_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
    
        # Save the evaluation metrics to a text file
        os.makedirs("evaluation_results", exist_ok=True)  
        with open(f"evaluation_results/evaluation_{self.model_name}.txt", "w") as file:
            file.write("Precision for each class:\n")
            file.write(f"{precision}\n\n")
            file.write("Recall for each class:\n")
            file.write(f"{recall}\n\n")
            file.write("F1-score for each class:\n")
            file.write(f"{f1}\n\n")
            file.write(f"Average Precision: {precision_avg}\n")
            file.write(f"Average Recall: {recall_avg}\n")
            file.write(f"Average F1-score: {f1_avg}\n\n")
            file.write("Classification Report:\n")
            file.write(classification_rep)
            file.write("Confusion Matrix:\n")
            file.write(f"{cm}\n\n")
            file.write(f"AUC: {auc}\n")
            file.write(f"Cross-Validation Scores: {cross_val_scores}\n")
            file.write(f"Average Cross-Validation Score: {cross_val_scores.mean()}\n")
    
        print(f"Evaluation metrics saved to 'evaluation_results/evaluation_{self.model_name}.txt'")
    
        # Plot and save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(set(self.y_test)), yticklabels=list(set(self.y_test)))
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(f"evaluation_results/confusion_matrix_{self.model_name}.png")  
        plt.close()
    
        # Plot and save ROC curve
        self.plot_roc_curve_multiclass(model)
    
    def plot_roc_curve_multiclass(self, model):
        """
        Plot the ROC curve for multi-class classification.

        Args:
        - model: The trained model to evaluate.
        """
        # Get predicted probabilities for each class
        y_probs = model.predict_proba(self.X_test)
        
        # Binarize the test labels for multi-class ROC curve
        y_test_bin = label_binarize(self.y_test, classes=list(set(self.y_test)))  
    
        plt.figure(figsize=(8, 6))  
        fpr, tpr, roc_auc = {}, {}, {}
        
        # Iterate over each class to calculate FPR, TPR, and AUC for the ROC curve
        for i in range(y_test_bin.shape[1]):  
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_probs[:, i])  
            roc_auc[i] = auc(fpr[i], tpr[i])  
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')  
    
        # Plot the ROC curve
        plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Multi-Class')
        plt.legend(loc='lower right')
        plt.show()  

        # Save the ROC curve
        plt.savefig(f"evaluation_results/roc_curve_{self.model_name}.png") 
        plt.close() 










