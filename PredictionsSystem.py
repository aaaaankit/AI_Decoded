import json
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text

root = Path(__file__).parent
cwd = Path(__file__).parent
path_Data_processing = cwd / 'Data Processing'
path_Random_forest = cwd / 'Classification Models' / 'Uninterpretable Models'
path_Decision_tree = cwd / 'Classification Models' / 'Interpretable Models'
path_post_hoc = cwd / 'Model Explanations' / 'Post-Hoc Analysis'
path_inherent = cwd / 'Model Explanations' / 'Inherently Interpretable Analysis'

sys.path
sys.path.append(str(path_Data_processing))
sys.path.append(str(path_Random_forest))
sys.path.append(str(path_Decision_tree))
sys.path.append(str(path_post_hoc))
sys.path.append(str(path_inherent))

import DataProcessor
import DataTransformer
import Randomforest
import DecisionTree 
import Explainable_Boosting_Machines
import MLPClassifier
import SHAP_posthoc
import Anchor_posthoc
import LIME_posthoc



class Predictor:
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
    def __init__(self, classification_model, local_explainer, global_explainer, data_point):
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
        self.classification_model = classification_model  
        self.local_explainer = local_explainer  
        self.global_explainer = global_explainer     
        self.inherent_models = ["Decision Tree", "Explainable Boosting Machine"]

        df = pd.read_csv('Dataset/cox-violent-parsed_filt.csv')
        df = df.dropna(subset=["score_text"])
        df['race'] = df['race'].str.replace('African-American', 'African American')     

        relevant = ["sex","age","race","juv_fel_count","juv_misd_count","juv_other_count",
                    "c_charge_degree","r_charge_degree","r_days_from_arrest",
                   "vr_charge_degree"]

        target = "score_text"

        # Define the columns to One-Hot Encode
        one_hot_columns = ['race']

        self.y = df[target]
        df.drop(target, axis=1, inplace=True)

        self.processor = DataTransformer.DataTransformer(
            df=df,
            y=y,
            relevant_columns=relevant,
            onehot_cols=one_hot_columns,
        )

        self.processor.load_pipeline("Data Processing/DataTransformer.pkl")

        self.X_train, self.X_test, self.y_train, self.y_test = self.processor.split_data()
        new_data_df = pd.DataFrame([data_point])
        self.transformed_data = self.processor.pipeline.transform(new_data_df)



    def classification(self):
        if self.classification_model == "Multi-layered Perceptron":
            model = MLPClassifier.NeuralNetworkTrainer(self.X_train, self.y_train, self.X_test, self.y_test, model_path="Classification Models/Saved Models/Test_NeuralNet")
            model.load_neural_network()
            self.trained_model = model
        elif self.classification_model == "RandomForest":
            model = Randomforest.RandomForestTrainer(self.X_train, self.y_train, self.X_test, self.y_test, model_path="Classification Models/Saved Models/Test_RandomForest")
            model.load_random_forest()
            self.trained_model = model.get_model()
        elif self.classification_model == "Explainable Boosting Machine":
            model = Explainable_Boosting_Machines.ExplainableBoostingTrainer(self.X_train, self.y_train, self.X_test, self.y_test, model_path="Classification Models/Saved Models/Test_ExplainableBoosting", evaluation_results="Eval")
            model.load_ebm()
            self.trained_model = model.get_model()
        elif self.classification_model == "Decision Tree":
            model = DecisionTree.DecisionTreeTrainer(self.X_train, self.y_train, self.X_test, self.y_test, model_path="Classification Models/Saved Models/Test_DecisionTree")
            model.load_decision_tree()
            self.trained_model = model.get_model()

        prediction = self.trained_model.predict(self.transformed_data)
        return prediction

    def performance_evaluation(self):
        if self.classification_model == "Multi-layered Perceptron":
            model_evaluation_path = ["Classification Models/Evaluation Results/confusion_matrix_MLPClassifier.png",
                                     "Classification Models/Evaluation Results/roc_curve_MLPClassifier.png",
                                     "Classification Models/Evaluation Results/evaluation_MLPClassifier.txt"]
            return model_evaluation_path
        elif self.classification_model == "RandomForest":
            model_evaluation_path = ["Classification Models/Evaluation Results/confusion_matrix_RandomForestClassifier.png",
                                     "Classification Models/Evaluation Results/roc_curve_RandomForestClassifier.png",
                                     "Classification Models/Evaluation Results/evaluation_RandomForestClassifier.txt"]
            return model_evaluation_path
        elif self.classification_model == "Explainable Boosting Machine":
            model_evaluation_path = ["Classification Models/Evaluation Results/confusion_matrix_ExplainableBoostingClassifier.png",
                                     "Classification Models/Evaluation Results/roc_curve_ExplainableBoostingClassifier.png",
                                     "Classification Models/Evaluation Results/evaluation_ExplainableBoostingClassifier.txt"]
            return model_evaluation_path
        elif self.classification_model == "Decision Tree":
            model_evaluation_path = ["Classification Models/Evaluation Results/confusion_matrix_DecisionTreeClassifier.png",
                                     "Classification Models/Evaluation Results/roc_curve_DecisionTreeClassifier.png",
                                     "Classification Models/Evaluation Results/evaluation_DecisionTreeClassifier.txt"]
            return model_evaluation_path 
        
    def global_explanation(self):
        if self.global_explainer == "SHAP":
            if self.classification_model == "Multi-layered Perceptron":
                return "Model Explanations/Post-Hoc Analysis/Post-Hoc Analysis Results/....."
            elif self.classification_model == "RandomForest":
                return "Model Explanations/Post-Hoc Analysis/Post-Hoc Analysis Results/....."
            elif self.classification_model == "Explainable Boosting Machine":
                return "Model Explanations/Post-Hoc Analysis/Post-Hoc Analysis Results/....."
            elif self.classification_model == "Decision Tree":
                return "Model Explanations/Post-Hoc Analysis/Post-Hoc Analysis Results/....."
        elif self.global_explainer == "Inherent":
            if self.classification_model not in self.inherent_models:
                return Exception
        
            if self.classification_model == "Explainable Boosting Machine":
                return "Model Explanations/Post-Hoc Analysis/Post-Hoc Analysis Results/....."
            elif self.classification_model == "Decision Tree":
                return "Model Explanations/Post-Hoc Analysis/Post-Hoc Analysis Results/....."
    
    def local_explanation(self):
        if self.global_explainer == "SHAP":
            shap_model = SHAP_posthoc.SHAPAnalysis(self.trained_model, self.X_train, self.X_test, self.y_test, self.processor.get_feature_names(), "Model Explanations/Post-Hoc Analysis/Post-Hoc Analysis Results")
            return shap_model.perform_shap_local_explanation_instance(self.transformed_data)
        elif self.global_explainer == "Inherent":
            if self.classification_model not in self.inherent_models:
                return Exception

            if self.classification_model == "Decision Tree":
                return #TODO
            elif self.classification_model == "Explainable Boosting Machine":
                return #TODO
        elif self.global_explainer == "LIME":
            lime_model = LIME_posthoc.LimeAnalysis(self.trained_model, self.X_train, self.X_test, self.y_test, self.processor.get_feature_names(), self.y.unique().tolist())
            return lime_model.perform_lime_analysis_instance(self.transformed_data)
        elif self.global_explainer == "Anchors":
            anchor_rf = Anchor_posthoc.AnchorAnalysis(self.trained_model, self.X_train, self.X_test, self.y_test, self.processor.get_feature_names(), self.y.unique().tolist())
            return anchor_rf.perform_anchor_analysis_instance(self.transformed_data)