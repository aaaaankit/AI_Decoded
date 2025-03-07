import pandas as pd
import sys
from pathlib import Path

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
    def __init__(self, classification_model, local_explainer, data_point):
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
        self.inherent_models = ["Decision Tree", "Explainable Boosting Machine"]

        df = pd.read_csv('AI_Decoded/Dataset/cox-violent-parsed_filt.csv')
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
            y=self.y,
            relevant_columns=relevant,
            onehot_cols=one_hot_columns,
        )

        self.processor.load_pipeline("AI_Decoded/Data Processing/DataTransformer.pkl")

        self.X_train, self.X_test, self.y_train, self.y_test = self.processor.split_data()
        new_data_df = pd.DataFrame([data_point])
        self.transformed_data = self.processor.pipeline.transform(new_data_df).reshape(-1)



    def classification(self):
        model_classes = {
            "Multi-layered Perceptron": MLPClassifier.NeuralNetworkTrainer,
            "RandomForest": Randomforest.RandomForestTrainer,
            "Explainable Boosting Machine": Explainable_Boosting_Machines.ExplainableBoostingTrainer,
            "Decision Tree": DecisionTree.DecisionTreeTrainer,
        }

        model_paths = {
            "Multi-layered Perceptron": "AI_Decoded/Classification Models/Saved Models/Test_NeuralNet",
            "RandomForest": "AI_Decoded/Classification Models/Saved Models/Test_RandomForest",
            "Explainable Boosting Machine": "AI_Decoded/Classification Models/Saved Models/Test_ExplainableBoosting",
            "Decision Tree": "AI_Decoded/Classification Models/Saved Models/Test_DecisionTree",
        }

        if self.classification_model in model_classes:
            model = model_classes[self.classification_model](self.X_train, self.y_train, self.X_test, self.y_test, model_path=model_paths[self.classification_model])
            load_methods = {
                "Multi-layered Perceptron": model.load_neural_network,
                "RandomForest": model.load_random_forest,
                "Explainable Boosting Machine": model.load_ebm,
                "Decision Tree": model.load_decision_tree,
            }
            load_methods[self.classification_model]()

            self.trained_model = model.get_model()
            return self.trained_model.predict(self.transformed_data)


    def local_explanation(self):
        explainer_classes = {
            "SHAP": lambda: SHAP_posthoc.SHAPAnalysis(self.trained_model, self.X_train, self.X_test, self.y_test, self.processor.get_feature_names(), "AI_Decoded/Model Explanations/Post-Hoc Analysis/Post-Hoc Analysis Results").perform_shap_local_explanation_instance(self.transformed_data),
            "LIME": lambda: LIME_posthoc.LimeAnalysis(self.trained_model, self.X_train, self.X_test, self.y_test, self.processor.get_feature_names(), self.y.unique().tolist()).perform_lime_analysis_instance(self.transformed_data),
            "Anchors": lambda: Anchor_posthoc.AnchorAnalysis(self.trained_model, self.X_train, self.X_test, self.y_test, self.processor.get_feature_names(), self.y.unique().tolist()).perform_anchor_analysis_instance(self.transformed_data),
        }

        if self.global_explainer == "Inherent" and self.classification_model not in self.inherent_models:
            return Exception

        return explainer_classes.get(self.global_explainer, lambda: None)()