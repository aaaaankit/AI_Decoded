import pickle
import pandas as pd
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

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

import Vizualize_tree
import Vizualize_ebm

def transform_and_save(input_data, output_file_path):
    # Step 1: Load the transformer (pickled pipeline)
    with open('Data Processing/DataTransformer.pkl', 'rb') as f:
        transformer = pickle.load(f)
    
    # Step 2: Convert input data to DataFrame
    # Input data is expected to be a dictionary, converting it to a DataFrame
    new_data_df = pd.DataFrame([input_data])
    
    # Step 3: Apply the transformation using the loaded transformer
    transformed_data = transformer.transform(new_data_df)
    
    # Step 4: Reshape the transformed data (if needed)
    transformed_data = transformed_data.reshape(-1)
    
    # Step 5: Prepare the transformed data as a dictionary
    # You can map the output to column names to create the desired dictionary format
    transformed_dict = {
        "age": transformed_data[0],
        "juv_fel_count": transformed_data[1],
        "juv_misd_count": transformed_data[2],
        "juv_other_count": transformed_data[3],
        "r_days_from_arrest": transformed_data[4],
        "race_African American": transformed_data[5],
        "race_Asian": transformed_data[6],
        "race_Caucasian": transformed_data[7],
        "race_Hispanic": transformed_data[8],
        "race_Native American": transformed_data[9],
        "race_Other": transformed_data[10],
        "sex_Female": transformed_data[11],
        "sex_Male": transformed_data[12],
        "c_charge_degree": transformed_data[13],
        "r_charge_degree": transformed_data[14],
        "vr_charge_degree": transformed_data[15]
    }
    
    # Step 6: Save the transformed data to a new text file
    with open(output_file_path, 'w') as f:
        f.write(str(transformed_dict))
    return transformed_dict


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

        transformed_dataset = True 
        df = pd.read_csv('Dataset/transformed_dataset.csv')
        df = df.dropna(subset=["score_text"])
           

        if not transformed_dataset:
            df['race'] = df['race'].str.replace('African-American', 'African American')  
            relevant = ["sex","age","race","juv_fel_count","juv_misd_count","juv_other_count",
                        "c_charge_degree","r_charge_degree","r_days_from_arrest",
                       "vr_charge_degree"]

            

            one_hot_columns = ['race', 'sex']

        target = "score_text"
        self.y = df[target]
        df.drop(target, axis=1, inplace=True)

        if not transformed_dataset:
            self.processor = DataTransformer.DataTransformer(
                df=df,
                y=self.y,
                relevant_columns=relevant,
                onehot_cols=one_hot_columns,
            )

            self.processor.load_pipeline("Data Processing/DataTransformer.pkl")
            self.X_train, self.X_test, self.y_train, self.y_test = self.processor.split_data()
            self.feature_names = self.processor.get_feature_names()
        else:
            self.feature_names = df.columns.tolist()
            X_train, X_test, y_train, y_test = train_test_split(df, self.y, test_size=0.2, random_state=42)
            self.X_train, self.X_test, self.y_train, self.y_test = X_train.values, X_test.values, y_train.values, y_test.values

        data_point = transform_and_save(data_point, 'transformed_data.txt')
        new_data_df = pd.DataFrame([data_point])
        print(new_data_df)

        if transformed_dataset:  # Assuming is_data_transformed is a function you can define
            self.transformed_data = new_data_df.values.reshape(-1)  # No transformation applied, just reshape
        else:
            self.transformed_data = self.processor.pipeline.transform(new_data_df)
            self.transformed_data = self.transformed_data.reshape(-1)
        print(self.transformed_data)
        

        model_classes = {
            "Multi-layered Perceptron": MLPClassifier.NeuralNetworkTrainer,
            "RandomForest": Randomforest.RandomForestTrainer,
            "Explainable Boosting Machine": Explainable_Boosting_Machines.ExplainableBoostingTrainer,
            "Decision Tree": DecisionTree.DecisionTreeTrainer,
        }

        model_paths = {
            "Multi-layered Perceptron": "Classification Models/Saved Models/Test_NeuralNet",
            "RandomForest": "Classification Models/Saved Models/Test_RandomForest",
            "Explainable Boosting Machine": "Classification Models/Saved Models/Test_ExplainableBoosting",
            "Decision Tree": "Classification Models/Saved Models/Test_DecisionTree",
        }

        if self.classification_model in model_classes:
            if self.classification_model == "Explainable Boosting Machine":
                model = model_classes[self.classification_model](self.X_train, self.y_train, self.X_test, self.y_test, self.feature_names, model_path=model_paths[self.classification_model])
            elif self.classification_model == "Decision Tree":
                model = model_classes[self.classification_model](self.X_train, self.y_train, self.X_test, self.y_test, model_path=model_paths[self.classification_model], max_depth=3)
            else:
                model = model_classes[self.classification_model](self.X_train, self.y_train, self.X_test, self.y_test, model_path=model_paths[self.classification_model])
            if (self.classification_model == "Multi-layered Perceptron"):
                model.load_neural_network()
            elif (self.classification_model == "RandomForest"):
                model.load_random_forest()
            elif (self.classification_model == "Explainable Boosting Machine"):
                model.load_ebm()
            else:
                model.load_decision_tree()

            self.trained_model = model.get_model()

    def classification(self):
        print(self.trained_model.predict([self.transformed_data]))
        return self.trained_model.predict([self.transformed_data])

    def local_explanation(self):
        if self.classification_model == "Explainable Boosting Machine":
            print("EBM")
            explainer_classes = {
                "SHAP": lambda: SHAP_posthoc.SHAPAnalysis(self.trained_model, self.X_train, self.X_test, self.y_test, self.feature_names, "De").perform_shap_local_explanation_instance(self.transformed_data),
                "LIME": lambda: LIME_posthoc.LimeAnalysis(self.trained_model, self.X_train, self.X_test, self.y_test, self.feature_names, self.y.unique().tolist()).perform_lime_analysis_instance(self.transformed_data),
                "Anchors": lambda: Anchor_posthoc.AnchorAnalysis(self.trained_model, self.X_train, self.X_test, self.y_test, self.feature_names, self.y.unique().tolist()).perform_anchor_analysis_instance(self.transformed_data),

                # for EBM
                "Textual": lambda: Vizualize_ebm.VizEBM(self.trained_model, self.X_train, self.X_test, self.y_test, self.feature_names, self.y.unique().tolist()).local_explanation_text(self.transformed_data),
                "Visualize": lambda: Vizualize_ebm.VizEBM(self.trained_model, self.X_train, self.X_test, self.y_test, self.feature_names, self.y.unique().tolist()).local_explanation_visual(self.transformed_data),
                "Feature Importance": lambda: Vizualize_ebm.VizEBM(self.trained_model, self.X_train, self.X_test, self.y_test, self.feature_names, self.y.unique().tolist()).local_explanation_importance(self.transformed_data),
            }
        
        elif self.classification_model == "Decision Tree":
            print("Decision Tree")
            print(self.transformed_data)
            explainer_classes = {
                "SHAP": lambda: SHAP_posthoc.SHAPAnalysis(self.trained_model, self.X_train, self.X_test, self.y_test, self.feature_names, "De").perform_shap_local_explanation_instance(self.transformed_data),
                "LIME": lambda: LIME_posthoc.LimeAnalysis(self.trained_model, self.X_train, self.X_test, self.y_test, self.feature_names, self.y.unique().tolist()).perform_lime_analysis_instance(self.transformed_data),
                "Anchors": lambda: Anchor_posthoc.AnchorAnalysis(self.trained_model, self.X_train, self.X_test, self.y_test, self.feature_names, self.y.unique().tolist()).perform_anchor_analysis_instance(self.transformed_data),

                # for Decision Tree
                "Textual": lambda: Vizualize_tree.VizTree(self.trained_model, self.X_train, self.X_test, self.y_test, self.feature_names, self.y.unique().tolist()).local_explanation_text(self.transformed_data),
                "Visualize": lambda: Vizualize_tree.VizTree(self.trained_model, self.X_train, self.X_test, self.y_test, self.feature_names, self.y.unique().tolist()).local_explanation_visual(self.transformed_data),
                "Feature Importance": lambda: Vizualize_tree.VizTree(self.trained_model, self.X_train, self.X_test, self.y_test, self.feature_names, self.y.unique().tolist()).local_explanation_importance(self.transformed_data),
            }
        else:
            print("Other")
            explainer_classes = {
                "SHAP": lambda: SHAP_posthoc.SHAPAnalysis(self.trained_model, self.X_train, self.X_test, self.y_test, self.feature_names, "De").perform_shap_local_explanation_instance(self.transformed_data),
                "LIME": lambda: LIME_posthoc.LimeAnalysis(self.trained_model, self.X_train, self.X_test, self.y_test, self.feature_names, self.y.unique().tolist()).perform_lime_analysis_instance(self.transformed_data),
                "Anchors": lambda: Anchor_posthoc.AnchorAnalysis(self.trained_model, self.X_train, self.X_test, self.y_test, self.feature_names, self.y.unique().tolist()).perform_anchor_analysis_instance(self.transformed_data),
            }
        inherent_list = ["Textual", "Visualize", "Feature Importance"]
        if self.local_explainer in inherent_list and self.classification_model not in self.inherent_models:
            raise ValueError(f"Inherent explanation is not supported for model: {self.classification_model}")

        if self.local_explainer not in explainer_classes:
            raise ValueError(f"Unknown local explainer: {self.local_explainer}")

        return explainer_classes[self.local_explainer]()