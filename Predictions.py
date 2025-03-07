import json
import pandas as pd
import sys
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

import DataTransformer
import Randomforest
import DecisionTree 
import Explainable_Boosting_Machines
import MLPClassifier
import SHAP_posthoc
import Anchor_posthoc
import LIME_posthoc
import Feature_Importance_posthoc
import Partial_Dependence_Plots_posthoc
import Permutation_Importance_posthoc
#import Vizualize_tree
#from Vizualize_tree import VizTree

def read_json(file_path):
    """
    Reads a JSON-formatted text file and parses it into a Python dictionary.

    Args:
        file_path (str): The path to the text file containing JSON data.

    Returns:
        dict: Parsed JSON data.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        json_string = file.read()  # Read the entire content of the file
        data_point = json.loads(json_string)  # Parse the JSON string
    return data_point




df = pd.read_csv('Dataset/cox-violent-parsed_filt.csv')
df = df.dropna(subset=["score_text"])
df['race'] = df['race'].str.replace('African-American', 'African American')     # replace African-American with African American in the race column

relevant = ["sex","age","race","juv_fel_count","juv_misd_count","juv_other_count",
            "c_charge_degree","r_charge_degree","r_days_from_arrest",
           "vr_charge_degree"]

target = "score_text"

y = df[target]
df.drop(target, axis=1, inplace=True)

# Define the columns to One-Hot Encode
one_hot_columns = ['race', 'sex']


# Data transformer loading
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------

processor = DataTransformer.DataTransformer(
    df=df,
    y=y,
    relevant_columns=relevant,
    onehot_cols=one_hot_columns,
)

processor.load_pipeline("Data Processing/DataTransformer.pkl")

X_train, X_test, y_train, y_test = processor.split_data()



# Model loading
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
rf = Randomforest.RandomForestTrainer(X_train, y_train, X_test, y_test, model_path="Classification Models/Saved Models/Test_RandomForest")
rf.load_random_forest()
trained_rf = rf.get_model()

#----------------------------------------------------------------------------------------------------------------------------------------

dt = DecisionTree.DecisionTreeTrainer(X_train, y_train, X_test, y_test, model_path="Classification Models/Saved Models/Test_DecisionTree")
dt.load_decision_tree()
trained_dt = dt.get_model()

#----------------------------------------------------------------------------------------------------------------------------------------

ebm = Explainable_Boosting_Machines.ExplainableBoostingTrainer(X_train, y_train, X_test, y_test, model_path="Classification Models/Saved Models/Test_ExplainableBoosting")
ebm.load_ebm()
trained_ebm = ebm.get_model()

#----------------------------------------------------------------------------------------------------------------------------------------

nn = MLPClassifier.NeuralNetworkTrainer(X_train, y_train, X_test, y_test, model_path="Classification Models/Saved Models/Test_NeuralNet")
nn.load_neural_network()
trained_nn = nn.get_model()



# Data entry
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
data_point = read_json('dataPoint.txt')
new_data_df = pd.DataFrame([data_point])
transformed_data = processor.pipeline.transform(new_data_df)
transformed_data = transformed_data.reshape(-1)
#print(X_test[2].shape)
#print(transformed_data.shape)



# Local (prediction level explanations) Inherent
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
inherent_ebm = trained_ebm.local_explanation(transformed_data)


# Local (prediction level explanations) LIME
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
lime_rf = LIME_posthoc.LimeAnalysis(trained_rf, X_train, X_test, y_test, processor.get_feature_names(), y.unique().tolist())
lime_rf.perform_lime_analysis_instance(transformed_data)

lime_dt = LIME_posthoc.LimeAnalysis(trained_dt, X_train, X_test, y_test, processor.get_feature_names(), y.unique().tolist())
lime_dt.perform_lime_analysis_instance(transformed_data)

lime_ebm = LIME_posthoc.LimeAnalysis(trained_ebm, X_train, X_test, y_test, processor.get_feature_names(), y.unique().tolist())
lime_ebm.perform_lime_analysis_instance(transformed_data)

lime_nn = LIME_posthoc.LimeAnalysis(trained_nn, X_train, X_test, y_test, processor.get_feature_names(), y.unique().tolist())       #TODO
lime_nn.perform_lime_analysis_instance(transformed_data)


# Local (prediction level explanations) SHAP
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
shap_rf = SHAP_posthoc.SHAPAnalysis(trained_rf, X_train, X_test, y_test, processor.get_feature_names(), "RandomForest")
shap_rf.perform_shap_local_explanation_instance(transformed_data)

shap_dt = SHAP_posthoc.SHAPAnalysis(trained_dt, X_train, X_test, y_test, processor.get_feature_names(), "Decision Tree")
shap_dt.perform_shap_local_explanation_instance(transformed_data)

shap_ebm = SHAP_posthoc.SHAPAnalysis(trained_ebm, X_train, X_test, y_test, processor.get_feature_names(), "Explainable Boosted Machine")
shap_ebm.perform_shap_local_explanation_instance(transformed_data)

shap_nn = SHAP_posthoc.SHAPAnalysis(trained_nn, X_train, X_test, y_test, processor.get_feature_names(), "MLP NN")       #TODO
shap_nn.perform_shap_local_explanation_instance(transformed_data)


# Local (prediction level explanations) Anchors
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
anchor_rf = Anchor_posthoc.AnchorAnalysis(trained_rf, X_train, X_test, y_test, processor.get_feature_names(), y.unique().tolist())
anchor_rf.perform_anchor_analysis_instance(transformed_data)

anchor_dt = Anchor_posthoc.AnchorAnalysis(trained_dt, X_train, X_test, y_test, processor.get_feature_names(), y.unique().tolist())
anchor_dt.perform_anchor_analysis_instance(transformed_data)

anchor_ebm = Anchor_posthoc.AnchorAnalysis(trained_ebm, X_train, X_test, y_test, processor.get_feature_names(), y.unique().tolist())
anchor_ebm.perform_anchor_analysis_instance(transformed_data)

anchor_nn = Anchor_posthoc.AnchorAnalysis(trained_nn, X_train, X_test, y_test, processor.get_feature_names(), y.unique().tolist())
anchor_nn.perform_anchor_analysis_instance(transformed_data)


# Global (model level explanations) Feature Importance
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
feature_importance_rf = Feature_Importance_posthoc.FeatureImportanceAnalysis(trained_rf, X_train, X_test, y_test, processor.get_feature_names())
feature_importance_rf.explain_instance(transformed_data)

feature_importance_dt = Feature_Importance_posthoc.FeatureImportanceAnalysis(trained_dt, X_train, X_test, y_test, processor.get_feature_names())
feature_importance_dt.explain_instance(transformed_data)

feature_importance_ebm = Feature_Importance_posthoc.FeatureImportanceAnalysis(trained_ebm, X_train, X_test, y_test, processor.get_feature_names())
feature_importance_ebm.explain_instance(transformed_data)

feature_importance_nn = Feature_Importance_posthoc.FeatureImportanceAnalysis(trained_nn, X_train, X_test, y_test, processor.get_feature_names())
feature_importance_nn.explain_instance(transformed_data)

# Global (model level explanations) Partial Dependence Plots
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
# partial_dependence_rf = Partial_Dependence_Plots_posthoc.PartialDependencePlotsAnalysis(trained_rf, X_train, X_test, y_test, processor.get_feature_names())
# partial_dependence_rf.explain_instance(transformed_data, processor.get_feature_names(), target)

# partial_dependence_dt = Partial_Dependence_Plots_posthoc.PartialDependencePlotsAnalysis(trained_dt, X_train, X_test, y_test, processor.get_feature_names())
# partial_dependence_dt.explain_instance(transformed_data, processor.get_feature_names(), target)

# partial_dependence_ebm = Partial_Dependence_Plots_posthoc.PartialDependencePlotsAnalysis(trained_ebm, X_train, X_test, y_test, processor.get_feature_names())
# partial_dependence_ebm.explain_instance(transformed_data, processor.get_feature_names(), target)

# partial_dependence_nn = Partial_Dependence_Plots_posthoc.PartialDependencePlotsAnalysis(trained_nn, X_train, X_test, y_test, processor.get_feature_names())
# partial_dependence_nn.explain_instance(transformed_data, processor.get_feature_names(), target)

# Global (model level explanations) Permutation Importance
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
permutation_importance_rf = Permutation_Importance_posthoc.PermutationImportanceAnalysis(trained_rf, X_train, X_test, y_test, processor.get_feature_names())
permutation_importance_rf.explain_instance(transformed_data)

permutation_importance_dt = Permutation_Importance_posthoc.PermutationImportanceAnalysis(trained_dt, X_train, X_test, y_test, processor.get_feature_names())
permutation_importance_dt.explain_instance(transformed_data)

permutation_importance_ebm = Permutation_Importance_posthoc.PermutationImportanceAnalysis(trained_ebm, X_train, X_test, y_test, processor.get_feature_names())
permutation_importance_ebm.explain_instance(transformed_data)

permutation_importance_nn = Permutation_Importance_posthoc.PermutationImportanceAnalysis(trained_nn, X_train, X_test, y_test, processor.get_feature_names())
permutation_importance_nn.explain_instance(transformed_data)

