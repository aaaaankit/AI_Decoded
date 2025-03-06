from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())
from interpret import show

# Add paths to import custom modules
root = Path(__file__).parent
cwd = Path(__file__).parent
path_Data_processing = cwd / 'Data Processing'
path_Random_forest = cwd / 'Classification Models' / 'Uninterpretable Models'
path_Decision_tree = cwd / 'Classification Models' / 'Interpretable Models'
path_post_hoc = cwd / 'Model Explanations' / 'Post-Hoc Analysis'
path_inherent = cwd / 'Model Explanations' / 'Inherently Interpretable Analysis'

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
import SHAP_posthoc
import MLPClassifier
import Anchor_posthoc




# Load dataset
df = pd.read_csv('Dataset/cox-violent-parsed_filt.csv')
df = df.dropna(subset=["score_text"])

# Preprocessing: Replace 'African-American' with 'African American' in the 'race' column
df['race'] = df['race'].str.replace('African-American', 'African American')

# Define relevant columns and target
relevant = ["sex","age","race","juv_fel_count","juv_misd_count","juv_other_count",
            "c_charge_degree","r_charge_degree","r_days_from_arrest","vr_charge_degree"]
target = "score_text"

# Define columns to One-Hot Encode
one_hot_columns = ['race', 'sex']

y = df[target]


# Data transformer setup
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
processor = DataTransformer.DataTransformer(
    df=df,
    y=y,
    relevant_columns=relevant,
    onehot_cols=one_hot_columns,
)

processor.fit_pipeline()
processor.save_pipeline("Data Processing/DataTransformer.pkl")
X_train, X_test, y_train, y_test = processor.split_data()



# Model saving
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
rf = Randomforest.RandomForestTrainer(X_train, y_train, X_test, y_test, model_path="Test_RandomForest", evaluation_results="Eval")

rf.train_random_forest()
rf.evaluate_random_forest()
trained_rf = rf.get_model()
rf.save_random_forest()


##----------------------------------------------------------------------------------------------------------------------------------------

dt = DecisionTree.DecisionTreeTrainer(X_train, y_train, X_test, y_test, model_path="Classification Models/Saved Models/Test_DecisionTree", evaluation_results="Eval")

dt.train_decision_tree()
dt.evaluate_decision_tree()
trained_dt = dt.get_model()
dt.save_decision_tree()

##-----------------------------------------------------------------------------------------------------------------------------------------

ebm = Explainable_Boosting_Machines.ExplainableBoostingTrainer(X_train, y_train, X_test, y_test, model_path="Classification Models/Saved Models/Test_ExplainableBoosting", evaluation_results="Eval")

ebm.train_ebm()
ebm.evaluate_ebm()
trained_ebm = ebm.get_model()
ebm.save_ebm()

##----------------------------------------------------------------------------------------------------------------------------------------

nn = MLPClassifier.NeuralNetworkTrainer(X_train, y_train, X_test, y_test, model_path="Classification Models/Saved Models/Test_NeuralNet", evaluation_results="Eval")

nn.train_neural_network()
nn.evaluate_neural_network()
trained_nn = nn.get_model()
nn.save_neural_network()



# General Model explanations (Inherent)
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
trained_ebm.explain_global()


# General Model explanations (SHAP)
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
shap_rf = SHAP_posthoc.SHAPAnalysis(trained_rf, X_train, X_test, y_test, processor.get_feature_names(), "Model Explanations/Post-Hoc Analysis/Post-Hoc Analysis Results")
shap_rf.perform_shap_general_explanation() 

shap_dt = SHAP_posthoc.SHAPAnalysis(trained_dt, X_train, X_test, y_test, processor.get_feature_names(), "Model Explanations/Post-Hoc Analysis/Post-Hoc Analysis Results")
shap_dt.perform_shap_general_explanation()

shap_mlp = SHAP_posthoc.SHAPAnalysis(trained_nn, X_train, X_test,y_test, "Model Explanations/Post-Hoc Analysis/Post-Hoc Analysis Results")
shap_mlp.perform_shap_general_explanation()




















#data_point = {
#    "sex": "Male",
#    "age": 27,
#    "age_cat": "25 - 45",
#    "race": "Caucasian",
#    "juv_fel_count": 1,
#    "decile_score": 5,
#    "juv_misd_count": 2,
#    "juv_other_count": 0,
#    "priors_count": 4,
#    "days_b_screening_arrest": 10,
#    "c_jail_in": "01/01/2015 12:00",
#    "c_jail_out": "01/01/2015 14:00",
#    "c_days_from_compas": 20,
#    "c_charge_degree": "(F1)",
#    "c_charge_desc": "Battery",
#    "is_recid": 0,
#    "r_charge_degree": "(F2)",
#    "r_days_from_arrest": 15,
#    "r_offense_date": "10/02/2015",
#    "r_jail_in": "10/02/2015 12:00",
#    "violent_recid": 0,
#    "is_violent_recid": 0,
#    "vr_charge_degree": "(M1)",
#    "vr_offense_date": "11/02/2015",
#    "score_text": "Battery",
#    "screening_date": "01/01/2015",
#    "v_decile_score": 4,
#    "v_score_text": "Medium",
#    "event": 0
#}
#
## Apply the transformation to the new data point
#new_data_df = pd.DataFrame([data_point])
#transformed_data = processor.pipeline.transform(new_data_df)
#
## Output the transformed data
#print("Transformed Data:")
#print(transformed_data)
#
#prediction = model.predict(transformed_data)
#
## Output the prediction
#print(f"Prediction for the new data point: {prediction[0]}")
#anchor_rf.perform_anchor_analysis_instance(transformed_data)




