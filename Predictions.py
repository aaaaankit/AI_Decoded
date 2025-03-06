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
import MLPClassifier
import SHAP_posthoc
import Anchor_posthoc
import LIME_posthoc
#import Vizualize_tree
#from Vizualize_tree import VizTree




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
one_hot_columns = ['race']


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

ebm = DecisionTree.DecisionTreeTrainer(X_train, y_train, X_test, y_test, model_path="Classification Models/Saved Models/Test_ExplainableBoosting")
ebm.load_decision_tree()
trained_ebm = ebm.get_model()

#----------------------------------------------------------------------------------------------------------------------------------------

nn = MLPClassifier.NeuralNetworkTrainer(X_train, y_train, X_test, y_test, model_path="Classification Models/Saved Models/Test_NeuralNet")
nn.load_neural_network()
nn.evaluate_neural_network()



# Local (prediction level explanations) Inherent
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
inherent_ebm = trained_ebm.explain_local(X_test[:5], y_test[:5])


# Local (prediction level explanations) LIME
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
lime_rf = LIME_posthoc.LimeAnalysis(trained_rf, X_train, X_test, y_test, processor.get_feature_names(), y.unique().tolist())
lime_rf.perform_lime_analysis(5)

lime_dt = LIME_posthoc.LimeAnalysis(trained_dt, X_train, X_test, y_test, processor.get_feature_names(), y.unique().tolist())
lime_dt.perform_lime_analysis(5)

lime_nn = LIME_posthoc.LimeAnalysis(nn, X_train, X_test, y_test, processor.get_feature_names(), y.unique().tolist())       #TODO
lime_nn.perform_lime_analysis(5)



# Local (prediction level explanations) Anchors
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
anchor_rf = Anchor_posthoc.AnchorAnalysis(trained_rf, X_train, X_test, y_test, processor.get_feature_names(), y.unique().tolist())
anchor_rf.perform_anchor_analysis(5)

anchor_dt = Anchor_posthoc.AnchorAnalysis(trained_dt, X_train, X_test, y_test, processor.get_feature_names(), y.unique().tolist())
anchor_dt.perform_anchor_analysis(5)

anchor_nn = Anchor_posthoc.AnchorAnalysis(nn, X_train, X_test, y_test, processor.get_feature_names(), y.unique().tolist())
anchor_nn.perform_anchor_analysis(5)






def read_json(json_string):
    data_point = json.loads(json_string)
    return data_point
