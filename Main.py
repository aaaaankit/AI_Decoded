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

path_Decision_tree = cwd / 'Classification Models'

path_post_hoc = cwd / 'Post-Hoc Analysis'

path_inherent = cwd / 'Inherently_Interpretable_Analysis'

print(path_Decision_tree)

sys.path
sys.path.append(str(path_Data_processing))
sys.path.append(str(path_Random_forest))
sys.path.append(str(path_Decision_tree))
sys.path.append(str(path_post_hoc))
sys.path.append(str(path_inherent))

import DataProcessor
import DataTransformer
import Randomforest
import DecisionTree # TODO WHY IMPORT NOT WORKING 
import MLPClassifier
import SHAP_posthoc
import Anchor_posthoc
#import Vizualize_tree
#from Vizualize_tree import VizTree


# Data transformer setup
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------

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


processor = DataTransformer.DataTransformer(
    df=df,
    y=y,
    relevant_columns=relevant,
    onehot_cols=one_hot_columns,
)

# Fit and save the pipeline
processor.fit_pipeline()
processor.save_pipeline("DataTransformer.pkl")


# Data transformer loading
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------

processor2 = DataTransformer.DataTransformer(
    df=df,
    y=y,
    relevant_columns=relevant,
    onehot_cols=one_hot_columns,
)

processor2.load_pipeline("DataTransformer.pkl")

X_train, X_test, y_train, y_test = processor2.split_data()






#***************************************************************************************************************************************
#***************************************************************************************************************************************
#***************************************************************************************************************************************
#***************************************************************************************************************************************




# Model saving
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------

rf = Randomforest.RandomForestTrainer(X_train, y_train, X_test, y_test, model_path="Test", evaluation_results="Eval")

rf.train_random_forest()
rf.evaluate_random_forest()
trained_rf = rf.get_model()
rf.save_random_forest()


#----------------------------------------------------------------------------------------------------------------------------------------

dt = DecisionTree.DecisionTreeTrainer(X_train, y_train, X_test, y_test, model_path="Test", evaluation_results="Eval")

dt.train_decision_tree()
dt.evaluate_decision_tree()
trained_dt = dt.get_model()
dt.save_decision_tree()


#----------------------------------------------------------------------------------------------------------------------------------------

nn = MLPClassifier.NeuralNetworkTrainer(X_train, y_train, X_test, y_test, model_path="Test", evaluation_results="Eval")

nn.train_neural_network()
nn.evaluate_neural_network()
trained_nn = nn.get_model()
nn.save_neural_network()




# Model loading
#****************************************************************************************************************************************

rf2 = Randomforest.RandomForestTrainer(X_train, y_train, X_test, y_test, model_path="Test", evaluation_results="Eval")
rf2.load_random_forest()

#----------------------------------------------------------------------------------------------------------------------------------------

dt2 = DecisionTree.DecisionTreeTrainer(X_train, y_train, X_test, y_test, model_path="Test", evaluation_results="Eval")
dt2.load_decision_tree()

#----------------------------------------------------------------------------------------------------------------------------------------

nn2 = MLPClassifier.NeuralNetworkTrainer(X_train, y_train, X_test, y_test, model_path="Test", evaluation_results="Eval")
nn2.load_neural_network()




#***************************************************************************************************************************************
#***************************************************************************************************************************************
#***************************************************************************************************************************************
#***************************************************************************************************************************************



"""
# Model explaining (SHAP)
#****************************************************************************************************************************************

shap_rf = SHAP_posthoc.SHAPAnalysis(trained_rf, X_train, X_test,y_test, "evaluation_results")
shap_rf.perform_shap_analysis()

shap_dt = SHAP_posthoc.SHAPAnalysis(trained_dt, X_train, X_test,y_test, "evaluation_results")
shap_dt.perform_shap_analysis()

#viz_tree = VizTree(trained_dt, X_train, X_test)
#viz_tree.visualize_tree(output_path="Inherent_eval_result/tree_visualization")
#viz_tree.feature_importance()
#viz_tree.export_rules()
#viz_tree.explain_instance(instance_index=0)

shap_mlp = SHAP_posthoc.SHAPAnalysis(trained_nn, X_train, X_test,y_test, "evaluation_results")
shap_mlp.perform_shap_analysis()
"""
