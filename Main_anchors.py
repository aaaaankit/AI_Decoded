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
import Randomforest
import DecisionTree # TODO WHY IMPORT NOT WORKING 
import SHAP_posthoc
import Anchor_posthoc



#----------------------------------------------------------------------------------------------------------------------------------------
df = pd.read_csv('Dataset/cox-violent-parsed_filt_processed2.csv')

df = df.dropna(subset=["score_text"])

class_names = ["Low", "Medium", "High"]

#----------------------------------------------------------------------------------------------------------------------------------------

data = df[df.columns.to_list()[:-1]].to_numpy()
target = df[df.columns.to_list()[-1]].to_numpy()
feature_names = df.columns.to_list()[:-1]
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=7)
anchor_test_instance_index = 133

rf = Randomforest.RandomForestTrainer(X_train, y_train, X_test, y_test, model_path="Test", evaluation_results="Eval")

rf.train_random_forest()
rf.evaluate_random_forest()
trained_rf = rf.get_model()

anchor_rf = Anchor_posthoc.AnchorAnalysis(trained_rf, X_train, X_test, y_test, feature_names, class_names)
anchor_rf.perform_anchor_analysis(anchor_test_instance_index)


#----------------------------------------------------------------------------------------------------------------------------------------

dt = DecisionTree.DecisionTreeTrainer(X_train, y_train, X_test, y_test, model_path="Test", evaluation_results="Eval")

dt.train_decision_tree()
dt.evaluate_decision_tree()
trained_dt = dt.get_model()

anchor_rf = Anchor_posthoc.AnchorAnalysis(trained_dt, X_train, X_test, y_test, feature_names, class_names)
anchor_rf.perform_anchor_analysis(anchor_test_instance_index)