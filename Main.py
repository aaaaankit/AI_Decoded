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
#import Vizualize_tree
#from Vizualize_tree import VizTree


#----------------------------------------------------------------------------------------------------------------------------------------

#df = pd.read_csv('Dataset/cox-violent-parsed_filt_processed.csv')
df = pd.read_csv('Dataset/cox-violent-parsed_filt.csv')


df = df.dropna(subset=["score_text"])

# replace African-American with African American in the race column
df['race'] = df['race'].str.replace('African-American', 'African American')

relevant = ["sex","age","race","juv_fel_count","juv_misd_count","juv_other_count",
            "c_charge_degree","r_charge_degree","r_days_from_arrest",
           "is_recid","vr_charge_degree"]

target = "score_text"


class_names = ["Low", "Medium", "High"]
class_names = ["0", "1", "2"]

#----------------------------------------------------------------------------------------------------------------------------------------

y = df[target]

df.drop(target, axis=1, inplace=True)
#processor_processed = DataProcessor.DataProcessor(df, y, relevant, normalizer_enabled=False, encoder_enabled=False, imputer_enabled=False)
processor_raw = DataProcessor.DataProcessor(df, y, relevant, normalizer_enabled=True, encoder_enabled=True, imputer_enabled=True)
(X_train, X_test, y_train, y_test) = processor_raw.process_data()


#----------------------------------------------------------------------------------------------------------------------------------------

rf = Randomforest.RandomForestTrainer(X_train, y_train, X_test, y_test, model_path="Test", evaluation_results="Eval")

rf.train_random_forest()
rf.evaluate_random_forest()
trained_rf = rf.get_model()

shap_rf = SHAP_posthoc.SHAPAnalysis(trained_rf, X_train, X_test,y_test, "evaluation_results")
shap_rf.perform_shap_analysis()


#----------------------------------------------------------------------------------------------------------------------------------------

dt = DecisionTree.DecisionTreeTrainer(X_train, y_train, X_test, y_test, model_path="Test", evaluation_results="Eval")

dt.train_decision_tree()
dt.evaluate_decision_tree()
trained_dt = dt.get_model()

#viz_tree = VizTree(trained_dt, X_train, X_test)
#viz_tree.visualize_tree(output_path="Inherent_eval_result/tree_visualization")
#viz_tree.feature_importance()
#viz_tree.export_rules()
#viz_tree.explain_instance(instance_index=0)

shap_dt = SHAP_posthoc.SHAPAnalysis(trained_dt, X_train, X_test,y_test, "evaluation_results")
shap_dt.perform_shap_analysis()