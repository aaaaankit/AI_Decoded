import pandas as pd
import sys
import os
from pathlib import Path

root = Path(__file__).parent
cwd = Path(__file__).parent
path_Data_processing = cwd / 'Data Processing'
path_Random_forest = cwd / 'Classification Models' / 'Uninterpretable Models'

path_Decision_tree = cwd / 'Classification Models'

print(path_Decision_tree)

sys.path
sys.path.append(str(path_Data_processing))
sys.path.append(str(path_Random_forest))
sys.path.append(str(path_Decision_tree))

import DataProcessor
import Randomforest
import DecisionTree # TODO WHY IMPORT NOT WORKING

#----------------------------------------------------------------------------------------------------------------------------------------

#df = pd.read_csv('Dataset/cox-violent-parsed_filt_processed.csv')
df = pd.read_csv('Dataset/cox-violent-parsed_filt.csv')
df = df.dropna(subset=["score_text"])


relevant = ["sex","age","race","juv_fel_count","juv_misd_count","juv_other_count",
            "c_charge_degree","r_charge_degree","r_days_from_arrest",
           "is_recid","vr_charge_degree","event"]


#  "is_violent_recid",

target = "score_text"


class_names = ["Low", "Medium", "High"]
class_names = ["0", "1", "2"]

#----------------------------------------------------------------------------------------------------------------------------------------

y = df[target]
df.drop(target, axis=1, inplace=True)
#processor_processed = DataProcessor.DataProcessor(df, y, relevant, normalizer_enabled=False, encoder_enabled=True, imputer_enabled=True)
processor_raw = DataProcessor.DataProcessor(df, y, relevant, normalizer_enabled=True, encoder_enabled=True, imputer_enabled=True)
(X_train, X_test, y_train, y_test) = processor_raw.process_data()

#----------------------------------------------------------------------------------------------------------------------------------------

rf = Randomforest.RandomForestTrainer(X_train, y_train, X_test, y_test, model_path="Test", evaluation_results="Eval")

rf.train_random_forest()
rf.evaluate_random_forest()

#----------------------------------------------------------------------------------------------------------------------------------------

dt = DecisionTree.DecisionTreeTrainer(X_train, y_train, X_test, y_test, model_path="Test", evaluation_results="Eval")

dt.train_decision_tree()
dt.evaluate_decision_tree()

#----------------------------------------------------------------------------------------------------------------------------------------
