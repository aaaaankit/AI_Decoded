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

df = pd.read_csv('Dataset/propublica_data_for_fairml.csv')


relevant = ["sex","age","race","juv_fel_count","juv_misd_count","juv_other_count",
            "priors_count","days_b_screening_arrest","c_jail_in","c_jail_out","c_days_from_compas",
            "c_charge_degree","c_charge_desc","event"]

relevant = df.columns.to_list()
target = "Two_yr_Recidivism"
relevant.remove(target)  


class_names = ["Negative", "Neutral", "Positive"]
class_names = ["0", "1"]

#----------------------------------------------------------------------------------------------------------------------------------------

y = df[target]
df.drop(target, axis=1, inplace=True)
processor_simplified = DataProcessor.DataProcessor(df, y, relevant, normalizer_enabled=False, encoder_enabled=False, imputer_enabled=False)
(X_train, X_test, y_train, y_test) = processor_simplified.process_data()

#----------------------------------------------------------------------------------------------------------------------------------------

rf = Randomforest.RandomForestTrainer(X_train, y_train, X_test, y_test, model_path="Test", evaluation_results_path="Eval")

rf.train_random_forest()
rf.evaluate_random_forest()

#----------------------------------------------------------------------------------------------------------------------------------------

# dt = DecisionTree.DecisionTreeTrainer(X_train, y_train, X_test, y_test, model_path="Test", evaluation_results_path="Eval")

# dt.train_random_forest()
# dt.evaluate_random_forest()

#----------------------------------------------------------------------------------------------------------------------------------------
