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
           "vr_charge_degree"]
# "is_recid",

target = "score_text"


class_names = ["Low", "Medium", "High"]
class_names = ["0", "1", "2"]

# Custom Label Encoding mapping for 'charge_degree'
charge_degree_mapping = {
    'F2': 0, 'F3': 1, 'F5': 2, 'F6': 3, 'F7': 4, 
    'M1': 5, 'M2': 6, 'M03': 7, 'Unknown': -1
}

# Define the columns to One-Hot Encode
one_hot_columns = ['race']

#----------------------------------------------------------------------------------------------------------------------------------------

y = df[target]

df.drop(target, axis=1, inplace=True)
#processor_processed = DataProcessor.DataProcessor(df, y, relevant, normalizer_enabled=False, encoder_enabled=False, imputer_enabled=False)
processor_raw = DataProcessor.DataProcessor(df, y, relevant, normalizer_enabled=True, encoder_enabled=True, imputer_enabled=True, one_hot_columns=one_hot_columns, label_mappings=charge_degree_mapping)
(X_train, X_test, y_train, y_test) = processor_raw.process_data()

## EDA
## get the columns of X_train
#columns = X_train.columns
#print(columns)
#
#"""
#Index(['sex', 'age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
#       'r_days_from_arrest', 'is_recid', 'race_Asian', 'race_Caucasian',
#       'race_Hispanic', 'race_Native American', 'race_Other',
#       'c_charge_degree_(CT)', 'c_charge_degree_(F1)', 'c_charge_degree_(F2)',
#       'c_charge_degree_(F3)', 'c_charge_degree_(F5)', 'c_charge_degree_(F6)',
#       'c_charge_degree_(F7)', 'c_charge_degree_(M1)', 'c_charge_degree_(M2)',
#       'c_charge_degree_(MO3)', 'c_charge_degree_(NI0)',
#       'c_charge_degree_(TCX)', 'c_charge_degree_(X)',
#       'c_charge_degree_Unknown', 'r_charge_degree_(F1)',
#       'r_charge_degree_(F2)', 'r_charge_degree_(F3)', 'r_charge_degree_(F5)',
#       'r_charge_degree_(F6)', 'r_charge_degree_(F7)', 'r_charge_degree_(M1)',
#       'r_charge_degree_(M2)', 'r_charge_degree_(MO3)',
#       'r_charge_degree_Unknown', 'vr_charge_degree_(F2)',
#       'vr_charge_degree_(F3)', 'vr_charge_degree_(F5)',
#       'vr_charge_degree_(F6)', 'vr_charge_degree_(F7)',
#       'vr_charge_degree_(M1)', 'vr_charge_degree_(M2)',
#       'vr_charge_degree_(MO3)', 'vr_charge_degree_Unknown'],
#      dtype='object')
#"""
#
#X_train_reduced_race = X_train[['race_Asian', 'race_Caucasian', 'race_Hispanic', 'race_Native American', 'race_Other']]
#
##X_train_reduced_c_charge_degree = X_train[['c_charge_degree_(CT)', 'c_charge_degree_(F1)', 'c_charge_degree_(F2)',
##       'c_charge_degree_(F3)', 'c_charge_degree_(F5)', 'c_charge_degree_(F6)',
##       'c_charge_degree_(F7)', 'c_charge_degree_(M1)', 'c_charge_degree_(M2)',
##       'c_charge_degree_(MO3)', 'c_charge_degree_(NI0)',
##       'c_charge_degree_(TCX)', 'c_charge_degree_(X)',
##       'c_charge_degree_Unknown']]
##
##X_train_reduced_r_charge_degree = X_train[['r_charge_degree_(F1)',
##       'r_charge_degree_(F2)', 'r_charge_degree_(F3)', 'r_charge_degree_(F5)',
##       'r_charge_degree_(F6)', 'r_charge_degree_(F7)', 'r_charge_degree_(M1)',
##       'r_charge_degree_(M2)', 'r_charge_degree_(MO3)',
##       'r_charge_degree_Unknown']]
##
##X_train_reduced_vr_charge_degree = X_train[['vr_charge_degree_(F2)',
##       'vr_charge_degree_(F3)', 'vr_charge_degree_(F5)',
##       'vr_charge_degree_(F6)', 'vr_charge_degree_(F7)',
##       'vr_charge_degree_(M1)', 'vr_charge_degree_(M2)',
##       'vr_charge_degree_(MO3)', 'vr_charge_degree_Unknown']]
#
#
## List of DataFrames and corresponding titles
#dataframes = [
#    (X_train, "Correlation Matrix X_train"),
#    (X_train_reduced_race, "Correlation Matrix X_train_reduced_race"),
#    #(X_train_reduced_c_charge_degree, "Correlation Matrix X_train_reduced_c_charge_degree"),
#    #(X_train_reduced_r_charge_degree, "Correlation Matrix X_train_reduced_r_charge_degree"),
#    #(X_train_reduced_vr_charge_degree, "Correlation Matrix X_train_reduced_vr_charge_degree")
#]
#
## Loop through each DataFrame, compute the correlation matrix, plot, and save the figure
#for df, title in dataframes:
#    corr_matrix = df.corr()
#
#    plt.figure(figsize=(12, 8))
#    if title != "Correlation Matrix X_train":
#        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
#    else:
#        sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", fmt=".2f", linewidths=0.5)
#    
#    plt.title(title)
#
#    # Save the figure as a PNG file
#    plt.savefig(f"{title}.png", bbox_inches='tight')
#
#    # Show the plot (optional)
#    plt.show()



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
