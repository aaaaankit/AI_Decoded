import pandas as pd
import sys
from pathlib import Path
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())

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

import DataTransformer
import Randomforest
import DecisionTree
import Explainable_Boosting_Machines
import SHAP_posthoc
import MLPClassifier




# Data Loading
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
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
rf = Randomforest.RandomForestTrainer(X_train, y_train, X_test, y_test, model_path="Classification Models/Saved Models/Test_RandomForest")

rf.train_random_forest()
rf.evaluate_random_forest()
trained_rf = rf.get_model()
rf.save_random_forest()


###----------------------------------------------------------------------------------------------------------------------------------------
dt = DecisionTree.DecisionTreeTrainer(X_train, y_train, X_test, y_test, model_path="Classification Models/Saved Models/Test_DecisionTree")

dt.train_decision_tree()
dt.evaluate_decision_tree()
trained_dt = dt.get_model()
dt.save_decision_tree()

###-----------------------------------------------------------------------------------------------------------------------------------------
ebm = Explainable_Boosting_Machines.ExplainableBoostingTrainer(X_train, y_train, X_test, y_test, processor.get_feature_names(), model_path="Classification Models/Saved Models/Test_ExplainableBoosting")

ebm.train_ebm()
ebm.evaluate_ebm()
trained_ebm = ebm.get_model()
ebm.save_ebm()

##----------------------------------------------------------------------------------------------------------------------------------------
nn = MLPClassifier.NeuralNetworkTrainer(X_train, y_train, X_test, y_test, model_path="Classification Models/Saved Models/Test_NeuralNet")

nn.train_neural_network()
nn.evaluate_neural_network()
trained_nn = nn.get_model()
nn.save_neural_network()



# General Model explanations (Inherent)
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
ebm.global_explanation()



# General Model explanations (SHAP)
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
shap_rf = SHAP_posthoc.SHAPAnalysis(trained_rf, X_train, X_test, y_test, processor.get_feature_names(), "RandomForest")
shap_rf.perform_shap_general_explanation() 

shap_dt = SHAP_posthoc.SHAPAnalysis(trained_dt, X_train, X_test, y_test, processor.get_feature_names(), "Decision Tree")
shap_dt.perform_shap_general_explanation()

#shap_ebm = SHAP_posthoc.SHAPAnalysis(trained_ebm, X_train, X_test, y_test, processor.get_feature_names(), "Explainable Boosted Machine")
#shap_ebm.perform_shap_general_explanation()

#shap_mlp = SHAP_posthoc.SHAPAnalysis(trained_nn, X_train, X_test,y_test, processor.get_feature_names(), "MLP NN")
#shap_mlp.perform_shap_general_explanation()