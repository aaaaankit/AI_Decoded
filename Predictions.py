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

import pandas as pd
import pickle
import json

def transform_and_save(input_file_path, transformer_pickle_path, output_file_path):
    # Step 1: Read the input data from the txt file
    with open(input_file_path, 'r') as file:
        input_data = json.load(file)  # Assuming the input is in JSON format
    
    # Step 2: Load the transformer (pickled pipeline)
    with open(transformer_pickle_path, 'rb') as f:
        transformer = pickle.load(f)
    
    # Step 3: Convert input data to DataFrame
    new_data_df = pd.DataFrame([input_data])
    
    # Step 4: Apply the transformation using the loaded transformer
    transformed_data = transformer.transform(new_data_df)
    
    # Step 5: Reshape the transformed data (if needed)
    transformed_data = transformed_data.reshape(-1)
    
    # Step 6: Prepare the transformed data as a dictionary
    transformed_dict = {
        "age": transformed_data[0],
        "juv_fel_count": transformed_data[1],
        "juv_misd_count": transformed_data[2],
        "juv_other_count": transformed_data[3],
        "r_days_from_arrest": transformed_data[4],
        "race_African American": transformed_data[5],
        "race_Asian": transformed_data[6],
        "race_Caucasian": transformed_data[7],
        "race_Hispanic": transformed_data[8],
        "race_Native American": transformed_data[9],
        "race_Other": transformed_data[10],
        "sex_Female": transformed_data[11],
        "sex_Male": transformed_data[12],
        "c_charge_degree": transformed_data[13],
        "r_charge_degree": transformed_data[14],
        "vr_charge_degree": transformed_data[15]
    }
    
    # Step 7: Save the transformed data to a new text file
    with open(output_file_path, 'w') as f:
        f.write(json.dumps(transformed_dict, indent=4))  # Save as JSON format for readability


# Data Loading
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
transformed_dataset = True
df = pd.read_csv('Dataset/transformed_dataset.csv')
df = df.dropna(subset=["score_text"])
if not transformed_dataset:
    df['race'] = df['race'].str.replace('African-American', 'African American')     # replace African-American with African American in the race column

    relevant = ["sex","age","race","juv_fel_count","juv_misd_count","juv_other_count",
                "c_charge_degree","r_charge_degree","r_days_from_arrest",
               "vr_charge_degree"]
    
    one_hot_columns = ['race', 'sex']

target = "score_text"

y = df[target]
df.drop(target, axis=1, inplace=True)

    
    
# Data transformer loading
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
if not transformed_dataset:
    processor = DataTransformer.DataTransformer(
        df=df,
        y=y,
        relevant_columns=relevant,
        onehot_cols=one_hot_columns,
    )

    processor.load_pipeline("Data Processing/DataTransformer.pkl")
    X_train, X_test, y_train, y_test = processor.split_data()
    feature_names = processor.get_feature_names()
else:
    feature_names = df.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)   
    X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values





# Model loading
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
rf = Randomforest.RandomForestTrainer(X_train, y_train, X_test, y_test, model_path="Classification Models/Saved Models/Test_RandomForest")
rf.load_random_forest()
trained_rf = rf.get_model()

##----------------------------------------------------------------------------------------------------------------------------------------

dt = DecisionTree.DecisionTreeTrainer(X_train, y_train, X_test, y_test, model_path="Classification Models/Saved Models/Test_DecisionTree")
dt.load_decision_tree()
trained_dt = dt.get_model()

##----------------------------------------------------------------------------------------------------------------------------------------

ebm = Explainable_Boosting_Machines.ExplainableBoostingTrainer(X_train, y_train, X_test, y_test, feature_names, model_path="Classification Models/Saved Models/Test_ExplainableBoosting")
ebm.load_ebm()
trained_ebm = ebm.get_model()

#----------------------------------------------------------------------------------------------------------------------------------------

nn = MLPClassifier.NeuralNetworkTrainer(X_train, y_train, X_test, y_test, model_path="Classification Models/Saved Models/Test_NeuralNet")
nn.load_neural_network()
trained_nn = nn.get_model()



# Data entry
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
input_file_path = 'dataPoint.txt'  # Path to the input text file
transformer_pickle_path = 'Data Processing/DataTransformer.pkl'  # Path to the transformer pickle file
output_file_path = 'transformed_data.txt'  # Path to save the transformed data
transform_and_save(input_file_path, transformer_pickle_path, output_file_path)

data_point = read_json('transformed_data.txt')
new_data_df = pd.DataFrame([data_point])

if transformed_dataset:  # Assuming is_data_transformed is a function you can define
    transformed_data = new_data_df.values.reshape(-1)  # No transformation applied, just reshape
else:
    transformed_data = processor.pipeline.transform(new_data_df)
    transformed_data = transformed_data.reshape(-1)

print(transformed_data.shape)
print(X_test.shape)
print(transformed_data)



# Local (prediction level explanations) Inherent
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
inherent_ebm = ebm.local_explanation(transformed_data)


# Local (prediction level explanations) LIME
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
lime_rf = LIME_posthoc.LimeAnalysis(trained_rf, X_train, X_test, y_test, feature_names, y.unique().tolist())
lime_rf.perform_lime_analysis_instance(transformed_data)

lime_dt = LIME_posthoc.LimeAnalysis(trained_dt, X_train, X_test, y_test, feature_names, y.unique().tolist())
lime_dt.perform_lime_analysis_instance(transformed_data)

lime_ebm = LIME_posthoc.LimeAnalysis(trained_ebm, X_train, X_test, y_test, feature_names, y.unique().tolist())
lime_ebm.perform_lime_analysis_instance(transformed_data)

lime_nn = LIME_posthoc.LimeAnalysis(trained_nn, X_train, X_test, y_test, feature_names, y.unique().tolist())       #TODO
lime_nn.perform_lime_analysis_instance(transformed_data)


# Local (prediction level explanations) SHAP
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
shap_rf = SHAP_posthoc.SHAPAnalysis(trained_rf, X_train, X_test, y_test, feature_names, "RandomForest")
shap_rf.perform_shap_local_explanation_instance(transformed_data)

shap_dt = SHAP_posthoc.SHAPAnalysis(trained_dt, X_train, X_test, y_test, feature_names, "Decision Tree")
shap_dt.perform_shap_local_explanation_instance(transformed_data)

shap_ebm = SHAP_posthoc.SHAPAnalysis(trained_ebm, X_train, X_test, y_test, feature_names, "Explainable Boosted Machine")
shap_ebm.perform_shap_local_explanation_instance(transformed_data)

shap_nn = SHAP_posthoc.SHAPAnalysis(trained_nn, X_train, X_test, y_test, feature_names, "MLP NN")       #TODO
shap_nn.perform_shap_local_explanation_instance(transformed_data)


# Local (prediction level explanations) Anchors
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
anchor_rf = Anchor_posthoc.AnchorAnalysis(trained_rf, X_train, X_test, y_test, feature_names, y.unique().tolist())
print(anchor_rf.perform_anchor_analysis_instance(transformed_data))

anchor_dt = Anchor_posthoc.AnchorAnalysis(trained_dt, X_train, X_test, y_test, feature_names, y.unique().tolist())
print(anchor_dt.perform_anchor_analysis_instance(transformed_data))

anchor_ebm = Anchor_posthoc.AnchorAnalysis(trained_ebm, X_train, X_test, y_test, feature_names, y.unique().tolist())
print(anchor_ebm.perform_anchor_analysis_instance(transformed_data))

anchor_nn = Anchor_posthoc.AnchorAnalysis(trained_nn, X_train, X_test, y_test, feature_names, y.unique().tolist())
print(anchor_nn.perform_anchor_analysis_instance(transformed_data))


# Global (model level explanations) Feature Importance
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
#feature_importance_rf = Feature_Importance_posthoc.FeatureImportanceAnalysis(trained_rf, X_train, X_test, y_test, feature_names)
#feature_importance_rf.explain_instance(transformed_data)
#
#feature_importance_dt = Feature_Importance_posthoc.FeatureImportanceAnalysis(trained_dt, X_train, X_test, y_test, feature_names)
#feature_importance_dt.explain_instance(transformed_data)
#
#feature_importance_ebm = Feature_Importance_posthoc.FeatureImportanceAnalysis(trained_ebm, X_train, X_test, y_test, feature_names)
#feature_importance_ebm.explain_instance(transformed_data)
#
#feature_importance_nn = Feature_Importance_posthoc.FeatureImportanceAnalysis(trained_nn, X_train, X_test, y_test, feature_names)
#feature_importance_nn.explain_instance(transformed_data)

# Global (model level explanations) Partial Dependence Plots
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
# partial_dependence_rf = Partial_Dependence_Plots_posthoc.PartialDependencePlotsAnalysis(trained_rf, X_train, X_test, y_test, feature_names)
# partial_dependence_rf.explain_instance(transformed_data, feature_names, target)

# partial_dependence_dt = Partial_Dependence_Plots_posthoc.PartialDependencePlotsAnalysis(trained_dt, X_train, X_test, y_test, feature_names)
# partial_dependence_dt.explain_instance(transformed_data, feature_names, target)

# partial_dependence_ebm = Partial_Dependence_Plots_posthoc.PartialDependencePlotsAnalysis(trained_ebm, X_train, X_test, y_test, feature_names)
# partial_dependence_ebm.explain_instance(transformed_data, feature_names, target)

# partial_dependence_nn = Partial_Dependence_Plots_posthoc.PartialDependencePlotsAnalysis(trained_nn, X_train, X_test, y_test, feature_names)
# partial_dependence_nn.explain_instance(transformed_data, feature_names, target)

# Global (model level explanations) Permutation Importance
#****************************************************************************************************************************************
#----------------------------------------------------------------------------------------------------------------------------------------
#permutation_importance_rf = Permutation_Importance_posthoc.PermutationImportanceAnalysis(trained_rf, X_train, X_test, y_test, feature_names)
#permutation_importance_rf.explain_instance(transformed_data)
#
#permutation_importance_dt = Permutation_Importance_posthoc.PermutationImportanceAnalysis(trained_dt, X_train, X_test, y_test, feature_names)
#permutation_importance_dt.explain_instance(transformed_data)
#
#permutation_importance_ebm = Permutation_Importance_posthoc.PermutationImportanceAnalysis(trained_ebm, X_train, X_test, y_test, feature_names)
#permutation_importance_ebm.explain_instance(transformed_data)
#
#permutation_importance_nn = Permutation_Importance_posthoc.PermutationImportanceAnalysis(trained_nn, X_train, X_test, y_test, feature_names)
#permutation_importance_nn.explain_instance(transformed_data)

