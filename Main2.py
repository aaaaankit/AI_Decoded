from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Add paths to import custom modules
root = Path(__file__).parent
cwd = Path(__file__).parent
path_Data_processing = cwd / 'Data Processing'
path_Random_forest = cwd / 'Classification Models' / 'Uninterpretable Models'
path_Decision_tree = cwd / 'Classification Models'
path_post_hoc = cwd / 'Post-Hoc Analysis'
path_inherent = cwd / 'Inherently_Interpretable_Analysis'

sys.path.append(str(path_Data_processing))
sys.path.append(str(path_Random_forest))
sys.path.append(str(path_Decision_tree))
sys.path.append(str(path_post_hoc))
sys.path.append(str(path_inherent))

import DataProcessor
import DataTransformer
import Randomforest
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

# Apply preprocessing
processor = DataTransformer.DataTransformer(
    df=df,
    y=y,
    relevant_columns=relevant,
    onehot_cols=one_hot_columns,
)

# Fit the pipeline
processor.fit_pipeline()

# Transform the dataset
X_transformed = processor.pipeline.transform(processor.df)

# Split into train and test sets
X_train, X_test, y_train, y_test = processor.split_data()

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the pipeline for later use
processor.save_pipeline('cox_pipeline.pkl')

# Now apply Anchor Analysis
# Create an AnchorAnalysis instance for post-hoc analysis
feature_names = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 
 'r_days_from_arrest', 'race_African American', 'race_Asian', 'race_Caucasian', 'race_Hispanic', 
 'race_Native American', 'race_Other', 'sex_Female', 'sex_Male', 'c_charge_degree', 'r_charge_degree', 'vr_charge_degree']
feature_names = processor.get_feature_names()
print(feature_names)
class_names = y.unique().tolist()  # List of unique classes in the target variable
print(class_names)

# Set the index of the test instance to analyze with anchors
anchor_test_instance_index = 133  # Change this to any other index you want to analyze

# Perform Anchor Analysis
anchor_rf = Anchor_posthoc.AnchorAnalysis(model, X_train, X_test, y_test, feature_names, class_names)
anchor_rf.perform_anchor_analysis(anchor_test_instance_index)

# Optional: Visualize the anchor analysis results (you can customize this part)
# anchor_rf.plot_anchor_analysis()

print("Processing complete. Transformed data is ready.")

data_point = {
    "sex": "Male",
    "age": 27,
    "age_cat": "25 - 45",
    "race": "Caucasian",
    "juv_fel_count": 1,
    "decile_score": 5,
    "juv_misd_count": 2,
    "juv_other_count": 0,
    "priors_count": 4,
    "days_b_screening_arrest": 10,
    "c_jail_in": "01/01/2015 12:00",
    "c_jail_out": "01/01/2015 14:00",
    "c_days_from_compas": 20,
    "c_charge_degree": "(F1)",
    "c_charge_desc": "Battery",
    "is_recid": 0,
    "r_charge_degree": "(F2)",
    "r_days_from_arrest": 15,
    "r_offense_date": "10/02/2015",
    "r_jail_in": "10/02/2015 12:00",
    "violent_recid": 0,
    "is_violent_recid": 0,
    "vr_charge_degree": "(M1)",
    "vr_offense_date": "11/02/2015",
    "score_text": "Battery",
    "screening_date": "01/01/2015",
    "v_decile_score": 4,
    "v_score_text": "Medium",
    "event": 0
}

# Apply the transformation to the new data point
new_data_df = pd.DataFrame([data_point])
transformed_data = processor.pipeline.transform(new_data_df)

# Output the transformed data
print("Transformed Data:")
print(transformed_data)

prediction = model.predict(transformed_data)

# Output the prediction
print(f"Prediction for the new data point: {prediction[0]}")
anchor_rf.perform_anchor_analysis_instance(transformed_data)