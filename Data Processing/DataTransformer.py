import pickle
import joblib
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class DataTransformer:
    """
    A class for data preprocessing including handling missing values,
    normalization, encoding, and splitting data. The class supports saving and
    loading the preprocessing pipeline for future use.

    Attributes:
        df (pandas.DataFrame): The input dataset.
        y (pandas.Series): The target variable.
        relevant_columns (list): Columns to keep for processing.
        onehot_cols (list): Columns to apply OneHotEncoder.
        labelencode_cols (list): Columns to apply LabelEncoder.
        label_mapping (dict): Dictionary mapping categorical labels to integers.
        pipeline (sklearn.pipeline.Pipeline): The processing pipeline.
    """

    def __init__(self, df, y, relevant_columns, onehot_cols):
        self.df = df[relevant_columns].copy()
        self.y = y
        self.relevant_columns = relevant_columns
        self.onehot_cols = onehot_cols
        self.numeric_cols = [col for col in self.relevant_columns if df[col].dtype in ['float64', 'int64']]

        self.c_charge_degree_mapping = ['(CT)', '(F1)', '(F2)', '(F3)', '(F4)', '(F5)', '(F6)', '(F7)', '(M1)', '(M2)', '(MO3)', '(NI0)', '(TCX)', '(X)', 'Unknown']
        self.r_charge_degree_mapping = ['(F1)', '(F2)', '(F3)', '(F5)', '(F6)', '(F7)', '(M1)', '(M2)', '(MO3)', 'Unknown']
        self.vr_charge_degree_mapping = ['(F2)', '(F3)', '(F5)', '(F6)', '(F7)', '(M1)', '(M2)', '(MO3)', 'Unknown']

        
        # Define preprocessing steps
        self.pipeline = ColumnTransformer(
            transformers=[
                # Imputation and Scaling for Numeric Columns
                ("num_impute_scale", Pipeline([
                    ("imputer", KNNImputer(n_neighbors=3)),  
                    ("scaler", StandardScaler())             
                ]), self.numeric_cols),
        
                # Imputation for Categorical Columns (Replace NaN with 'Unknown') + One-Hot Encoding
                ("onehot", Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]), self.onehot_cols),
        
                # Imputation for Ordinal Columns (Replace NaN with 'Unknown') + Ordinal Encoding
                ('c_charge_degree', Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("encoder", OrdinalEncoder(categories=[self.c_charge_degree_mapping], handle_unknown='use_encoded_value', unknown_value=-1))
                ]), ['c_charge_degree']),  

                # Imputation for Ordinal Columns (Replace NaN with 'Unknown') + Ordinal Encoding
                ('r_charge_degree', Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("encoder", OrdinalEncoder(categories=[self.r_charge_degree_mapping], handle_unknown='use_encoded_value', unknown_value=-1))
                ]), ['r_charge_degree']), 

                # Imputation for Ordinal Columns (Replace NaN with 'Unknown') + Ordinal Encoding
                ('vr_charge_degree', Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("encoder", OrdinalEncoder(categories=[self.vr_charge_degree_mapping], handle_unknown='use_encoded_value', unknown_value=-1))
                ]), ['vr_charge_degree']), 
            ]
        )


    def fit_pipeline(self):
        """Fit the preprocessing pipeline to the data."""
        self.pipeline.fit(self.df)

    def transform_data(self):
        """Transform the dataset using the fitted pipeline."""
        return self.pipeline.transform(self.df)

    def split_data(self, test_size=0.2, random_state=42):
        """Split the transformed dataset into train and test sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.transform_data(), self.y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test

    def save_pipeline(self, filepath):
        """Save the pipeline for future use."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)

    def load_pipeline(self, filepath):
        """Load a previously saved pipeline."""
        with open(filepath, 'rb') as f:
            self.pipeline = pickle.load(f)

    def get_feature_names(self):
        """Get the feature names after transformation."""
        # Get the names of numeric features (these remain the same)
        numeric_feature_names = self.numeric_cols
        
        # Get the names of one-hot encoded features
        onehot_feature_names = self.pipeline.transformers_[1][1].named_steps['encoder'].get_feature_names_out(self.onehot_cols)
        
        # Combine all feature names into a single list
        all_feature_names = numeric_feature_names + list(onehot_feature_names) + ['c_charge_degree', 'r_charge_degree', 'vr_charge_degree']
        
        return all_feature_names







    #def transform_single_entry(self, entry):
    #    """Transform a single new data entry."""
    #    entry_df = pd.DataFrame([entry])
    #    
    #    # Fill missing categorical values in the new entry with 'Unknown'
    #    for col in self.onehot_cols + self.labelencode_cols:
    #        if col in entry_df:
    #            entry_df[col] = entry_df[col].fillna("Unknown")
#
    #    # Apply label encoding (custom mapping)
    #    for col, mapping in self.label_mapping.items():
    #        if col in entry_df:
    #            entry_df[col] = entry_df[col].map(mapping).fillna(-1)
#
    #    # Label encode categorical columns
    #    for col in self.labelencode_cols:
    #        if col not in self.label_mapping and col in entry_df:
    #            entry_df[col] = self.label_encoders[col].transform(entry_df[col])
#
    #    return self.pipeline.transform(entry_df)
