import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Union, Tuple
import datetime
from collections import Counter

def resample_with_smotenc(df, target_proportions, age_bins=None, random_state=42):
    """
    Resample a dataframe using SMOTENC to achieve target proportions for specified columns.
    Handles continuous age variable by binning it first.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe to resample
    target_proportions : dict
        Dictionary with keys as column names and values as dictionaries of category:proportion pairs
        For age (continuous), the categories should be bin labels
        Example: {'sex': {'Female': 0.5, 'Male': 0.5}, 
                 'age_bin': {'18-25': 0.2, '26-45': 0.5, '46+': 0.3},
                 'race': {'African-American': 0.33, 'Caucasian': 0.33, 'Hispanic': 0.34}}
    age_bins : list or None, default=None
        List of bin edges for age variable. If None and 'age_bin' in target_proportions,
        bins will be created based on the keys in target_proportions['age_bin']
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        Resampled dataframe with desired proportions
    """
    df_copy = df.copy()
    categorical_columns = ['sex', 'race', 'c_charge_degree', 'r_charge_degree', 
                           'vr_charge_degree', 'score_text']
    
    # fill missing values
    for col in categorical_columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna('Unknown')
    
    # fill with median the nans of r_days_from_arrest
    if 'r_days_from_arrest' in df_copy.columns:
        df_copy['r_days_from_arrest'] = df_copy['r_days_from_arrest'].fillna(df_copy['r_days_from_arrest'].median())
    
    # fill age bins with median
    if 'age_bin' in target_proportions:
        df_copy['age'] = df_copy['age'].fillna(df_copy['age'].median())
        
        # if age_bins not provided
        if age_bins is None:
            bin_labels = list(target_proportions['age_bin'].keys())
            
            # parse bin edges from labels ( format like '18-25', '26-45', '46+')
            bin_edges = [0] 
            for label in bin_labels:
                if '-' in label:
                    upper = label.split('-')[1]
                    if upper.isdigit():
                        bin_edges.append(int(upper) + 1)
                elif '+' in label:
                    lower = label.split('+')[0]
                    if lower.isdigit():
                        bin_edges.append(int(lower))
                        bin_edges.append(int(df_copy['age'].max()) + 1)  
            
            # sort and remove duplicates
            bin_edges = sorted(list(set(bin_edges)))
            
            if len(bin_edges) < 2:
                raise ValueError("Could not parse valid bin edges from age_bin labels")
        else:
            bin_edges = age_bins
            bin_labels = [f"{bin_edges[i]}-{bin_edges[i+1]-1}" for i in range(len(bin_edges)-2)]
            bin_labels.append(f"{bin_edges[-2]}+")
        
        # create the age_bin column
        df_copy['age_bin'] = pd.cut(df_copy['age'], bins=bin_edges, labels=bin_labels, right=False)
        df_copy['age_bin'] = df_copy['age_bin'].astype(str)
        
        # add age_bin to categorical columns
        categorical_columns.append('age_bin')
    
    # create a combined target column for SMOTENC based on the target proportions
    df_copy['_target'] = ''
    
    target_cols = list(target_proportions.keys())
    for idx, row in df_copy.iterrows():
        # Create a tuple of values from target columns
        target_val = []
        for col in target_cols:
            if col in row.index:
                target_val.append(str(row[col]))
            elif col == 'age_bin' and 'age_bin' in row.index:
                target_val.append(str(row['age_bin']))
            else:
                target_val.append('Unknown')
        df_copy.at[idx, '_target'] = '-'.join(target_val)
    
    # Calculate sample counts for each combination
    total_desired_samples = len(df_copy)
    target_combinations = df_copy['_target'].unique()
    
    desired_counts = {}
    for combo in target_combinations:
        combo_values = combo.split('-')
        if len(combo_values) != len(target_cols):
            continue
            
        # Get the proportion for each individual value from target_proportions
        combo_proportion = 1.0
        for i, col in enumerate(target_cols):
            val = combo_values[i]
            if col in target_proportions and val in target_proportions[col]:
                combo_proportion *= target_proportions[col][val]
            else:
                combo_proportion = 0
                break
                
        if combo_proportion > 0:
            desired_counts[combo] = int(combo_proportion * total_desired_samples)
    
    # categorical column indices for SMOTENC
    all_columns = list(df_copy.columns)
    categorical_features = []
    for i, col in enumerate(all_columns):
        if col in categorical_columns:
            categorical_features.append(i)
    
    # drop the target column 
    target_idx = all_columns.index('_target') if '_target' in all_columns else -1
    if target_idx in categorical_features:
        categorical_features.remove(target_idx)
    
    # prep data for SMOTENC
    X = df_copy.drop('_target', axis=1)
    y = df_copy['_target']
    
    # get sampling strategy
    sampling_strategy = {}
    current_counts = Counter(y)
    
    for combo, desired_count in desired_counts.items():
        actual_count = current_counts.get(combo, 0)
        if desired_count > actual_count:  # oversample if desired count is higher
            sampling_strategy[combo] = desired_count
    
    # apply SMOTENC 
    if sampling_strategy:
        try:
            smotenc = SMOTENC(categorical_features=categorical_features, 
                              sampling_strategy=sampling_strategy,
                              random_state=random_state,
                              k_neighbors=1)
            X_resampled, y_resampled = smotenc.fit_resample(X, y)
            
            df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            
            # print proportions
            print("Achieved proportions:")
            for col in target_proportions:
                check_col = col
                if col == 'age_bin':
                    check_col = 'age_bin'
                
                if check_col in df_resampled.columns:
                    value_counts = df_resampled[check_col].value_counts(normalize=True)
                    print(f"\n{check_col}:")
                    for category, proportion in value_counts.items():
                        print(f"  {category}: {proportion:.2f}")
            
            # keep the original age column if we created age_bin
            if 'age_bin' in df_resampled.columns and 'age_bin' not in df.columns:
                df_resampled = df_resampled.drop('age_bin', axis=1)
            
            return df_resampled
        
        # error handling
        except Exception as e:
            print(f"Error during resampling: {e}")
            # clean up
            if 'age_bin' in df_copy.columns and 'age_bin' not in df.columns:
                df_copy = df_copy.drop('age_bin', axis=1)
            return df_copy.drop('_target', axis=1)
    
    # if not possible to get sampling strat
    else:
        print("No valid sampling strategy could be determined based on target proportions.")
        # clean up
        if 'age_bin' in df_copy.columns and 'age_bin' not in df.columns:
            df_copy = df_copy.drop('age_bin', axis=1)
        return df_copy.drop('_target', axis=1)
    