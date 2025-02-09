from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

class DataProcessor:
    """
    A class to preprocess the dataset and prepare it for machine learning modeling.
    
    This class provides functionality to filter relevant columns, handle missing values,
    normalize numerical columns, encode categorical features, and split the data into training
    and testing sets. It allows for flexibility through enabling/disabling each step.

    Attributes:
        df (pandas.DataFrame): The input dataset containing features.
        y (pandas.Series): The target variable.
        relevant_columns (list): List of columns to include in the processing.
        normalizer_enabled (bool): Flag to enable/disable normalization of numerical features.
        encoder_enabled (bool): Flag to enable/disable encoding of categorical features.
        imputer_enabled (bool): Flag to enable/disable missing value imputation.

    Methods:
        preprocess_data(): 
            Filters the dataset to include only the relevant columns specified in `relevant_columns`.
        fill_missing_values(): 
            Handles missing values by imputing numerical values with KNN and filling categorical missing values with 'Unknown'.
        normalize_numerical_columns(): 
            Normalizes numerical columns using Min-Max scaling to the range [0, 1].
        encode_categorical_features(): 
            Encodes categorical features using One-Hot Encoding for nominal features and Label Encoding for binary/ordinal features.
        split_data(): 
            Splits the dataset into training and testing sets.
        process_data(): 
            Runs the full preprocessing pipeline: filters columns, handles missing values, normalizes and encodes features, and splits the data.
    """
    
    def __init__(self, df, y, relevant_columns, normalizer_enabled=True, encoder_enabled=True, imputer_enabled=True):
        """
        Initializes the DataProcessor with the given dataset and settings.

        :param df: The input dataset as a pandas DataFrame.
        :param y: The target variable as a pandas Series or column from the DataFrame.
        :param relevant_columns: A list of relevant columns to be processed.
        :param normalizer_enabled: Boolean flag to enable/disable normalization of numerical columns (default is True).
        :param encoder_enabled: Boolean flag to enable/disable encoding of categorical columns (default is True).
        :param imputer_enabled: Boolean flag to enable/disable missing value imputation (default is True).
        """
        self.df = df
        self.y = y
        self.normalizer_enabled = normalizer_enabled
        self.encoder_enabled = encoder_enabled
        self.imputer_enabled = imputer_enabled
        self.relevant_columns = relevant_columns

    def preprocess_data(self):
        """
        Filters the dataset to include only the relevant columns specified in `relevant_columns`.

        This method updates the dataframe to contain only the columns that are specified as relevant
        for the analysis, as indicated by the `relevant_columns` attribute.
        """
        self.df = self.df[self.relevant_columns]

    def fill_missing_values(self):
        """
        Handles missing values in the dataset.
        
        For numerical columns, missing values are imputed using KNN imputation.
        For categorical columns, missing values are filled with the string 'Unknown'.
        """
        knn_imputer = KNNImputer(n_neighbors=5)  

        # Iterate through the columns in the dataframe
        for col in self.df.columns.tolist():
            if self.df[col].dtype == 'object':  
                self.df[col].fillna('Unknown', inplace=True)  
            elif self.df[col].dtype in ['int64', 'float64']: 
                self.df[col] = knn_imputer.fit_transform(self.df[[col]])  

    def normalize_numerical_columns(self):
        """
        Normalizes numerical columns using Min-Max scaling.
        
        This method scales numerical columns to the range [0, 1] using the `MinMaxScaler` from scikit-learn.
        """
        scaler = MinMaxScaler() 

        # Iterate through the columns and normalize numerical columns
        for col in self.df.columns:
            if self.df[col].dtype in ['int64', 'float64']: 
                self.df[col] = scaler.fit_transform(self.df[[col]]) 

    def encode_categorical_features(self):
        """
        Encodes categorical features in the dataset.
        
        This method applies:
        - One-Hot Encoding for nominal categorical features (with more than two unique values).
        - Label Encoding for binary or ordinal categorical features (with two unique values).
        """
        le = LabelEncoder()  

        # Loop through all categorical columns to apply encoding
        for col in self.df.columns:
            if self.df[col].dtype == 'object':  
                if len(self.df[col].unique()) > 2:  
                    self.df = pd.get_dummies(self.df, columns=[col], drop_first=True)  
                else:  
                    self.df[col] = le.fit_transform(self.df[col]) 

    def split_data(self, test_size=0.2, random_state=42):
        """
        Splits the dataset into training and testing sets.

        :param test_size: Proportion of the data to use for the test set (default is 20%).
        :param random_state: Random seed for reproducibility (default is 42).
        :return: The training and testing data (X_train, X_test, y_train, y_test).
        """
        if self.df is None:
            print("No data loaded.")
            return None, None, None, None

        # Split the dataset into training and testing sets using train_test_split
        return train_test_split(self.df, self.y, test_size=test_size, random_state=random_state)

    def process_data(self):
        """
        Runs the entire data processing pipeline in sequence:
        1. Filters the relevant columns.
        2. Handles missing values (if enabled).
        3. Normalizes numerical columns (if enabled).
        4. Encodes categorical features (if enabled).
        5. Splits the processed data into training and testing sets.

        :return: X_train, X_test, y_train, y_test - The processed and split datasets.
        """
        self.preprocess_data()
        if self.imputer_enabled:
            self.fill_missing_values()
        if self.normalizer_enabled:
            self.normalize_numerical_columns()
        if self.encoder_enabled:
            self.encode_categorical_features()
        return self.split_data()
