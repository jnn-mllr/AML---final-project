import numpy as np
import pandas as pd
from utils.splits import k_fold_split

def labels_encoding(Y, labels=None, pos_value=1, neg_value=-1):
    """
    Encodes class labels into a one-vs-rest style matrix with custom values.

    Parameters:
    - Y: array-like of shape (N,) – class labels
    - labels: optional list of label values in desired order; if None, inferred from sorted unique values
    - pos_value: value for the positive (true) class (default: 1)
    - neg_value: value for the negative class (default: -1)

    Returns:
    - encoded: ndarray of shape (N, K), where K = number of classes
    """
    Y = np.asarray(Y)
    if labels is None:
        labels = np.unique(Y)
    label_to_index = {label: i for i, label in enumerate(labels)}
    
    K = len(labels)
    N = len(Y)
    
    encoded = np.full((N, K), neg_value, dtype=float)
    for i, y in enumerate(Y):
        k = label_to_index[y]
        encoded[i, k] = pos_value

    return encoded

def labels_to_numbers(labels, class_names=None):
    if class_names is None:
        class_names = np.unique(labels)
    label_to_number = {label: i for i, label in enumerate(class_names)}
    return np.array([label_to_number[label] for label in labels])

def preprocess_data(df, nan_columns=None):
    """
    Handles duplicate rows and missing values in a DataFrame.

    This function first removes duplicate rows to ensure data integrity. It then
    replaces missing values (NaN) in specified categorical columns with the
    string 'Missing', treating them as a distinct category.

    Parameters:
    - df: pandas.DataFrame
        The DataFrame to process.
    - nan_columns: list of str, optional
        A list of column names in which to fill missing values. If None, the function
        targets all columns with an 'object' dtype.

    Returns:
    - pandas.DataFrame
        The preprocessed DataFrame with duplicates removed and missing values handled.
    """
    # to avoid warnings
    df = df.copy()

    # remove duplicates
    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        print(f"{num_duplicates} duplicate observations in the dataset were removed.")
        # Reassign instead of using inplace=True
        df = df.drop_duplicates(keep='first')
    else:
        print("no duplicated observations in the dataset.")

    # treat missing values as a separate category
    if nan_columns is None:
        nan_columns = df.select_dtypes(include=['object']).columns
    
    for col in nan_columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna('Missing')

    # remove leading/trailing whitespace
    for col in df.select_dtypes(include='object').columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip()

    # convert objects to categories to save memory and speed up processing
    for col in df.select_dtypes(include=['object']).columns:
       df[col] = df[col].astype('category')
    return df

def transform_skewed_features(df, columns):
    """
    Applies a log1p transformation to skewed features and creates
    binary indicators for non-zero values.
    """
    for col in columns:
        # binary indicator for non-zero values
        df[f'has_{col}'] = (df[col] > 0).astype(int)
        # log1p transformation (log(1+x)) to handle zeros
        df[col] = np.log1p(df[col])
    print(f"log1p transformation and binary indicators for: {columns}")
    return df

def ordinal_encode(df, ordinal_cols):
    """Applies ordinal encoding to specified columns using
       labels_to_numbers."""
    for col, order in ordinal_cols.items():
        df[col + '_ordinal'] = labels_to_numbers(df[col], class_names=order)
        df.drop(col, axis=1, inplace=True)
    return df


def one_hot_encode(df, one_hot_config):
    """
    Applies one-hot encoding to specified columns.

    Parameters:
    - df: pandas.DataFrame
        The input DataFrame.
    - one_hot_config: dict
        - dict: A dictionary where keys are column names and values are the specific
          categories to drop for each column to avoid multicollinearity.

    Returns:
    - pandas.DataFrame
        The DataFrame with one-hot encoded features.
    """
    for col, category_to_drop in one_hot_config.items():
        # one-hot-encoding
        unique_cats = sorted(df[col].astype(str).unique())
        encoded_matrix = labels_encoding(df[col], labels=unique_cats, pos_value=1, neg_value=0)
        encoded_df = pd.DataFrame(encoded_matrix, columns=[f"{col}_{cat}" for cat in unique_cats], index=df.index)

        if category_to_drop:
            # drop the specified category
            col_to_drop = f"{col}_{category_to_drop}"
            encoded_df.drop(columns=col_to_drop, inplace=True)
        else:
            encoded_df.drop(columns=encoded_df.columns[0], inplace=True)
        # concat the dataframes
        df = pd.concat([df, encoded_df], axis=1)
        df.drop(col, axis=1, inplace=True)
    return df


def frequency_encode(df, freq_cols):
    """Frequency encoding to specified columns."""
    for col in freq_cols:
        freq_map = df[col].value_counts() / len(df)
        df[col + '_freq'] = df[col].map(freq_map)
        df.drop(col, axis=1, inplace=True)
    return df

def target_encode(df, target_cols_list, n_splits=5, fit_maps=None):
    """Target encoding with CV for fitting, and simple mapping for transforming."""
    df = df.copy()
    
    # Handle the dictionary format from encoding_strategies
    if isinstance(target_cols_list, dict):
        features_to_encode = list(target_cols_list.keys())
        target_col = list(target_cols_list.values())[0]
    else: # Keep old list-based logic for backward compatibility
        target_col = target_cols_list[0]
        features_to_encode = target_cols_list[1:]

    is_fitting = fit_maps is None

    if is_fitting:
        # FIT MODE: Use CV and learn global maps
        fit_maps = {'_global_mean': df[target_col].mean()}
        for col in features_to_encode:
            fit_maps[col] = df.groupby(col, observed=False)[target_col].mean()

        for col in features_to_encode:
            encoded_col_name = col + '_target'
            df[encoded_col_name] = np.nan
            
            # WORKAROUND: Pass indices to k_fold_split to get folds of indices
            all_indices = np.arange(len(df))
            # k_fold_split requires a Y array, so we create a dummy one.
            dummy_y = np.zeros(len(df))
            # The function returns (X_folds, Y_folds). We only need the first part.
            val_index_folds = k_fold_split(all_indices, dummy_y, k=n_splits)[0]
            
            for val_indices in val_index_folds:
                # Determine train indices by excluding validation indices
                train_indices = np.setdiff1d(all_indices, val_indices)
                
                # Use the integer indices to safely slice the DataFrame
                train_fold, val_fold = df.iloc[train_indices], df.iloc[val_indices]
                
                target_mean_map = train_fold.groupby(col,  observed=False)[target_col].mean()
                fold_global_mean = train_fold[target_col].mean()
                
                # Convert to float BEFORE filling, to avoid TypeError with categorical dtype
                mapped_values = val_fold[col].map(target_mean_map).astype(float).fillna(fold_global_mean)
                df.loc[val_fold.index, encoded_col_name] = mapped_values
        
        df.drop(columns=features_to_encode, inplace=True)
        return df, fit_maps
    
    else:
        # TRANSFORM MODE: Apply learned maps
        global_mean = fit_maps['_global_mean']
        for col in features_to_encode:
            mapping = fit_maps[col]
            # Convert to float before filling NaNs to avoid TypeError
            df[col + '_target'] = df[col].map(mapping).astype(float).fillna(global_mean)
        
        df.drop(columns=features_to_encode, inplace=True)
        return df

def encode_features(df, encoding_strategies, fit_params=None):
    """
    Encodes features of a DataFrame based on specified strategies.
    Can operate in 'fit' (learning) or 'transform' (applying) mode.
    
    Args:
        df (pd.DataFrame): The dataframe to encode.
        encoding_strategies (dict): The strategies for encoding.
        fit_params (dict, optional): Learned parameters from a previous fit. 
                                     If None, the function is in 'fit' mode.
                                     Defaults to None.

    Returns:
        If fitting (fit_params is None):
            - pd.DataFrame: The encoded dataframe.
            - dict: The learned parameters.
        If transforming (fit_params is not None):
            - pd.DataFrame: The encoded dataframe.
    """
    df = df.copy()
    is_fitting = fit_params is None
    if is_fitting:
        fit_params = {}

    # Ordinal encoding is stateless as the order is predefined
    if 'ordinal' in encoding_strategies:
        df = ordinal_encode(df, encoding_strategies['ordinal'])

    # Target encoding with fit/transform logic
    if 'target' in encoding_strategies:
        if is_fitting:
            df, target_maps = target_encode(df, encoding_strategies['target'])
            fit_params['target_maps'] = target_maps
        else:
            df = target_encode(df, encoding_strategies['target'], fit_maps=fit_params.get('target_maps'))

    # One-hot encoding with fit/transform logic
    if 'one-hot' in encoding_strategies:
        # Using pandas get_dummies is more standard and handles the logic well
        cols_to_encode = list(encoding_strategies['one-hot'].keys())
        if is_fitting:
            df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
            fit_params['one_hot_columns'] = df_encoded.columns.tolist()
            df = df_encoded
        else:
            df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
            # Ensure test set has same columns as train set
            df = df.reindex(columns=fit_params.get('one_hot_columns', df.columns), fill_value=0)

    if is_fitting:
        return df, fit_params
    else:
        return df