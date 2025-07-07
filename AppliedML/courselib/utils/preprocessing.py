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
    # remove duplicates
    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        print(f"{num_duplicates} duplicate observations in the dataset were removed.")
        df.drop_duplicates(keep='first', inplace=True)
    else:
        print("no duplicated observations in the dataset.")

    # treat missing values aas separate category
    if nan_columns is None:
        nan_columns = df.select_dtypes(include=['object']).columns
    
    for col in nan_columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna('Missing', inplace=True)
            
    return df

def ordinal_encode(df, ordinal_cols):
    """Applies ordinal encoding to specified columns using labels_to_numbers."""
    print("Applying ordinal encoding...")
    for col, order in ordinal_cols.items():
        # Create a new column with the ordinal encoding
        df[col + '_ordinal'] = labels_to_numbers(df[col], class_names=order)
        # Drop the original column
        df.drop(col, axis=1, inplace=True)
    return df

def one_hot_encode(df, one_hot_cols):
    """Applies one-hot encoding to specified columns using labels_encoding."""
    print("Applying one-hot encoding...")
    for col in one_hot_cols:
        # Get the unique categories to use as column names
        unique_cats = sorted(df[col].unique())
        
        # Perform one-hot encoding
        encoded_matrix = labels_encoding(df[col], labels=unique_cats, pos_value=1, neg_value=0)
        
        # Create a new DataFrame with the one-hot encoded columns
        encoded_df = pd.DataFrame(encoded_matrix, columns=[f"{col}_{cat}" for cat in unique_cats], index=df.index)
        
        # To avoid multicollinearity, we can drop one of the columns (optional, but good practice)
        encoded_df.drop(columns=encoded_df.columns[0], inplace=True)

        # Concatenate the new encoded columns with the original DataFrame
        df = pd.concat([df, encoded_df], axis=1)
        
        # Drop the original categorical column
        df.drop(col, axis=1, inplace=True)
        
    return df

def frequency_encode(df, freq_cols):
    """Frequency encoding to specified columns."""
    for col in freq_cols:
        freq_map = df[col].value_counts() / len(df)
        df[col + '_freq'] = df[col].map(freq_map)
        df.drop(col, axis=1, inplace=True)
    return df

def target_encode(df, target_cols_list):
    """Target encoding to specified columns using cross-validation."""
    target_col = target_cols_list[0]
    features_to_encode = target_cols_list[1:]
    global_mean = df[target_col].mean()

    for col in features_to_encode:
        encoded_col_name = col + '_target'
        df[encoded_col_name] = 0.0

        # Using the custom k_fold_split from courselib
        all_indices = np.arange(len(df))
        # Shuffle indices to ensure random folds
        np.random.shuffle(all_indices)
        fold_indices = np.array_split(all_indices, 5)

        for i in range(5):
            val_indices = fold_indices[i]
            train_indices = np.concatenate([fold_indices[j] for j in range(5) if i != j])

            train_fold_data = df.iloc[train_indices]
            target_mean_map = train_fold_data.groupby(col)[target_col].mean()

            val_fold_data = df.iloc[val_indices]
            df.loc[val_indices, encoded_col_name] = val_fold_data[col].map(target_mean_map)
        
        df[encoded_col_name].fillna(global_mean, inplace=True)
        df.drop(col, axis=1, inplace=True)
        
    return df

def encode_features(df, encoding_strategies):
    """
    Applies specified encoding strategies to categorical features in a DataFrame
    by calling dedicated encoding functions.

    Parameters:
    - df: pandas.DataFrame
        The input DataFrame.
    - encoding_strategies: dict
        A dictionary where keys are encoding strategy names ('one-hot', 'ordinal',
        'frequency', 'target') and values are the columns or configurations
        for that strategy.

    Returns:
    - pandas.DataFrame
        The DataFrame with features encoded according to the specified strategies.
    """
    df_encoded = df.copy()

    if 'ordinal' in encoding_strategies:
        df_encoded = ordinal_encode(df_encoded, encoding_strategies['ordinal'], {})

    if 'one-hot' in encoding_strategies:
        df_encoded = one_hot_encode(df_encoded, encoding_strategies['one-hot'])

    if 'frequency' in encoding_strategies:
        df_encoded = frequency_encode(df_encoded, encoding_strategies['frequency'])

    if 'target' in encoding_strategies:
        df_encoded = target_encode(df_encoded, encoding_strategies['target'])

    return df_encoded
