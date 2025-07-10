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

def target_encode(df, target_cols_list, n_splits=5):
    """Target encoding to specified columns using cross-validation."""
    target_col = target_cols_list[0]
    features_to_encode = target_cols_list[1:]
    # overall mean of the target variable
    global_mean = df[target_col].mean()
    for col in features_to_encode:
        encoded_col_name = col + '_target'
        # temporary DataFrame with feature, target, and original index
        temp_df = pd.DataFrame({
            'feature': df[col],
            'target': df[target_col],
            'original_index': df.index
        })
        # k_fold_split
        folds = k_fold_split(temp_df.to_numpy(), temp_df.to_numpy(), k=n_splits)[0]
        df[encoded_col_name] = np.nan
        # iterate over folds
        for i in range(len(folds)):
            # current fold = validation set, rest = training 
            val_fold_np = folds[i]
            train_folds_np = np.vstack([folds[j] for j in range(len(folds)) if i != j])
            train_fold_df = pd.DataFrame(train_folds_np, columns=temp_df.columns)
            val_fold_df = pd.DataFrame(val_fold_np, columns=temp_df.columns)

            # mean of the target for each category on the training fold
            target_mean_map = train_fold_df.groupby('feature')['target'].mean()
            # global_mean as a fallback for categories not in the training fold
            mapped_values = val_fold_df['feature'].map(lambda x: target_mean_map.get(x, global_mean))
            
            # original indices for the validation fold
            original_indices = val_fold_df['original_index'].astype(int)
            df.loc[original_indices, encoded_col_name] = mapped_values.values

    df.drop(columns=features_to_encode, inplace=True)
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
        df_encoded = ordinal_encode(df_encoded, encoding_strategies['ordinal']) 

    if 'one-hot' in encoding_strategies:
        df_encoded = one_hot_encode(df_encoded, encoding_strategies['one-hot'])

    if 'frequency' in encoding_strategies:
        df_encoded = frequency_encode(df_encoded, encoding_strategies['frequency'])

    if 'target' in encoding_strategies:
        df_encoded = target_encode(df_encoded, encoding_strategies['target'])

    return df_encoded