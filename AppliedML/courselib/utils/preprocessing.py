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
    Removes duplicate rows and missing values in a DataFrame.

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

def log_transform(df, columns):
    """
    Applies a log1p transformation to skewed features and creates
    binary indicators for non-zero values.
    """
    for col in columns:
        # log1p transformation (log(1+x)) to handle zeros
        df[col] = np.log1p(df[col])
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
    - df: input DataFrame.
    - one_hot_config: dictionary where keys are column names and values are the specific
          categories to drop for each column to avoid multicollinearity.

    Returns:
    - df: DataFrame with one-hot encoded features.
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


def frequency_encode(df, freq_cols, fit_maps=None):
    """
    Frequency encoding with fit/transform logic to prevent data leakage.
    'fit' mode learns the frequencies, 'transform' mode applies them.
    """
    is_fitting = fit_maps is None
    if is_fitting:
        fit_maps = {} # store to apply to test data

    for col in freq_cols:
        if is_fitting:
            # learn the frequency map from the training data
            freq_map = df[col].value_counts(normalize=True)
            fit_maps[col] = freq_map
            # apply the map to the training data
            df[col + '_freq'] = df[col].map(freq_map)
        else:
            # apply the learned map from training to the new data
            freq_map = fit_maps.get(col)
            if freq_map is not None:
                df[col + '_freq'] = df[col].map(freq_map)

        # fill unseen categories with 0 frequency and drop original
        if col + '_freq' in df.columns:
            df[col + '_freq'].fillna(0, inplace=True)
        df.drop(col, axis=1, inplace=True)

    if is_fitting:
        return df, fit_maps
    else:
        return df

def target_encode(df, target_cols_list, n_splits=5, fit_maps=None):
    """
    Target encoding with CV for fitting and mapping for transforming the test data.
    Prevents data leakage by ensuring that for each row, the encoding is computed
    without using its own target value.
    """
    df = df.copy()

    # which columns have to be encoded, which is the target
    features_to_encode = list(target_cols_list.keys())
    target_col = list(target_cols_list.values())[0]
    # check if we're fitting or just transforming
    is_fitting = fit_maps is None

    # fitting: learn encodingse from train data
    if is_fitting:
        fit_maps = {} # store to apply to test data
        global_mean = df[target_col].mean()
        fit_maps['_global_mean'] = global_mean
        for col in features_to_encode:
            # save mapping for test set transform (mean per category)
            fit_maps[col] = df.groupby(col, observed=False)[target_col].mean()
            # out-of-fold encoding for train set with k_fold_split
            df[col + '_target'] = np.nan
            all_indices = np.arange(len(df))
            dummy_y = np.zeros(len(df)) # placeholder for funciton
            val_index_folds = k_fold_split(all_indices, dummy_y, k=n_splits)[0]
            for val_indices in val_index_folds:
                train_indices = np.setdiff1d(all_indices, val_indices)
                train_fold, val_fold = df.iloc[train_indices], df.iloc[val_indices]
                # only use train fold to get means, so no leakage
                target_mean_map = train_fold.groupby(col, observed=False)[target_col].mean()
                fold_global_mean = train_fold[target_col].mean()
                # map means to val fold, fill unknowns with fold global mean
                mapped_values = val_fold[col].map(target_mean_map).astype(float).fillna(fold_global_mean)
                df.loc[val_fold.index, col + '_target'] = mapped_values
            # fill remaining NaNs (unseen categories) with global mean
            df[col + '_target'] = df[col + '_target'].astype(float).fillna(global_mean)
        df.drop(columns=features_to_encode, inplace=True)
        return df, fit_maps

    else:
        # transform using train mappings
        global_mean = fit_maps['_global_mean']
        for col in features_to_encode:
            mapping = fit_maps[col]
            # fill unknowns with global mean
            df[col + '_target'] = df[col].map(mapping).astype(float).fillna(global_mean)
        df.drop(columns=features_to_encode, inplace=True)
        return df

def encode_features(df, encoding_strategies, fit_params=None):
    """
    Encodes features of the data set in on bundled together function.
    'fit' (learning/training data) or 'transform' (applying/test data).
    """
    df = df.copy()
    is_fitting = fit_params is None
    if is_fitting:
        fit_params = {}

    # ordinal encoding is stateless, we give the order
    if 'ordinal' in encoding_strategies:
        df = ordinal_encode(df, encoding_strategies['ordinal'])

    # target encoding with fit/transform logic
    if 'target' in encoding_strategies:
        if is_fitting:
            df, target_maps = target_encode(df, encoding_strategies['target'])
            fit_params['target_maps'] = target_maps
        else:
            df = target_encode(df, encoding_strategies['target'], fit_maps=fit_params.get('target_maps'))

    # frequency encoding with fit/transform logic
    if 'frequency' in encoding_strategies:
        freq_cols = encoding_strategies['frequency']
        if is_fitting:
            df, freq_maps = frequency_encode(df, freq_cols)
            fit_params['freq_maps'] = freq_maps
        else:
            df = frequency_encode(df, freq_cols, fit_maps=fit_params.get('freq_maps'))

    # one-hot encoding of cols
    if 'one-hot' in encoding_strategies:
        one_hot_config = encoding_strategies['one-hot']
        df = one_hot_encode(df, one_hot_config)
        if is_fitting:
            # save columns, so test set matches train set
            fit_params['one_hot_columns'] = df.columns.tolist()
        else:
            #fill missing cols with 0 in test data (i.e. unseen categories in training process)
            df = df.reindex(columns=fit_params.get('one_hot_columns', df.columns), fill_value=0)

    if is_fitting:
        return df, fit_params
    else:
        return df