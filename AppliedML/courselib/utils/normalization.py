import numpy as np

def standardize(x):
    """
    Standardization normalization.
    Scales the data to have mean 0 and standard deviation 1.
    Parameters:
    - x: numpy array of shape (n_samples, n_features)
    Returns:
    - numpy array of the same shape as x, with mean 0 and std 1
    """
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)


def min_max(x):
    """
    Min-Max normalization.
    Scales the data to a range of [0, 1] by transforming each feature individually.
    Parameters:
    - x: numpy array of shape (n_samples, n_features)
    Returns:
    - numpy array of the same shape as x, with all values scaled to the range [0, 1]
    """
    return (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))

# our implementations
class StandardScaler:
    """
    Standardizes features by removing the mean and scaling to variance 1.
    This class is needed to avoid data leakage when applying the normal
    standardization without separate fi and transform steps. 
    """
    def __init__(self, numerical_indices=None):
        """
        Initializes the StandardScaler.
        """
        self.mean_ = None
        self.scale_ = None
        self.numerical_indices = numerical_indices

    def fit(self, X):
        """
        Compute the mean and standard deviation of the features
        """
        # select data
        data_to_fit = X[:, self.numerical_indices] if self.numerical_indices is not None else X
        
        self.mean_ = np.mean(data_to_fit, axis=0)
        self.scale_ = np.std(data_to_fit, axis=0)
        # avoid division by zero (if we have 0 var in a variable, usually shouldnt happen) 
        self.scale_[self.scale_ == 0] = 1

    def transform(self, X):
        """
        Perform standardization.
        """

        # avoid modifying the original array
        X_scaled = X.copy()
        # transform only the specified columns (indices provided)
        if self.numerical_indices is not None:
            X_scaled[:, self.numerical_indices] = (X[:, self.numerical_indices] - self.mean_) / self.scale_
        else:
            X_scaled = (X - self.mean_) / self.scale_
        return X_scaled

    def fit_transform(self, X):
        """
        Fit to data, then transform.
        """
        self.fit(X)
        return self.transform(X)

def scale_numerical_features(X_train, X_test, train_df_encoded, numerical_cols):
    """
    Scales only the specified numerical columns in X_train and X_test using StandardScaler.
    Returns the scaled versions of X_train and X_test, and the scaler object.
    """
    # column indices for numerical columns
    X_columns = train_df_encoded.drop('income', axis=1).columns
    numerical_indices = [X_columns.get_loc(col) for col in numerical_cols]

    # initialize and fit scaler
    scaler = StandardScaler(numerical_indices=numerical_indices)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled