import numpy as np


class StandardScaler:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    """

    def __init__(self, numerical_indices=None):
        """
        Initializes the StandardScaler.
        
        Args:
            numerical_indices (list of int, optional): A list of indices for the columns to scale.
                                                     If None, all columns are scaled. Defaults to None.
        """
        self.mean_ = None
        self.scale_ = None
        self.numerical_indices = numerical_indices

    def fit(self, X):
        """
        Compute the mean and standard deviation to be used for later scaling.

        Args:
            X (np.ndarray): The data used to compute the mean and standard deviation.
                            Shape (n_samples, n_features)
        """
        # If indices are provided, slice the data to compute stats only on those columns
        data_to_fit = X[:, self.numerical_indices] if self.numerical_indices is not None else X
        
        self.mean_ = np.mean(data_to_fit, axis=0)
        self.scale_ = np.std(data_to_fit, axis=0)
        # Avoid division by zero for features with zero variance
        self.scale_[self.scale_ == 0] = 1

    def transform(self, X):
        """
        Perform standardization by centering and scaling.

        Args:
            X (np.ndarray): The data to scale. Shape (n_samples, n_features)

        Returns:
            np.ndarray: The scaled data.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("You must fit the scaler before transforming the data.")

        # Create a copy to avoid modifying the original array
        X_scaled = X.copy()

        # If indices are provided, transform only those columns
        if self.numerical_indices is not None:
            X_scaled[:, self.numerical_indices] = (X[:, self.numerical_indices] - self.mean_) / self.scale_
        else:
            X_scaled = (X - self.mean_) / self.scale_
            
        return X_scaled

    def fit_transform(self, X):
        """
        Fit to data, then transform it.

        Args:
            X (np.ndarray): The data to fit and scale. Shape (n_samples, n_features)

        Returns:
            np.ndarray: The scaled data.
        """
        self.fit(X)
        return self.transform(X)


class MinMaxScaler:
    """
    MinMaxScaler scales features to a given range, typically [0, 1].
    """

    def __init__(self):
        self.min_ = None
        self.scale_ = None

    def fit(self, X):
        """
        Compute the minimum and maximum to be used for later scaling.
        """
        self.min_ = np.min(X, axis=0)
        self.scale_ = np.max(X, axis=0) - self.min_
        # To avoid division by zero
        self.scale_[self.scale_ == 0] = 1
        return self

    def transform(self, X):
        """
        Scale features of X according to feature_range.
        """
        if self.min_ is None or self.scale_ is None:
            raise RuntimeError("You must fit the scaler before transforming data.")
        return (X - self.min_) / self.scale_

    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        """
        return self.fit(X).transform(X)


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