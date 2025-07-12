from utils.metrics import confusion_matrix
import time
import numpy as np

import itertools

class ModelEvaluator:
    """
    Wrapper class to standardize training and evaluation of different models.
    """
    def __init__(self, model, model_name, metrics_dict, plot_cm=True, **fit_params):
        self.model = model
        self.name = model_name
        self.fit_params = fit_params
        self.plot_cm = plot_cm # whether to plot the confusion matrix
        self.metrics_dict = metrics_dict
        self.training_time = 0.0
        self.use_svm_labels = 'SVM' in self.name # detect if we include an SVM -> requires different logic

    def train(self, X_train, y_train):
        """
        Trains the model on the data and measures training time.

        For SVMs: conversion of labels from {0, 1} to {-1, 1}.

        Args:
            X_train: The training feature data.
            y_train: The training target.
        """
        print(f"--- Training {self.name} ---")
        
        y_train_final = y_train
        if self.use_svm_labels:
            y_train_svm = y_train.copy()
            y_train_svm[y_train == 0] = -1
            y_train_final = y_train_svm

        start_time = time.time()
        self.model.fit(X_train, y_train_final, **self.fit_params)
        self.training_time = time.time() - start_time


    def evaluate(self, X_test, y_test):
        """
        Evaluates the model using the metrics dictionary and returns the results.
        Args:
            X_test: The test feature data.
            y_test: The test target.

        Returns:
            performance: dictionary containing the model's name, training time,
                            and the scores for each specified metric.
        """
        # prections of the test data
        y_pred = self.model(X_test)

        if self.use_svm_labels:
            y_pred[y_pred == -1] = 0
        
        # calc specified metrics
        performance = {'Model': self.name, 'Training Time (s)': self.training_time}
        for metric_name, metric_func in self.metrics_dict.items():
            score = metric_func(y_test, y_pred)
            performance[metric_name] = score
            print(f"{metric_name}: {score:.4f}")

        # plot confusion matrix
        if self.plot_cm:
            self.plot_confusion_matrix(y_test, y_pred)
        return performance

    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plots a confusion matrix using the courselib implementation.
        """
        class_names = ['<=50K', '>50K']
        title = f'Confusion Matrix for {self.name}'
        confusion_matrix(
            y_true, 
            y_pred, 
            num_classes=2, 
            plot=True, 
            class_names=class_names, 
            title=title
        )



# Hyperparameter-Tuning via Grid-Search
class GridSearch:
    """
    A simple grid search implementation to find the best hyperparameters for a model.
    """
    def __init__(self, model_class, param_grid, scoring_metric, **const_params):
        """
        Args:
            model_class: The class of the model to tune (e.g., LinearSVM).
            param_grid (dict): Dictionary with parameter names as keys and lists of
                               parameter settings to try as values.
            scoring_metric (function): The metric to use for evaluation (e.g., f1_score).
            **const_params: Constant parameters for the model initializer.
        """
        self.model_class = model_class
        self.param_grid = param_grid
        self.scoring_metric = scoring_metric
        self.const_params = const_params
        self.best_score = -1
        self.best_params = None
        self.best_model = None

    def fit(self, X_train, y_train, X_val, y_val, **fit_params):
        """
        Runs the grid search.
        
        Args:
            X_train, y_train: Training data.
            X_val, y_val: Validation data for scoring models.
            **fit_params: Constant parameters for the model's fit method.
        """
        # Generate all combinations of parameters
        keys, values = zip(*self.param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        print(f"Starting Grid Search for {self.model_class.__name__}...")
        print(f"Testing {len(param_combinations)} combinations.")

        for params in param_combinations:
            # Combine constant and grid parameters
            current_params = {**self.const_params, **params}
            
            # Create a new model instance with the current parameters
            model = self.model_class(**current_params)
            
            # Train the model
            model.fit(X_train, y_train, **fit_params)
            
            # Evaluate on the validation set
            y_pred = model(X_val)
            
            # Handle SVM label conversion for scoring
            if 'SVM' in self.model_class.__name__:
                y_pred[y_pred == -1] = 0

            score = self.scoring_metric(y_val, y_pred)
            
            print(f"Params: {params} -> Score: {score:.4f}")

            # Update best score and params if current model is better
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                self.best_model = model
        
        print("\n--- Grid Search Complete ---")
        print(f"Best Score: {self.best_score:.4f}")
        print(f"Best Parameters: {self.best_params}")

    def predict(self, X):
        if self.best_model is None:
            raise RuntimeError("You must call 'fit' before 'predict'.")
        return self.best_model(X)