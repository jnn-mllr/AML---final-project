import pandas as pd
from utils.metrics import accuracy, confusion_matrix
from utils.splits import k_fold_split
import time
import numpy as np
import itertools

class ModelTraining:
    """
    Wrapper for training and evaluation of models.
    """
    def __init__(self, model_class, model_name, init_params, metrics_dict, plot_cm=True, **fit_params):
        # create the model
        self.model_class = model_class
        self.init_params = init_params
        self.model_name = model_name
        self.fit_params = fit_params
        self.model = self.model_class(**self.init_params)

        # other parameters
        self.plot_cm = plot_cm # whether to plot the confusion matrix
        self.metrics_dict = metrics_dict
        self.training_time = 0.0
        self.use_svm_labels = 'SVM' in self.model_name # detect if we include an SVM -> requires different logic
        self.history = None

    def train(self, X_train, y_train, X_test=None, y_test=None):
        """
        Trains the model on the data and measures training time.

        For SVMs: conversion of labels from {0, 1} to {-1, 1}.

        Args:
            X_train: The training feature data.
            y_train: The training target.
        """        
        y_train_final = y_train
        if self.use_svm_labels:
            y_train[y_train.copy() == 0] = -1

        start_time = time.time()
        # re-initialize the model before training to ensure no leakage for cv
        self.model = self.model_class(**self.init_params)
        self.history = self.model.train(
            X_train, y_train, 
            X_test, y_test, 
            **self.fit_params
        )        
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
        performance = {'Model': self.model_name, 'Training Time (s)': self.training_time}
        for metric_name, metric_func in self.metrics_dict.items():
            score = metric_func(y_test, y_pred)
            performance[metric_name] = score

        # plot confusion matrix
        if self.plot_cm:
            self.plot_confusion_matrix(y_test, y_pred)
        return performance

    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plots a confusion matrix using the courselib implementation.
        """
        class_names = ['<=50K', '>50K']
        title = f'Confusion Matrix for {self.model_name}'
        confusion_matrix(
            y_true, 
            y_pred, 
            num_classes=2, 
            plot=True, 
            class_names=class_names, 
            title=title
        )
        
    def cross_validate(self, X, y, k_folds=5):
        """
        Performs k-fold cross-validation.

        Args:
            X (np.ndarray): The full training feature dataset.
            y (np.ndarray): The full training label dataset.
            k_folds (int): The number of folds to use for cross-validation.

        Returns:
            avg_performance: dictionary with mean and std for each metric.
        """
        print(f"CV for {self.model_name} with {k_folds} folds")
        X_folds, y_folds = k_fold_split(X, y, k=k_folds)
        
        fold_metrics = {metric_name: [] for metric_name in self.metrics_dict.keys()}
        fold_metrics['Training Time (s)'] = []

        for i in range(k_folds):
            print(f"Fold {i+1}")
            X_train_fold = np.concatenate([X_folds[j] for j in range(k_folds) if j != i])
            y_train_fold = np.concatenate([y_folds[j] for j in range(k_folds) if j != i])
            X_val_fold, y_val_fold = X_folds[i], y_folds[i]

            # train model on current fold
            fold_evaluator = ModelTraining(
                model_class=self.model_class,
                init_params=self.init_params,
                model_name=self.model_name,
                metrics_dict=self.metrics_dict,
                plot_cm=False,  # no plotting for every fold
                **self.fit_params
            )

            fold_evaluator.train(X_train_fold, y_train_fold)
            performance = fold_evaluator.evaluate(X_val_fold, y_val_fold)
            
            for metric_name in fold_metrics.keys():
                fold_metrics[metric_name].append(performance[metric_name])

        # average performance across all folds
        avg_performance = {'Model': self.model_name}
        for metric_name, scores in fold_metrics.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            avg_performance[f"Mean {metric_name}"] = mean_score
            avg_performance[f"Std {metric_name}"] = std_score        
        return avg_performance



# Hyperparameter-Tuning via Grid-Search
class GridSearch:
    """
    Simple grid search implementation to find the best hyperparameters for a model.
    Does not use scikit-learn.
    """
    def __init__(self, model_class, param_grid, fixed_params=None, scoring='accuracy', cv=3):
        self.model_class = model_class
        self.param_grid = param_grid
        self.fixed_params = fixed_params if fixed_params else {}
        self.scoring = scoring
        self.cv = cv
        self.cv_results_ = None
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_model_ = None
        self.preprocess_params = None

        self.scoring_metric = self._get_scoring_function(scoring)

    def _get_scoring_function(self, scoring_name):
        if scoring_name == 'accuracy':
            return lambda y_true, y_pred: accuracy(y_pred, y_true, one_hot_encoded_labels=False)
        else:
            raise ValueError(f"Scoring method '{scoring_name}' not supported.")

    def fit(self, X, y, X_val=None, y_val=None):
        # Generate all combinations of parameters without sklearn
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        results_list = []

        for params in param_combinations:
            current_params = {**self.fixed_params, **params}
            
            if self.preprocess_params and callable(self.preprocess_params):
                current_params = self.preprocess_params(current_params.copy())

            init_params = {k: v for k, v in current_params.items() if k in self.model_class.__init__.__code__.co_varnames}
            fit_params = {k: v for k, v in current_params.items() if k not in init_params}
            
            model = self.model_class(**init_params)
            
            # Use the correct variable names X and y from the method arguments
            model.fit(X, y, **fit_params)
            
            y_pred = model(X_val)
            
            # Handle SVM label conversion for scoring if necessary
            y_pred_eval = y_pred.copy()
            if 'SVM' in self.model_class.__name__:
                y_pred_eval[y_pred_eval == -1] = 0

            score = self.scoring_metric(y_val, y_pred_eval)
            
            print(f"Params: {params} -> Score: {score:.4f}")
            
            run_result = {**params, 'score': score}
            results_list.append(run_result)

            if score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = params
                self.best_model_ = model
        
        self.cv_results_ = pd.DataFrame(results_list)
        print(f"Best Score: {self.best_score_:.4f}")
        print(f"Best Parameters: {self.best_params_}")

    def predict(self, X):
        return self.best_model_(X)