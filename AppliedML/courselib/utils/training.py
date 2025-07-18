import pandas as pd
from utils.metrics import accuracy, confusion_matrix
from utils.splits import k_fold_split
import time
import numpy as np

# new class for training process
class ModelTraining:
    """
    Wrapper for training and evaluation of models - originally to compare
    many different models now just for specifications of logreg :(
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
        self.history = None

    def train(self, X_train, y_train, X_test=None, y_test=None):
        """
        Trains the model on the data, measures training time.
        """       
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
        Evaluates the model using the metrics dictionary (here Acc, Recall and Precision),
        but could also be extended to include other params
        """
        # predictions of the test data
        y_pred = self.model(X_test)
        
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
        Plots a confusion matrix using the courselib implementation of cm.
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
        """
        print(f"CV for {self.model_name} with {k_folds} folds")
        X_folds, y_folds = k_fold_split(X, y, k=k_folds)
        fold_metrics = {metric_name: [] for metric_name in self.metrics_dict.keys()}
        fold_metrics['Training Time (s)'] = []

        for i in range(k_folds):
            # create cv splits
            X_train_fold = np.concatenate([X_folds[j] for j in range(k_folds) if j != i])
            y_train_fold = np.concatenate([y_folds[j] for j in range(k_folds) if j != i])
            X_val_fold, y_val_fold = X_folds[i], y_folds[i]

            # re-initialize the model for each fold to ensure it's fresh
            current_model = self.model_class(**self.init_params)

            # train model on current fold
            fold_trainer = ModelTraining(
                model_class=self.model_class,
                init_params=self.init_params,
                model_name=self.model_name,
                metrics_dict=self.metrics_dict,
                plot_cm=False,  # no plotting for every fold
                **self.fit_params
            )
            fold_trainer.model = current_model # assign the fresh model
            fold_trainer.train(X_train_fold, y_train_fold)
            performance = fold_trainer.evaluate(X_val_fold, y_val_fold)
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
    
    