# AML---final-project
final project for Applied Machine Learning for the Adult Data Set
# Income Prediction Using Categorical Data

## Project Overview

This project predicts whether an individual earns more than \$50,000/year using demographic and employment attributes from the UCI Adult Income dataset. Since most features are categorical, we explore how different encoding strategies impact both performance and interpretability in linear models.

We compare **one-hot, ordinal, frequency, and target encoding (with cross-validation leakage control)** and implement custom machine learning models (e.g., logistic regression) from scratch, fully integrated into a modular pipeline.

## Project Structure

```
project/
│project/
├── main.ipynb                        # Full end-to-end notebook: loading, encoding, training, evaluation
├── requirements.txt                  # List of Python dependencies
├── data/                             # Raw data files (adult.data, adult.test)
├── encoding/                         # Custom encoding modules (one-hot, ordinal, frequency, target with CV)
├── models/                           # Implementations of logistic regression, linear SVM, etc.
├── tune/                             # Grid search and hyperparameter tuning
├── utils/
│ ├── loaders.py                      # Data loading and parsing
│ ├── preprocessing.py                # Feature engineering and transformations
│ ├── normalization.py                # Custom scaler (standardization)
│ ├── splits.py                       # Stratified train-test split
│ ├── metrics.py                      # Accuracy, precision, recall, confusion matrix
│ ├── training.py                     # Model training wrapper and CV logic
```

## 📊 Encoding Strategies
| Encoding    | Description                                                     | Use Case                            |
| ----------- | --------------------------------------------------------------- | ----------------------------------- |
| One-hot     | Convert each category to separate binary columns                | Nominal features                    |
| Ordinal     | Map ordered categories to integers                              | Ordered features (e.g. education)   |
| Frequency   | Replace category with frequency in training set                 | High-cardinality features           |
| Target (CV) | Replace category with mean target, computed using CV for safety | Powerful, leakage-prone if careless |


## Results

| Encoding     | Model                  | Accuracy | Precision | Recall |
|--------------|------------------------|----------|-----------|--------|
| One-hot      | Logistic (base)        | 0.11     | 0.11      | 0.11   |
| Ordinal      | Logistic (ridge)       | 0.11     | 0.11      | 0.11   |
| Target (CV)  | Logistic (tuned)       | 0.11     | 0.11      | 0.11   |


## 🧪 Models & Evaluation

    Logistic Regression (no/weak/strong ridge/lasso regularization)

    Training method: Batch Gradient Descent

    Metrics: Accuracy, Precision, Recall

    Leakage prevention: Target encoding only uses out-of-fold statistics

    Scaling: StandardScaler applied to numeric columns

## 📌 Best Model (TO FILL)

    Type: Logistic Regression

    Penalty: TO_FILL (none / ridge / lasso)

    λ: TO_FILL

    Accuracy: TO_FILL

    Precision: TO_FILL

    Recall: TO_FILL


## References
~ UCI Adult Census Income

~ https://archive.ics.uci.edu/ml/datasets/adult

## Author
```
Janne Miller 
Yupeng Cheng
```
