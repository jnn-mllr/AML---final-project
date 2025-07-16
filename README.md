# Income Prediction Using Categorical Data

### 🎯 Overall Goal

- Predict whether an individual's income exceeds $50K per year:
    - **binary classification task**

- Build interpretable and reproducible machine learning models
    - **model interpretability, reproducibility**

- Explore how different categorical encoding strategies affect model performance
    - **encoding choice impacts accuracy and generalization**

### 📊 Dataset & Features

- UCI Adult dataset
    - **census income data, demographic variables, occupation, education**

- Mixed feature types
    - **numerical features: age, hours-per-week**
    - **categorical features: workclass, education, marital-status, occupation, etc.**

- Class imbalance
    - *more samples with income ≤ $50K*

### 🧠 Models & Methods

- Logistic Regression
    - **interpretable, baseline model, supports regularization (L1, L2, elasticnet)**

### 🔧 Encoding Strategies

- One-Hot Encoding
    - **high-dimensional, sparse representation**

- Ordinal Encoding
    - **order imposed, may introduce artificial relationship**

- Frequency Encoding
    - **encodes based on category frequency, compact but may lose semantics**

- Target Encoding with Cross-Validation
    - **risk of target leakage, CV necessary to avoid data leakage**

### 📂 Files
- [📘 Notebook: Income Prediction Using Categorical Data](notebooks/main.ipynb)

### 🧾 References
~ UCI Adult Census Income

🔗 ~ https://archive.ics.uci.edu/ml/datasets/adult

### 🧑‍🏫 Author
```
Janne Miller 
Yupeng Cheng
```
