## 🧠 Income Prediction Using Categorical Data

### 🎯 Project Objective

This project aims to **predict whether an individual's income exceeds \$50K per year**, based on U.S. census demographic data. We emphasize:
- **Binary classification**
- **Model interpretability and reproducibility**
- **Systematic comparison of categorical encoding strategies**

---

### 📦 Dataset

- **Source:** [UCI Adult Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- **Rows:** 48,842 instances  
- **Features:**  
  - **Numerical:** `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`  
  - **Categorical:** `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`  
- **Target:** `income` (binary: `>50K` vs `<=50K`)  
- **Challenge:** Highly imbalanced classes and many high-cardinality categorical variables

---

### 🚀 Models & Methods

- Logistic Regression
<<<<<<< HEAD
    - **interpretable, baseline model, regularization (L1, L2, elasticnet)**

---
=======
    - **interpretable, baseline model, supports regularization (L1, L2)**
>>>>>>> bc57715c92d3de495382cb09bf455db8310e379b

### 🔧 Encoding Strategies

- One-Hot Encoding
    - **high-dimensional, sparse representation**

- Ordinal Encoding
    - **order imposed, may introduce artificial relationship**

- Frequency Encoding
    - **encodes based on category frequency, compact but may lose semantics**

- Target Encoding with Cross-Validation
    - **risk of target leakage, CV necessary to avoid data leakage**

---

### 📌 How to Run

1. Clone this repository
2. Install dependencies (see `requirements.txt` or use `conda`)
3. Run the notebook:
   ```bash
   jupyter notebook notebooks/main.ipynb

---

### 📈 Key Results

- **Best accuracy**: ~0.85 (Logistic Regression with L2 and Target Encoding)
- **Target Encoding (with CV)**: offered the best trade-off between compactness and accuracy
- **Interpretable model**: via odds-ratios from logistic coefficients

---

### 👥 Authors
- 🧑‍💻 Janne Miller — j.miller@campus.lmu.de 
- 🧑‍💻 Yupeng Cheng — y.cheng1@campus.lmu.de  

