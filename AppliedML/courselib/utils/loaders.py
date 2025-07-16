import os
import pandas as pd

def load_or_download_csv(file_name, url, column_names=None, encoding='utf-8'):
    if os.path.exists(file_name):
        print(f"Loading from local `{file_name}`...")
        return pd.read_csv(file_name, index_col=0, encoding=encoding)
    else:
        print(f"Downloading from `{url}`...")
        df = pd.read_csv(url, names=column_names, encoding=encoding)
        df.to_csv(file_name, encoding=encoding)
        print("Saved to local file.")
        return df

def load_uciadult():
    url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    url_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    file_train = 'data/adult.data'
    file_test = 'data/adult.test'
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    try:
        df_train = load_or_download_csv(file_train, url_train, column_names=column_names)
        df_test = load_or_download_csv(file_test, url_test, column_names=column_names)
        df = pd.concat([df_train, df_test], ignore_index=True)
        print("Full dataset shape (rows, columns):", df.shape)    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
   
    # fix the ? values -> should be nan
    df = df.replace(" ?", pd.NA)
    # target variable "income" to binary: 1 if >50K or >50K., 0 else
    df["income"] = df["income"].apply(
        lambda x: 1 if isinstance(x, str) and x.strip().replace('.', '') == ">50K" else 0
    )
    return df