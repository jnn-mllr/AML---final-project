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
    file_name = 'data/adult.data'
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    try:
        df = load_or_download_csv(file_name, url, column_names)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
   
    # fix the ? values -> should be nan
    df.replace(" ?", pd.NA, inplace=True)
    # convert target variable "income" to binary: 1 if >50K, 0 else
    df["income"] = df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)
    return df