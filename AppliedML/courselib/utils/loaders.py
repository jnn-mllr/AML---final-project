import os
import pandas as pd

# added kwargs and header=False for this data set
def load_or_download_csv(file_path, url, **kwargs):
    """Downloads a file if it doesn't exist, then loads it with pandas."""
    if not os.path.exists(file_path):
        print(f"Downloading from `{url}`...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df_to_save = pd.read_csv(url, **kwargs)
        df_to_save.to_csv(file_path, index=False, header=False)
        print(f"Saved to `{file_path}`")
    
    return pd.read_csv(file_path, **kwargs)

def load_uciadult():
    """Loads, merges, and cleans the UCI Adult dataset."""
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]
    
    # parameters for pd.read_csv to handle quirks in the data files, mainly in the test data
    common_params = {
        "names": column_names,
        "na_values": "?",     # fix the ? values -> should be nan
        "skipinitialspace": True,
    }

    try:
        df_train = load_or_download_csv(
            'data/adult.data',
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
            **common_params
        )
        # test set has an extra header line
        df_test = load_or_download_csv(
            'data/adult.test',
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
            skiprows=1,
            **common_params
        )
        # merge the data sets
        df = pd.concat([df_train, df_test], ignore_index=True)
        # target variable "income" to binary: 1 if >50K, 0 else
        df["income"] = df["income"].str.contains(">50K").fillna(False).astype(int) 
        print("Dataset loaded successfully. Shape:", df.shape)
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None