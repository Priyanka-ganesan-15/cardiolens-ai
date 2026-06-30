import pandas as pd

def load_data(path="heart_failure_clinical_records_dataset.csv"):
    return pd.read_csv(path)