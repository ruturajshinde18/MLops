import pandas as pd
import numpy as np

import os

os.makedirs("Data/raw", exist_ok=True)

def load_data(path):
    df = pd.read_csv(path)
    return df
def export_data(df, path):
    df.to_csv(path, index=False)

if __name__ == "__main__":
    data =(load_data("input/heart.csv"))
    export_data(data, "Data/raw/raw.csv")
