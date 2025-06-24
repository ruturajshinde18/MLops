import pandas as pd
import numpy as np

import os

os.makedirs("Data/preprocessed", exist_ok=True)

def preprocessing(path):
    data = pd.read_csv(path)
    data.fillna(0, inplace=True)
    return data

def export_data(data,path):
    data.to_csv(path, index=False)



if __name__ == "__main__":
    data =preprocessing("Data/raw/raw.csv")
    export_data(data,"Data/preprocessed/preprocessed.csv")