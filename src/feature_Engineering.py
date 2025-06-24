from sklearn.model_selection import train_test_split
import pandas as pd
import os

os.makedirs("Data/feature", exist_ok=True)

def split_data(path):
    data = pd.read_csv(path)
    x = data.drop(columns=["target"])
    y = data["target"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=18)
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
     x_train,x_test,y_train,y_test =split_data("Data/preprocessed/preprocessed.csv")
     pd.DataFrame(x_train).to_csv("Data/feature/x_train.csv", index=False)
     pd.DataFrame(x_test).to_csv("Data/feature/x_test.csv", index=False)
     pd.DataFrame(y_train).to_csv("Data/feature/y_train.csv", index=False)
     pd.DataFrame(y_test).to_csv("Data/feature/y_test.csv", index=False)