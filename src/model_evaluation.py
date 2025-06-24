import pickle

import pandas as pd



def model_evaluation(model,x_test,y_test):
    return model.score(x_test,y_test)


if __name__ == "__main__":
    model = pickle.load(open("model.pkl", "rb"))
    x_test = pd.read_csv("Data/feature/x_test.csv")
    y_test = pd.read_csv("Data/feature/y_test.csv")
    score =model_evaluation(model,x_test,y_test)
    print("abc")
    print(score)