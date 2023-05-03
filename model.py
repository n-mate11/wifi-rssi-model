import glob
import numpy as np
import pandas as pd

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

import matplotlib.pyplot as plt

import pickle

F5_PATH = "./data/F5/"
F6_PATH = "./data/F6/"

RSSI_MIN = -100
RSSI_MAX = 0
NO_SIGNAL = 1

NUMBER_OF_APS = 325

X_COL = 325
Y_COL = 326
Z_COL = 327

NORTH_COL = 328
EAST_COL = 329
SOUTH_COL = 330
WEST_COL = 331

X_MAX = 70
Y_MAX = 70
Z_MAX = 12


def load_data(folder):
    csv_files = glob.glob(folder + "/*.csv")
    df_list = (pd.read_csv(file) for file in csv_files)
    df = pd.concat(df_list, ignore_index=True)
    return df


def split_data(df, target, test_size=0.2, random_state=0):
    X = df.iloc[:, 0 : NUMBER_OF_APS - 1]
    y = df.iloc[:, target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def normalize_rssi(df, columns):
    for column in columns:
        df[column] = df[column].apply(
            lambda x: (x - RSSI_MIN) / (RSSI_MAX - RSSI_MIN) if x != NO_SIGNAL else 0
        )
    return df


def normalize_xy(df, columns):
    for column in columns:
        df[column] = df[column].apply(
            lambda x: x / X_MAX if column == "x" else x / Y_MAX
        )
    return df


def normalize_z(df, columns):
    for column in columns:
        df[column] = df[column].apply(lambda x: x / Z_MAX)
    return df


def preprocess_data(df):
    df = normalize_rssi(df, df.columns[0 : NUMBER_OF_APS - 1])
    df = normalize_xy(df, df.columns[NUMBER_OF_APS - 1 : NUMBER_OF_APS + 1])
    df = normalize_z(df, df.columns[NUMBER_OF_APS + 1 : NUMBER_OF_APS + 2])
    return df


def add_directions(df):
    df["North"] = 0
    df["East"] = 0
    df["South"] = 0
    df["West"] = 0
    for i in range(0, len(df), 60):
        df["North"].iloc[i : i + 15] = 1
        df["East"].iloc[i + 15 : i + 30] = 1
        df["South"].iloc[i + 30 : i + 45] = 1
        df["West"].iloc[i + 45 : i + 60] = 1
    return df


def evaluate_model(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mdae = median_absolute_error(y_test, y_pred)
    return r2, mse, mae, mdae


def main():
    # load the data
    DF_F5 = load_data(F5_PATH)
    DF_F6 = load_data(F6_PATH)
    df = pd.concat([DF_F5, DF_F6], ignore_index=True)

    # add directions with one hot encoding
    df = add_directions(df)

    # preprocess the data with MinMaxScaler
    df = preprocess_data(df)

    # split the data
    X_train, X_test, y_train, y_test = split_data(df, target=(X_COL-1), test_size=0.2, random_state=0)

    # models
    RF = RandomForestRegressor()
    DT = DecisionTreeRegressor()
    SVM = SVR()
    KNN = KNeighborsRegressor()

    # train the models
    RF.fit(X_train, y_train)
    DT.fit(X_train, y_train)
    SVM.fit(X_train, y_train)
    KNN.fit(X_train, y_train)

    # predict the test data
    y_pred_RF = RF.predict(X_test)
    y_pred_DT = DT.predict(X_test)
    y_pred_SVM = SVM.predict(X_test)
    y_pred_KNN = KNN.predict(X_test)

    # evaluate the models
    r2_RF, mse_RF, mae_RF, mdae_RF = evaluate_model(y_test, y_pred_RF)
    r2_DT, mse_DT, mae_DT, mdae_DT = evaluate_model(y_test, y_pred_DT)
    r2_SVM, mse_SVM, mae_SVM, mdae_SVM = evaluate_model(y_test, y_pred_SVM)
    r2_KNN, mse_KNN, mae_KNN, mdae_KNN = evaluate_model(y_test, y_pred_KNN)

    # print the metrics
    print("Random Forest")
    print("R2: ", r2_RF)
    print("MSE: ", mse_RF)
    print("MAE: ", mae_RF)
    print("MDAE: ", mdae_RF)

    print("Decision Tree")
    print("R2: ", r2_DT)
    print("MSE: ", mse_DT)
    print("MAE: ", mae_DT)
    print("MDAE: ", mdae_DT)

    print("SVM")
    print("R2: ", r2_SVM)
    print("MSE: ", mse_SVM)
    print("MAE: ", mae_SVM)
    print("MDAE: ", mdae_SVM)

    print("KNN")
    print("R2: ", r2_KNN)
    print("MSE: ", mse_KNN)
    print("MAE: ", mae_KNN)
    print("MDAE: ", mdae_KNN)

    # # save the model in pickle format
    # pickle.dump(regressor, open('model.pkl','wb'))


if __name__ == "__main__":
    main()
