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

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential

import matplotlib.pyplot as plt

# constants
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

EPOCHS = 200
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2

# flags
USE_DIRECTION_FLAG = True
TRAIN_NN_FLAG = True
TRAIN_ML_FLAG = False
USE_PLOT_FLAG = False
USE_COORDS_FLAG = True


def load_data(folder):
    csv_files = glob.glob(folder + "/*.csv")
    df_list = (pd.read_csv(file) for file in csv_files)
    df = pd.concat(df_list, ignore_index=True)
    return df


def enrich_with_ap_coords(df, ap_coords):
    newarray = []
    for col in df.columns:
        newarray.append(col)
        if "AP05" in col or "AP06" in col:
            newarray.append(col + "_x")
            newarray.append(col + "_y")
            newarray.append(col + "_z")

    newdf = pd.DataFrame(columns=newarray)
    for index, row in df.iterrows():
        newrow = []
        ap_coords_index = 0
        for col_name in df.columns:
            newrow.append(row[col_name])
            if ('AP05' in col_name or 'AP06' in col_name) and row[col_name] in row.values:
                newrow.append(ap_coords.iloc[ap_coords_index]["x"])
                newrow.append(ap_coords.iloc[ap_coords_index]["y"])
                newrow.append(ap_coords.iloc[ap_coords_index]["z"])
                ap_coords_index += 1
        newdf.loc[index] = newrow
    return newdf


def split_data(df, test_size=0.2, random_state=0):
    X = df.loc[:, ~df.columns.isin(["x", "y", "z"])]
    if TRAIN_NN_FLAG:
        y = df[["x", "y", "z"]]
    else:
        y = df[["x"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def normalize_rssi(df):
    for column in df.columns:
        if (
            "_x" not in column
            and "_y" not in column
            and "_z" not in column
            and column != "x"
            and column != "y"
            and column != "z"
            and column != "North"
            and column != "East"
            and column != "South"
            and column != "West"
        ):
            df[column] = df[column].apply(
                lambda x: (x - RSSI_MIN) / (RSSI_MAX - RSSI_MIN)
                if x != NO_SIGNAL
                else 0
            )
    return df


def normalize_xyz(df):
    for column in df.columns:
        if "x" == column or "_x" in column:
            df[column] = df[column].apply(lambda x: x / X_MAX)
        elif "y" == column or "_y" in column:
            df[column] = df[column].apply(lambda x: x / Y_MAX)
        elif "z" == column or "_z" in column:
            df[column] = df[column].apply(lambda x: x / Z_MAX)
    return df

def preprocess_data(df):
    df = normalize_rssi(df)
    df = normalize_xyz(df)
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


def train(df, target):
    X_train, X_test, y_train, y_test = split_data(
        df, target=target, test_size=0.2, random_state=0
    )
    RF = RandomForestRegressor()
    DT = DecisionTreeRegressor()
    SVM = SVR()
    KNN = KNeighborsRegressor()
    RF.fit(X_train, y_train)
    DT.fit(X_train, y_train)
    SVM.fit(X_train, y_train)
    KNN.fit(X_train, y_train)
    y_pred_RF = RF.predict(X_test)
    y_pred_DT = DT.predict(X_test)
    y_pred_SVM = SVM.predict(X_test)
    y_pred_KNN = KNN.predict(X_test)
    return y_pred_RF, y_pred_DT, y_pred_SVM, y_pred_KNN, y_test


def evaluate_model(y_test, y_pred, name):
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mdae = median_absolute_error(y_test, y_pred)
    print_results((r2, mse, mae, mdae), name)


def print_results(eval_tuple, model_name):
    print(model_name)
    print("R2: ", eval_tuple[0])
    print("MSE: ", eval_tuple[1])
    print("MAE: ", eval_tuple[2])
    print("MDAE: ", eval_tuple[3])


def main():
    # load the data
    DF_F5 = load_data(F5_PATH)
    DF_F6 = load_data(F6_PATH)
    df = pd.concat([DF_F5, DF_F6], ignore_index=True)

    if USE_COORDS_FLAG:
        ap_coords = pd.read_csv("data/AP.csv", header=0)
        df = enrich_with_ap_coords(df, ap_coords)

    # add directions with one hot encoding
    df = add_directions(df)

    # preprocess the data
    df = preprocess_data(df)

    if TRAIN_ML_FLAG:
        # train and evaluate the models
        y_pred_RF, y_pred_DT, y_pred_SVM, y_pred_KNN, y_test = train(df, target=X_COL)
        evaluate_model(y_test, y_pred_RF, "Random Forest")
        evaluate_model(y_test, y_pred_DT, "Decision Tree")
        evaluate_model(y_test, y_pred_SVM, "Support Vector Machine")
        evaluate_model(y_test, y_pred_KNN, "K-Nearest Neighbors")
        print("--------------------------------------------------")

        y_pred_RF, y_pred_DT, y_pred_SVM, y_pred_KNN, y_test = train(df, target=Y_COL)
        evaluate_model(y_test, y_pred_RF, "Random Forest")
        evaluate_model(y_test, y_pred_DT, "Decision Tree")
        evaluate_model(y_test, y_pred_SVM, "Support Vector Machine")
        evaluate_model(y_test, y_pred_KNN, "K-Nearest Neighbors")
        print("--------------------------------------------------")

        y_pred_RF, y_pred_DT, y_pred_SVM, y_pred_KNN, y_test = train(df, target=Z_COL)
        evaluate_model(y_test, y_pred_RF, "Random Forest")
        evaluate_model(y_test, y_pred_DT, "Decision Tree")
        evaluate_model(y_test, y_pred_SVM, "Support Vector Machine")
        evaluate_model(y_test, y_pred_KNN, "K-Nearest Neighbors")
        print("--------------------------------------------------")

    if TRAIN_NN_FLAG:
        # split data
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=0)

        if USE_COORDS_FLAG:
            input_dim = X_train.shape[1]
            model = Sequential()
            model.add(BatchNormalization(input_shape=(input_dim,)))
            model.add(Dense(128, input_dim=input_dim, activation="relu"))
            model.add(Dense(64, input_dim=input_dim, activation="relu"))
            model.add(Dense(16, input_dim=input_dim, activation="relu"))
            model.add(Dense(3, activation="linear"))
            model.compile(
                loss="mean_squared_error", optimizer="adam", metrics=["mse", "mae"]
            )
            history = model.fit(
                X_train,
                y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=VALIDATION_SPLIT,
                shuffle=True,
            )
        else:
            input_dim = X_train.shape[1]
            model = Sequential()
            model.add(Dense(256, input_dim=input_dim, activation="relu"))
            model.add(Dropout(0.1))
            model.add(Dense(128, input_dim=input_dim, activation="relu"))
            model.add(Dropout(0.1))
            model.add(Dense(3, activation="linear"))
            model.compile(
                loss="mean_squared_error", optimizer="adam", metrics=["mse", "mae"]
            )
            history = model.fit(
                X_train,
                y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=VALIDATION_SPLIT,
                shuffle=True,
            )

        model.summary()

        if USE_PLOT_FLAG:
            # plot the loss and validation loss of the dataset
            plt.figure(figsize=(10, 5))
            plt.plot(history.history["loss"], label="Training Loss")
            plt.plot(history.history["val_loss"], label="Validation Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.show()

        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Mean Squared Error : ", scores[1])
        print("Mean Absolute Error : ", scores[2])

        # make predictions
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print("r2 score: ", r2.round(5) * 100, "%")

        if USE_PLOT_FLAG:
            y_pred = y_pred.flatten()
            plt.scatter(y_test, y_pred)
            plt.axes(aspect="equal")
            plt.xlabel("True values")
            plt.ylabel("Predicted values")
            plt.xlim([0, 50000])
            plt.ylim([0, 50000])
            plt.plot([0, 50000], [0, 50000])
            plt.plot()
            plt.savefig("scatter.png")


if __name__ == "__main__":
    main()
