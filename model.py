import glob
import numpy as np
import pandas as pd

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor

# flags
USE_DIRECTION_FLAG = False
TRAIN_NN_FLAG = False
TRAIN_ML_FLAG = True
USE_PLOT_FLAG = False
USE_COORDS_FLAG = False

if TRAIN_NN_FLAG:
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
            if ("AP05" in col_name or "AP06" in col_name) and row[
                col_name
            ] in row.values:
                newrow.append(ap_coords.iloc[ap_coords_index]["x"])
                newrow.append(ap_coords.iloc[ap_coords_index]["y"])
                newrow.append(ap_coords.iloc[ap_coords_index]["z"])
                ap_coords_index += 1
        newdf.loc[index] = newrow
    return newdf


def split_data(df, target=["x", "y", "z"], test_size=0.2, random_state=0):
    X = df.loc[:, ~df.columns.isin(target)]
    if TRAIN_NN_FLAG:
        y = df[target]
    if TRAIN_ML_FLAG:
        y = df[target]
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
            and column != "direction"
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


def normalize_directions(df):
    for column in df.columns:
        if "direction" in column:
            df[column] = df[column].apply(lambda x: x / 360)
    return df


def preprocess_data(df):
    df = normalize_rssi(df)
    df = normalize_xyz(df)
    if USE_DIRECTION_FLAG:
        df = normalize_directions(df)
    return df


def add_directions(df):
    df["direction"] = 0
    for i in range(0, len(df), 60):
        df["direction"].iloc[i : i + 15] = 0
        df["direction"].iloc[i + 15 : i + 30] = 90
        df["direction"].iloc[i + 30 : i + 45] = 180
        df["direction"].iloc[i + 45 : i + 60] = 270
    return df


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


def grid_search(estimator, params, X, y):
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=params,
        scoring="neg_mean_squared_error",
        cv=3,
        return_train_score=True,
        n_jobs=-1,
    )
    grid_search.fit(X, y)
    print("best_params_", grid_search.best_params_)
    print("best_estimator_", grid_search.best_estimator_)
    print("best_score_", grid_search.best_score_)
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    print("--------------------------------------------------")


def plot_3d(y_test, y_pred):
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    ax.scatter3D(y_test["x"], y_test["y"], y_test["z"], color="blue", label="y_test")
    ax.scatter3D(y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], color="red", label="y_pred")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Actual values vs Predicted values")
    plt.show()


def main():
    # load the data
    DF_F5 = load_data(F5_PATH)
    DF_F6 = load_data(F6_PATH)
    df = pd.concat([DF_F5, DF_F6], ignore_index=True)

    if USE_COORDS_FLAG:
        ap_coords = pd.read_csv("data/AP.csv", header=0)
        df = enrich_with_ap_coords(df, ap_coords)

    if USE_DIRECTION_FLAG:
        df = add_directions(df)

    # preprocess the data
    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=0)

    floors_train = y_train[["z"]]
    xy_train = y_train[["x", "y"]]

    if TRAIN_ML_FLAG:
        RF_xy = RandomForestRegressor(
            bootstrap=False,
            criterion="absolute_error",
            max_depth=75,
            max_features="sqrt",
            n_estimators=50,
        )
        RF_z = RandomForestRegressor(
            bootstrap=False,
            criterion="friedman_mse",
            max_depth=60,
            max_features="sqrt",
            n_estimators=255,
        )
        MOR = MultiOutputRegressor(RF_xy)

        y_pred_RF_xy = MOR.fit(X_train, xy_train).predict(X_test)
        y_pred_RF_z = RF_z.fit(X_train, floors_train.values.ravel()).predict(X_test)

        y_pred_RF = np.concatenate((y_pred_RF_xy, y_pred_RF_z.reshape(-1, 1)), axis=1)

        # evaluate_model(y_test[["x", "y"]], y_pred_RF_xy, "Random Forest")
        # print("--------------------------------------------------")
        # evaluate_model(y_test[["z"]], y_pred_RF_z, "Random Forest")

        evaluate_model(y_test, y_pred_RF, "Random Forest")

        print("--------------------------------------------------")

        DT_xy = DecisionTreeRegressor(
            criterion="poisson",
            max_depth=75,
            min_samples_leaf=2,
            ccp_alpha=0.0,
            min_impurity_decrease=0,
            max_leaf_nodes=280,
        )
        DT_z = DecisionTreeRegressor(
            criterion="squared_error",
            max_depth=95,
            min_samples_leaf=2,
            max_features=0.2,
            min_samples_split=6,
            min_impurity_decrease=0,
            max_leaf_nodes=180,
        )
        MOR = MultiOutputRegressor(DT_xy)

        y_pred_DT_xy = MOR.fit(X_train, xy_train).predict(X_test)
        y_pred_DT_z = DT_z.fit(X_train, floors_train.values.ravel()).predict(X_test)

        y_pred_DT = np.concatenate((y_pred_DT_xy, y_pred_DT_z.reshape(-1, 1)), axis=1)

        # evaluate_model(y_test[["x", "y"]], y_pred_DT_xy, "Decision Tree")
        # print("--------------------------------------------------")
        # evaluate_model(y_test[["z"]], y_pred_DT_z, "Decision Tree")

        evaluate_model(y_test, y_pred_DT, "Decision Tree")

        SVM_xy = SVR(degree=1, C=1, epsilon=0.0001, shrinking=False, verbose=True)
        SVM_z = SVR(degree=1, C=0.2, epsilon=0.0001, verbose=True)
        MOR = MultiOutputRegressor(SVM_xy)

        y_pred_SVM_xy = MOR.fit(X_train, xy_train).predict(X_test)
        y_pred_SVM_z = SVM_z.fit(X_train, floors_train.values.ravel()).predict(X_test)

        y_pred_SVM = np.concatenate(
            (y_pred_SVM_xy, y_pred_SVM_z.reshape(-1, 1)), axis=1
        )

        # evaluate_model(y_test[["x", "y"]], y_pred_SVM_xy, "Support Vector Machine")
        # print("--------------------------------------------------")
        # evaluate_model(y_test[["z"]], y_pred_SVM_z, "Support Vector Machine")

        evaluate_model(y_test, y_pred_SVM, "Support Vector Machine")

        print("--------------------------------------------------")

        KNN_xy = KNeighborsRegressor(
            n_neighbors=7, weights="distance", leaf_size=1, p=1
        )
        KNN_z = KNeighborsRegressor(
            algorithm="ball_tree", leaf_size=1, n_neighbors=7, p=1, weights="distance"
        )
        MOR = MultiOutputRegressor(KNN_xy)

        y_pred_KNN_xy = MOR.fit(X_train, xy_train).predict(X_test)
        y_pred_KNN_z = KNN_z.fit(X_train, floors_train.values.ravel()).predict(X_test)

        y_pred_KNN = np.concatenate(
            (y_pred_KNN_xy, y_pred_KNN_z.reshape(-1, 1)), axis=1
        )

        # evaluate_model(y_test[["x", "y"]], y_pred_KNN_xy, "K Nearest Neighbors")
        # print("--------------------------------------------------")
        # evaluate_model(y_test[["z"]], y_pred_KNN_z, "K Nearest Neighbors")

        evaluate_model(y_test, y_pred_KNN, "K Nearest Neighbors")

        plot_3d(y_test, y_pred_RF)
        plot_3d(y_test, y_pred_DT)
        plot_3d(y_test, y_pred_SVM)
        plot_3d(y_test, y_pred_KNN)

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
        print("Mean Squared Error: ", scores[1])
        print("Mean Absolute Error: ", scores[2])

        # make predictions
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print("r2 score: ", r2.round(5) * 100, "%")

        if USE_PLOT_FLAG:
            plot_3d(y_test, y_pred)


if __name__ == "__main__":
    main()
