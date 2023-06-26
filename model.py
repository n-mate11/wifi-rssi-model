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

from experiment import draw_predictions_on_map, preprocess_experiment_data

# flags
TRAIN_ML_FLAG = False
TRAIN_NN_FLAG = True

USE_DIRECTION_FLAG = False
USE_COORDS_FLAG = False

USE_PLOT_FLAG = True

EXPERIMENT = True

if TRAIN_NN_FLAG:
    import keras_tuner as kt
    from tuner import MyHyperModel

import matplotlib.pyplot as plt

# constants
F5_PATH = "./data/F5/"
F6_PATH = "./data/F6/"

RSSI_MIN = -100
RSSI_MAX = 0
NO_SIGNAL = 1

X_MAX = 70
Y_MAX = 70
Z_MAX = 12

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
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def scale_rssi(df):
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


def scale_xyz(df):
    for column in df.columns:
        if "x" == column or "_x" in column:
            df[column] = df[column].apply(lambda x: x / X_MAX)
        elif "y" == column or "_y" in column:
            df[column] = df[column].apply(lambda x: x / Y_MAX)
        elif "z" == column or "_z" in column:
            df[column] = df[column].apply(lambda x: x / Z_MAX)
    return df


def scale_directions(df):
    for column in df.columns:
        if "direction" in column:
            df[column] = df[column].apply(lambda x: x / 360)
    return df


def preprocess_data(df):
    df = scale_rssi(df)
    df = scale_xyz(df)
    if USE_DIRECTION_FLAG:
        df = scale_directions(df)
    return df


def add_directions(df):
    df["direction"] = 0
    for i in range(0, len(df), 60):
        df["direction"].iloc[i : i + 15] = 0
        df["direction"].iloc[i + 15 : i + 30] = 90
        df["direction"].iloc[i + 30 : i + 45] = 180
        df["direction"].iloc[i + 45 : i + 60] = 270
    return df


def train(regressor, X_train, X_test, y_train):
    MOR = MultiOutputRegressor(regressor)
    return MOR.fit(X_train, y_train).predict(X_test)


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


def plot_3d(y_test, y_pred, name):
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    SWAPPED = True
    if SWAPPED:
        color_map = plt.get_cmap("tab20c")
        colors = y_test["x"] + y_test["y"] + y_test["z"]
        ax.scatter3D(
            y_pred[:, 0],
            y_pred[:, 1],
            y_pred[:, 2],
            s=5,
            cmap=color_map,
            c=colors,
            label="y_pred",
        )
        ax.scatter3D(
            y_test["x"],
            y_test["y"],
            y_test["z"],
            s=20,
            cmap=color_map,
            c=colors,
            label="y_test",
        )
        ax.scatter3D(
            y_test["x"], y_test["y"], y_test["z"], s=3, color="black", label="y_test"
        )
    else:
        ax.scatter3D(
            y_test["x"], y_test["y"], y_test["z"], color="blue", label="y_test"
        )
        ax.scatter3D(
            y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], color="red", label="y_pred"
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Actual values vs Predicted values - {name}")
    plt.show()
    plt.savefig(f"{name}.png", bbox_inches="tight")


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

    # split the data
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=0)

    if EXPERIMENT:
        experiment_df = load_data("./data/experiment")
        experiment_df = preprocess_experiment_data(experiment_df)
        experiment_df = scale_rssi(experiment_df)
        experiment_df = scale_xyz(experiment_df)

        X_experiment_test = experiment_df.iloc[:, :-3]
        y_experiment_test = experiment_df.iloc[:, -3:]

    if TRAIN_ML_FLAG:
        if USE_DIRECTION_FLAG:
            print("Using directions")
            RF = RandomForestRegressor(
                n_estimators=170,
                max_features="sqrt",
                max_leaf_nodes=330,
                bootstrap=False,
                random_state=3,
            )
            y_pred_RF = train(RF, X_train, X_test, y_train)
            evaluate_model(y_test, y_pred_RF, "Random Forest")

            print("--------------------------------------------------")

            DT = DecisionTreeRegressor(
                criterion="poisson",
                min_samples_split=7,
                max_features="sqrt",
                max_leaf_nodes=50,
            )
            y_pred_DT = train(DT, X_train, X_test, y_train)
            evaluate_model(y_test, y_pred_DT, "Decision Tree")

            print("--------------------------------------------------")

            SVM = SVR(degree=1, gamma=1, coef0=0.001, C=1, epsilon=0.001)
            y_pred_SVM = train(SVM, X_train, X_test, y_train)
            evaluate_model(y_test, y_pred_SVM, "Support Vector Machine")

            print("--------------------------------------------------")

            KNN = KNeighborsRegressor(
                n_neighbors=7,
                weights="distance",
                algorithm="kd_tree",
                leaf_size=40,
                metric="manhattan",
            )
            y_pred_KNN = train(KNN, X_train, X_test, y_train)
            evaluate_model(y_test, y_pred_KNN, "K-Nearest Neighbors")
        elif USE_COORDS_FLAG:
            print("Using coordinates")
            RF = RandomForestRegressor(
                n_estimators=90, max_depth=55, max_features="sqrt", bootstrap=False
            )
            y_pred_RF = train(RF, X_train, X_test, y_train)
            evaluate_model(y_test, y_pred_RF, "Random Forest")

            print("--------------------------------------------------")

            DT = DecisionTreeRegressor(
                criterion="poisson",
                max_depth=85,
                max_features="sqrt",
                min_samples_split=7,
            )
            y_pred_DT = train(DT, X_train, X_test, y_train)
            evaluate_model(y_test, y_pred_DT, "Decision Tree")

            print("--------------------------------------------------")

            SVM = SVR(degree=10, kernel="poly", C=1, epsilon=0.01, shrinking=False)
            y_pred_SVM = train(SVM, X_train, X_test, y_train)
            evaluate_model(y_test, y_pred_SVM, "Support Vector Machine")

            print("--------------------------------------------------")

            KNN = KNeighborsRegressor(
                n_neighbors=7,
                weights="distance",
                algorithm="kd_tree",
                leaf_size=8,
                p=1,
                metric="manhattan",
            )
            y_pred_KNN = train(KNN, X_train, X_test, y_train)
            evaluate_model(y_test, y_pred_KNN, "K-Nearest Neighbors")
        else:
            print("Using basis model")
            RF = RandomForestRegressor(
                n_estimators=175,
                max_depth=60,
                min_samples_split=4,
                bootstrap=False,
                max_features="sqrt",
            )
            if EXPERIMENT:
                y_pred_RF = train(RF, X_train, X_experiment_test, y_train)
                evaluate_model(y_experiment_test, y_pred_RF, "Random Forest")
            else:
                y_pred_RF = train(RF, X_train, X_test, y_train)
                evaluate_model(y_test, y_pred_RF, "Random Forest")

            print("--------------------------------------------------")
            DT = DecisionTreeRegressor(
                criterion="poisson", min_samples_split=5, max_leaf_nodes=230
            )
            if EXPERIMENT:
                y_pred_DT = train(DT, X_train, X_experiment_test, y_train)
                evaluate_model(y_experiment_test, y_pred_DT, "Decision Tree")
            else:
                y_pred_DT = train(DT, X_train, X_test, y_train)
                evaluate_model(y_test, y_pred_DT, "Decision Tree")

            print("--------------------------------------------------")
            SVM = SVR(degree=1, coef0=1e-06, gamma=1, tol=1e-05, C=1)
            if EXPERIMENT:
                y_pred_SVM = train(SVM, X_train, X_experiment_test, y_train)
                evaluate_model(y_experiment_test, y_pred_SVM, "Support Vector Machine")
            else:
                y_pred_SVM = train(SVM, X_train, X_test, y_train)
                evaluate_model(y_test, y_pred_SVM, "Support Vector Machine")

            print("--------------------------------------------------")
            KNN = KNeighborsRegressor(
                algorithm="kd_tree",
                leaf_size=10,
                n_neighbors=7,
                weights="distance",
                metric="manhattan",
                p=1,
            )
            if EXPERIMENT:
                y_pred_KNN = train(KNN, X_train, X_experiment_test, y_train)
                evaluate_model(y_experiment_test, y_pred_KNN, "K-Nearest Neighbors")
            else:
                y_pred_KNN = train(KNN, X_train, X_test, y_train)
                evaluate_model(y_test, y_pred_KNN, "K-Nearest Neighbors")

        if USE_PLOT_FLAG:
            if EXPERIMENT:
                draw_predictions_on_map(y_experiment_test, y_pred_RF, "Random Forest")
                draw_predictions_on_map(y_experiment_test, y_pred_DT, "Decision Tree")
                draw_predictions_on_map(
                    y_experiment_test, y_pred_SVM, "Support Vector Machine"
                )
                draw_predictions_on_map(
                    y_experiment_test, y_pred_KNN, "K-Nearest Neighbors"
                )
            else:
                plot_3d(y_test, y_pred_RF, "Random Forest")
                plot_3d(y_test, y_pred_DT, "Decision Tree")
                plot_3d(y_test, y_pred_SVM, "Support Vector Machine")
                plot_3d(y_test, y_pred_KNN, "K-Nearest Neighbors")

    if TRAIN_NN_FLAG:
        # split data
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=0)

        if EXPERIMENT:
            VALIDATION_DATA = (X_experiment_test, y_experiment_test)
        else:
            VALIDATION_DATA = (X_test, y_test)

        PROJECT_NAME = (
            "with_directions"
            if USE_DIRECTION_FLAG
            else "with_coords"
            if USE_COORDS_FLAG
            else "experiment"
            if EXPERIMENT
            else "basis"
        )

        # keras tuner
        hyperModel = MyHyperModel()
        tuner = kt.BayesianOptimization(
            hyperModel,
            objective="mse",
            max_trials=5,
            executions_per_trial=3,
            directory="keras_hypermodels",
            project_name=PROJECT_NAME,
        )

        # search for best hyperparameters
        tuner.search(
            X_train,
            y_train,
            validation_split=VALIDATION_SPLIT,
            validation_data=VALIDATION_DATA,
        )

        best_model = tuner.get_best_models(num_models=1)[0]
        best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Build the model with the best hyperparameters
        model = tuner.hypermodel.build(best_hyperparameters)
        # Train the model.
        history = model.fit(
            X_train,
            y_train,
            validation_split=VALIDATION_SPLIT,
            validation_data=VALIDATION_DATA,
        )

        # re-train the model with the best hyperparameters
        model.fit(
            X_train,
            y_train,
            epochs=best_hyperparameters.values["epochs"],
            batch_size=best_hyperparameters.values["batch_size"],
            shuffle=best_hyperparameters.values["shuffle"],
            validation_split=VALIDATION_SPLIT,
            validation_data=VALIDATION_DATA,
        )

        # Evaluate the best model.
        loss, mse, mae = model.evaluate(X_test, y_test)
        print("----------------------------------------------")
        print("Mean Squared Error: ", mse)
        print("Mean Absolute Error: ", mae)

        # make predictions
        if EXPERIMENT:
            y_pred = model.predict(X_experiment_test)
            r2 = r2_score(y_experiment_test, y_pred)
        else:
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
        print("----------------------------------------------")
        print("r2 score: ", r2.round(5) * 100, "%")
        print("----------------------------------------------")

        print(model.summary())
        print("----------------------------------------------")
        print("Best Hyperparameters:")
        print("----------------------------------------------")
        print(best_hyperparameters.values)
        print("----------------------------------------------")

        if USE_PLOT_FLAG:
            if EXPERIMENT:
                draw_predictions_on_map(y_experiment_test, y_pred, "Neural Network")
            else:
                plot_3d(y_test, y_pred, "Neural Network")


if __name__ == "__main__":
    main()
