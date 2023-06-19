from model import load_data, scale_rssi, scale_xyz, split_data, train, plot_3d
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics

import kerastuner as kt
from tuner import MyHyperModel

VALIDATION_SPLIT = 0.2

START_COORDINATES = (10, 9, 6)
END_COORDINATES = (41, 9, 6)

# Distance in meters
DISTANCE = END_COORDINATES[0] - START_COORDINATES[0]


def preprocess_data(experiment_df):
    experiment_df["startTimestamp"] = pd.to_datetime(
        experiment_df["startTimestamp"], unit="ms"
    )
    experiment_df["stopTimestamp"] = pd.to_datetime(
        experiment_df["stopTimestamp"], unit="ms"
    )

    experiment_df["scanTime"] = pd.to_datetime(experiment_df["scanTime"], unit="ms")

    # Calculate average speed in m/s
    experiment_df["averageSpeed"] = (
        DISTANCE
        / (
            experiment_df["stopTimestamp"] - experiment_df["startTimestamp"]
        ).dt.total_seconds()
    )

    # Calculate distance to scan in meters and round to nearest integer
    experiment_df["x"] = (
        (
            (
                (
                    experiment_df["scanTime"] - experiment_df["startTimestamp"]
                ).dt.total_seconds()
            )
            * experiment_df["averageSpeed"]
        )
        .round()
        .astype(int)
    )

    experiment_df["y"] = 9
    experiment_df["z"] = 6

    experiment_df.drop(
        experiment_df.filter(regex="_timestamp").columns, axis=1, inplace=True
    )
    experiment_df.drop("averageSpeed", axis=1, inplace=True)
    experiment_df.drop("startTimestamp", axis=1, inplace=True)
    experiment_df.drop("stopTimestamp", axis=1, inplace=True)
    experiment_df.drop("scanTime", axis=1, inplace=True)

    return experiment_df


def main():
    experiment_df = load_data("./data/experiment")

    experiment_df = preprocess_data(experiment_df)

    experiment_df = scale_rssi(experiment_df)
    experiment_df = scale_xyz(experiment_df)

    X_train, X_test, y_train, y_test = split_data(experiment_df)

    RF = RandomForestRegressor()
    y_pred = train(experiment_df, RF, X_train, X_test, y_train)

    print("--- Random Forest ---")
    print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
    print("R2:", metrics.r2_score(y_test, y_pred))

    DF = DecisionTreeRegressor()
    y_pred = train(experiment_df, DF, X_train, X_test, y_train)

    print("--- Decision Tree ---")
    print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
    print("R2:", metrics.r2_score(y_test, y_pred))

    SVM = SVR()
    y_pred = train(experiment_df, SVM, X_train, X_test, y_train)

    print("--- Support Vector Machine ---")
    print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
    print("R2:", metrics.r2_score(y_test, y_pred))

    KNN = KNeighborsRegressor()
    y_pred = train(experiment_df, KNN, X_train, X_test, y_train)

    print("--- K-Nearest Neighbors ---")
    print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
    print("R2:", metrics.r2_score(y_test, y_pred))

    plot_3d(experiment_df, y_pred, "Random Forest")
    plot_3d(experiment_df, y_pred, "Decision Tree")
    plot_3d(experiment_df, y_pred, "Support Vector Machine")
    plot_3d(experiment_df, y_pred, "K-Nearest Neighbors")

    # create deep learning model and train it with tuner.py
    hyperModel = MyHyperModel()
    tuner = kt.BayesianOptimization(
        hyperModel,
        objective="mse",
        max_trials=5,
        executions_per_trial=3,
        directory="keras_hypermodels",
        project_name="experiment",
    )

    # search for best hyperparameters
    tuner.search(
        X_train,
        y_train,
        validation_split=VALIDATION_SPLIT,
        validation_data=(X_test, y_test),
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
        validation_data=(X_test, y_test),
    )

    # re-train the model with the best hyperparameters
    model.fit(
        X_train,
        y_train,
        epochs=best_hyperparameters.values["epochs"],
        batch_size=best_hyperparameters.values["batch_size"],
        shuffle=best_hyperparameters.values["shuffle"],
        validation_split=VALIDATION_SPLIT,
        validation_data=(X_test, y_test),
    )

    # Evaluate the best model.
    loss, mse, mae = model.evaluate(X_test, y_test)
    print("----------------------------------------------")
    print("Mean Squared Error: ", mse)
    print("Mean Absolute Error: ", mae)

    # make predictions
    y_pred = model.predict(X_test)
    r2 = metrics.r2_score(y_test, y_pred)
    print("----------------------------------------------")
    print("r2 score: ", r2.round(5) * 100, "%")
    print("----------------------------------------------")

    print(model.summary())
    print("----------------------------------------------")
    print("Best Hyperparameters:")
    print("----------------------------------------------")
    print(best_hyperparameters.values)
    print("----------------------------------------------")

    plot_3d(experiment_df, y_pred, "Deep Learning")


if __name__ == "__main__":
    main()
