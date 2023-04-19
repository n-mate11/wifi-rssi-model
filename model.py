import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import pickle
import matplotlib.pyplot as plt

training_data = '../CSVfiles/trainingData.csv'
floor6v2 = '../CSVfiles/FE6/floor6v2.csv'
floor6v3 = '../CSVfiles/FE6/floor6v3.csv'
floor6v4 = '../CSVfiles/FE6/floor6v4.csv'

def load_data(train_path):
    train_dataframe = pd.concat(map(pd.read_csv, [floor6v2, floor6v3, floor6v4]), ignore_index=True)
    # print(train_dataframe)
    #train_dataframe = pd.read_csv(train_path, sep=";")

    X = train_dataframe.iloc[:, 0:323].values
    y = train_dataframe.iloc[:, [324, 325, 326]].values
    
    return train_test_split(X, y, test_size=0.25, shuffle=True)


def main():
    # load the data
    X_train, X_test, y_train, y_test = load_data(training_data)
    
    # models
    LR = LinearRegression()
    RFR = RandomForestRegressor()
    DTR = DecisionTreeRegressor()
    
    # train the model
    linear = LR.fit(X_train, y_train)
    random = RFR.fit(X_train, y_train)
    tree = DTR.fit(X_train, y_train)

    # predictions
    y_pred_linear = linear.predict(X_test)
    y_pred_random = random.predict(X_test)
    y_pred_tree = tree.predict(X_test)

    # evaluation
    print("Linear regression")
    print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred_linear))
    print('R-squared: %.2f'
      % r2_score(y_test, y_pred_linear))
    
    print("Random Forest regression")
    print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred_random))
    print('R-squared: %.2f'
      % r2_score(y_test, y_pred_random))
    
    print("Decision tree regression")
    print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred_tree))
    print('R-squared: %.2f'
      % r2_score(y_test, y_pred_tree))

    # x_ax = range(len(y_test))
    # plt.plot(x_ax, y_test, linewidth=1, label="original")
    # plt.plot(x_ax, y_pred, linewidth=1.1, label="predicted")
    # plt.title("y-test and y-predicted data")
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.legend(loc='best',fancybox=True, shadow=True)
    # plt.grid(True)
    # plt.show() 

    # # save the model in pickle format
    # pickle.dump(regressor, open('model.pkl','wb'))

if __name__ == "__main__":
    main()