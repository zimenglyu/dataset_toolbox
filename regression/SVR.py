from sklearn.datasets import make_regression
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from gaussian_process import do_pca
from sklearn.preprocessing import MinMaxScaler

X_path = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209_lab_results_spectra.csv"
X_data = pd.read_csv(X_path)
y_path = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209_lab_results.csv"
y_data = pd.read_csv(y_path)
datetime = y_data['DateTime']
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_data.iloc[:, 1:])
X_pca = do_pca(X_scaled)
# X= X_pca
X = X_pca[:50, :]
X_test = X_pca[50:, :]

cols = y_data.columns[1:]
# Iterate over each column and perform Gaussian Process Regression
df = pd.DataFrame()
df['DateTime'] = datetime

mse_all = []
for col in cols:
    # X = df.drop(col, axis=1).values
    scaler_y = MinMaxScaler()
    # print(y_data[col].to_numpy().reshape(-1, 1).shape
    y_scaled = scaler_y.fit_transform(y_data[[col]])
    y = y_scaled[:50, :]
    y_test = y_scaled[50:, :]
    
    # Initialize the LeaveOneOut function
    loo = LeaveOneOut()

    # y_true, y_pred = list(), list()
    # for train_index, test_index in loo.split(X):
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]

        # Define the model
    model = SVR()

    # Define the grid of hyperparameters to search
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.1, 1, 10, 100],
        'epsilon':  [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
    }

    # Define the grid search
    grid = GridSearchCV(model, param_grid, cv=LeaveOneOut(), scoring='neg_mean_squared_error')

    # Fit the grid search
    grid.fit(X, y)

    # Show the best hyperparameters
    print('Best Hyperparameters: %s' % grid.best_params_)

    # Fit the model on the whole dataset using the best hyperparameters
    model = SVR(**grid.best_params_)
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate the mean squared error of the predictions
    mse = mean_squared_error(y_test, y_pred)

    mse_all.append(mse)
    print('MSE: %.3f' % mse)
print("total MSE:", np.mean(mse_all))


# For illustration, we generate a 2D dataset with 100 samples
# X, y = make_regression(n_samples=100, n_features=1, noise=0.1)


