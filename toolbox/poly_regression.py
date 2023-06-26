from sklearn.datasets import make_regression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from gaussian_process import do_pca
from sklearn.metrics import mean_squared_error
from gaussian_process import get_norm_method


if __name__ == '__main__':
    norm_name = "standard"
    norm_method = get_norm_method(norm_name)
    k_fold = 5

    X_path = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209_lab_results_spectra.csv"
    X_data = pd.read_csv(X_path)
    y_path = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209_lab_results.csv"
    y_data = pd.read_csv(y_path)
    datetime = y_data['DateTime']
    scaler_X = norm_method
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
        best_model = None
        best_degree = None
        best_score = -np.inf
        # X = df.drop(col, axis=1).values
        scaler_y = norm_method
        # print(y_data[col].to_numpy().reshape(-1, 1).shape
        y_scaled = scaler_y.fit_transform(y_data[[col]])
        y = y_scaled[:50, :]
        y_test = y_scaled[50:, :]
            
        #     # Define the model
        for degree in range(1, 5):
            poly_features = PolynomialFeatures(degree=degree)
            X_poly = poly_features.fit_transform(X)

            # Create a linear regression model
            model = LinearRegression()

            # Perform cross-validation on the training set
            scores = cross_val_score(model, X_poly, y, cv=k_fold)
            mean_score = np.mean(scores)

            # Check if this model has the best score so far
            if mean_score > best_score:
                best_score = mean_score
                best_degree = degree
                best_model = model

        # Step 4: Train the best model on the full training set
        poly_features = PolynomialFeatures(degree=best_degree)
        X_poly = poly_features.fit_transform(X)
        best_model.fit(X_poly, y)

        X_test_poly = poly_features.transform(X_test)
        y_pred = best_model.predict(X_test_poly)
        test_mse = mean_squared_error(y_test, y_pred)

        # Calculate the mean squared error of the predictions
        mse_all.append(test_mse)
        print("Best Degree:", best_degree)
        print("Test MSE:", test_mse)
        print("Actual Test Predictions:", y_pred)
    print("total MSE:", np.mean(mse_all))

    y_all = np.append(y,y_pred)

    df[[col]] = scaler_y.inverse_transform(y_all.reshape(-1, 1))
    # Save the predicted results to a CSV file
    filename = '202303_202105_202209_poly_regression_' + norm_name + '.csv'
    df.to_csv(filename, index=False)
