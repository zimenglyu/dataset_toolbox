from sklearn.datasets import make_regression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from gaussian_process import do_pca
from sklearn.metrics import mean_squared_error



if __name__ == '__main__':
    num_train = 53
    norm_method = StandardScaler()
    X_path = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209_lab_results_spectra.csv"
    X_data = pd.read_csv(X_path)
    y_path = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209_lab_results.csv"
    y_data = pd.read_csv(y_path)
    datetime = y_data['DateTime']
    scaler_X = norm_method
    X_scaled = scaler_X.fit_transform(X_data.iloc[:, 1:])
    X_pca = do_pca(X_scaled)
    # X= X_pca
    X = X_pca[:num_train, :]
    X_test = X_pca[num_train:, :]

    cols = y_data.columns[1:]
    # Iterate over each column and perform Gaussian Process Regression
    df = pd.DataFrame()
    df['DateTime'] = datetime

    mse_all = []
    for col in cols:
        # X = df.drop(col, axis=1).values
        scaler_y = norm_method
        # print(y_data[col].to_numpy().reshape(-1, 1).shape
        y_scaled = scaler_y.fit_transform(y_data[[col]])
        y = y_scaled[:num_train, :]
        y_test = y_scaled[num_train:, :]
            
        #     # Define the model
        model = LinearRegression()
        
        # Step 4: Perform cross-validation on the training set
        cv_scores = cross_val_score(model, X, y, cv=5)

        # Step 5: Fit the model on the full training set
        model.fit(X, y)

        # Step 6: Evaluate the model on the test set
        test_score = model.score(X_test, y_test)
        y_pred_test = model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred_test)

        # Calculate the mean squared error of the predictions
        mse_all.append(test_mse)
        print("Cross-Validation Scores:", cv_scores)
        print("Mean Cross-Validation Score:", np.mean(cv_scores))
        print("Test MSE:", test_mse)
        print("Actual Test Predictions:", y_pred_test)
    print("total MSE:", np.mean(mse_all))

    y_all = np.append(y,y_pred_test)

    df[[col]] = scaler_y.inverse_transform(y_all.reshape(-1, 1))
    # Save the predicted results to a CSV file
    df.to_csv('202303_202105_202209_Linear_regression_minmax.csv', index=False)
