import numpy as np
import pandas as pd
import optuna
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import LeaveOneOut, cross_val_predict, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import mean_squared_error
X = []
y = []

# def get_norm_method(norm_name):
#     if (norm_name == "minmax"):
#         norm_method = MinMaxScaler()
#     elif (norm_name == "standard"):
#         norm_method = StandardScaler()
#     return norm_method

def objective_RBF(trial):
    alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
    constant = trial.suggest_float('constant', 1e-5, 1e1, log=True)
    length_scale = trial.suggest_float('length_scale', 1e-1, 10.0, log=True)
    kernel = C(constant) * RBF(length_scale=length_scale)
    model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=5)

    # Perform cross-validation on the training set
    cv_scores = cross_val_score(model, X, y, cv=k_fold)
    return -np.mean(cv_scores)  # Minimize the negative mean cross-validation score

def objective_matern(trial):
    alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
    constant = trial.suggest_float('constant', 1e-5, 1e1, log=True)
    length_scale = trial.suggest_float('length_scale', 1e-1, 10.0, log=True)
    nu = trial.suggest_float('nu', 0.1, 2.5)
    kernel = C(constant) * Matern(length_scale=length_scale, nu=nu)
    model = GaussianProcessRegressor(kernel=kernel, alpha=alpha)

    # Perform cross-validation on the training set
    cv_scores = cross_val_score(model, X, y, cv=k_fold)
    return -np.mean(cv_scores)  # Minimize the negative mean cross-validation score


# def do_pca(X, portion=0.8):
#     # Run initial PCA to determine number of components that explain at least 80% variance
#     pca = PCA()
#     pca.fit(X)

#     explained_variance_ratio = pca.explained_variance_ratio_
#     cumulative_explained_variance = np.cumsum(explained_variance_ratio)
#     num_components = np.where(cumulative_explained_variance >= portion)[0][0] + 1
#     print(f"Number of components that explain at least 80% of the variance: {num_components}")

#     # Now, run PCA again with the desired number of components
#     pca = PCA(n_components=num_components)
#     data_pca = pca.fit_transform(X)
#     print("Variance percentage of each principal component:", pca.explained_variance_ratio_)
#     # print("Variance sum of each principal component:", np.sum(pca.explained_variance_ratio_))
#     return data_pca

def get_best_model_RBF(study):
    best_alpha = study.best_params['alpha']
    best_constant = study.best_params['constant']
    best_length_scale = study.best_params['length_scale']
    best_kernel = C(best_constant) * RBF(length_scale=best_length_scale)
    best_model = GaussianProcessRegressor(kernel=best_kernel, alpha=best_alpha)
    return best_model

def get_best_model_matern(study):
    best_alpha = study.best_params['alpha']
    best_constant = study.best_params['constant']
    best_length_scale = study.best_params['length_scale']
    best_nu = study.best_params['nu']
    best_kernel = C(best_constant) * Matern(length_scale=best_length_scale, nu=best_nu)
    best_model = GaussianProcessRegressor(kernel=best_kernel, alpha=best_alpha)
    return best_model

if __name__ == '__main__':
    num_train = 53
    norm_name = "standard"
    norm_method = get_norm_method(norm_name)
    kernal_function = "Matern"
    num_trails = 100
    k_fold = 5

    X_path = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209_lab_results_spectra.csv"
    y_path = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209_lab_results.csv"

    X_data = pd.read_csv(X_path)
    y_data = pd.read_csv(y_path)
    datetime = y_data['DateTime']
    scaler_X = norm_method
    X_scaled = scaler_X.fit_transform(X_data.iloc[:, 1:])
    X_pca = do_pca(X_scaled)
    X= X_pca
    X = X_pca[:num_train, :]
    X_test = X_pca[num_train:, :]

    cols = y_data.columns[1:]
    # Iterate over each column and perform Gaussian Process Regression
    df = pd.DataFrame()
    df['DateTime'] = datetime
    mse_all = []
    for col in cols:
        scaler_y = norm_method
        # print(y_data[col].to_numpy().reshape(-1, 1).shape
        y_scaled = scaler_y.fit_transform(y_data[[col]])
        y = y_scaled[:num_train, :]
        y_test = y_scaled[num_train:, :]
        
        study = optuna.create_study(direction='minimize')
        best_model = None
        if (kernal_function == "RBF"):
            study.optimize(objective_RBF, n_trials=num_trails)
            best_model = get_best_model_RBF(study)
        elif (kernal_function == "Matern"):
            study.optimize(objective_matern, n_trials=num_trails)
            best_model = get_best_model_matern(study)
        # print("Best trial: ", study.best_trial.params)
        
        best_model.fit(X, y)

        y_pred_test = best_model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred_test)

        # If you need to evaluate the performance of your model, you can use an appropriate metric
        # For example, using the mean squared error:
        print("Best Hyperparameters:", study.best_params)
        print("Test MSE:", test_mse)
        print("Actual Test Predictions:", y_pred_test)

        df[[col]] = scaler_y.inverse_transform(np.vstack((y, y_pred_test.reshape(-1, 1))).reshape(-1, 1))
        # df[[col]] = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        mse_all.append(test_mse)
        # print('MSE: %.3f' % mse)
    print("total MSE for all columns:", np.mean(mse_all))

    # Save the predicted results to a CSV file
    filename = "202303_202105_202209_GPR_" + kernal_function + "_" + norm_name + ".csv"
    df.to_csv(filename, index=False)
