from sklearn.datasets import make_regression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
# from gaussian_process import
from sklearn.metrics import mean_squared_error
import argparse
from sklearn.decomposition import PCA
from Regression import Regression
import os

def get_norm_method(norm_name):
    if (norm_name == "minmax"):
        norm_method = MinMaxScaler()
    elif (norm_name == "standard"):
        norm_method = StandardScaler()
    elif (norm_name == "robust"):
        norm_method = RobustScaler()
    return norm_method

def do_pca(X, portion=0.8):
    if (portion == 1.0):
        print("No PCA is performed")
        return X
    # Run initial PCA to determine number of components that explain at least 80% variance
    pca = PCA()
    pca.fit(X)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    num_components = np.where(cumulative_explained_variance >= portion)[0][0] + 1
    print(f"Number of components that explain at least 80% of the variance: {num_components}")

    # Now, run PCA again with the desired number of components
    pca = PCA(n_components=num_components)
    data_pca = pca.fit_transform(X)
    print("Variance percentage of each principal component:", pca.explained_variance_ratio_)
    # print("Variance sum of each principal component:", np.sum(pca.explained_variance_ratio_))
    return data_pca

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments for regression tasks")

    parser.add_argument(
        "--input_X_path",
        required=True,
        help="input training X data path",
    )

    parser.add_argument(
        "--input_y_path",
        required=True,
        help="input training y data path",
    )

    parser.add_argument(
        "--norm_method",
        required=True,
        help="normalization method",
    ) 

    parser.add_argument(
        "--pca_level",
        required=False,
        default=0.8,
        type=float,
        help="pca percentage",
    )
    
    parser.add_argument(
        "--num_train_datapoints",
        required=False,
        default=53,
        type=int,
        help="number of training datapoints",
    )

    parser.add_argument(
        "--dataset_name",
        required=True,
        help="dataset name",
    )

    parser.add_argument(
        "--num_k_fold",
        required=False,
        default=5,
        type=int,
        help="number of k fold",
    )

    parser.add_argument(
        "--regression_method",
        required=True,
        help="regression method",
    )

    parser.add_argument(
        "--fold",
        required=True,
        help="repeat",
    )

    parser.add_argument(
        "--kernal_function",
        required=False,
        default="RBF",
        help="kernal function",
    )

    parser.add_argument(
        "--num_trails",
        required=False,
        default=100,
        type=int,
        help="number of trails",
    )

    parser.add_argument(
        "--output_path",
        required=True,
        help="output path",
    )

    args = parser.parse_args()
    
    X_path = args.input_X_path
    y_path = args.input_y_path
    norm_name = args.norm_method
    pca_level = args.pca_level
    num_train = args.num_train_datapoints
    dataset_name = args.dataset_name
    k_fold = args.num_k_fold
    fold = args.fold
    regression_method = args.regression_method
    kernal_function = args.kernal_function
    output_path = args.output_path
    print()
    print("===================================================")
    print("Finished parsing arguments")
    print("Doing regression with method:", regression_method)
    if (regression_method == "gaussian"):
        print("kernal function:", kernal_function)
    print("normalization method:", norm_name)
    print("pca level:", pca_level)
    print("number of training datapoints:", num_train)
    print("dataset name:", dataset_name)
    print("number of k fold:", k_fold)
    print("doing repeat:", fold)
    print("output path:", output_path)
    

    norm_method = get_norm_method(norm_name)

    X_data = pd.read_csv(X_path)
    y_data = pd.read_csv(y_path)
    datetime = y_data['DateTime']
    scaler_X = norm_method
    X_scaled = scaler_X.fit_transform(X_data.iloc[:, 1:])
    X_pca = do_pca(X_scaled, pca_level)
    X = X_pca[:num_train, :]
    X_test = X_pca[num_train:, :]

    pred_result = pd.DataFrame()
    pred_result['DateTime'] = datetime

    regressor = Regression(regression_method, args)
    
    cols = y_data.columns[1:]

    for col in cols:
        print("doing for column:", col)
        scaler_y = norm_method
        y_scaled = scaler_y.fit_transform(y_data[[col]])
        y = y_scaled[:num_train, :]
        y_test = y_scaled[num_train:, :]
        y_pred_test, test_score = regressor.do_regression(X, y, X_test, y_test)
        y_all = np.append(y,y_pred_test)
        pred_result[[col]] = scaler_y.inverse_transform(y_all.reshape(-1, 1))
    # Save the predicted results to a CSV file
    if (regression_method == "gaussian"):
        filename =  dataset_name + "_" + kernal_function + "_" + norm_name + "_" + fold + ".csv"
    else:
        filename =  dataset_name + "_" + regression_method + "_" + norm_name + "_" + fold + ".csv"

    pred_result.to_csv(os.path.join(output_path, filename))
