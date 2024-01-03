import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import os


is_SOM = False
file_size = 53
num_train = int(file_size * 0.8)
print("num_train: ", num_train)
file_path = "/Users/zimenglyu/Documents/code/git/dataset_toolbox/regression/results_NEW/0.8"
# file_path = "/Users/zimenglyu/Documents/code/git/susi/SOM_energy_50/energy"
# file_path = f'/Users/zimenglyu/Documents/code/git/dataset_toolbox/regression/results_energy_{file_size}/pca_1'
sample = pd.read_csv("/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/cyclone_10/Cyclone_10_202303_202105_202209_lab_results.csv", index_col=0)
print("sample columns:")
print(sample.columns)
method_index = 0
gpr_methods = [ "RBF", "Matern"]
# for n in [30]:
#     for nei in [7]:
#         print("--------------------------------------")
# for regression_method in ["linear", "poly", "dnn", "gaussian", "gaussian"]:
for regression_method in ["k-nei"]:
    if (regression_method != "gaussian"):
        method = regression_method
    else:
        method = gpr_methods[method_index]
        method_index += 1
    print("--------------------------------------")
    print("doing for method: ", method)  
    for norm in ["minmax", "standard", "robust"]:
        print("doing for norm: ", norm)
        i = 0
        mse_df = pd.DataFrame(columns=sample.columns)
        # for file in range(10):
        # label_path = f'/Users/zimenglyu/Documents/datasets/regression/{file_size}/energydata_{file}_label.csv'
        # obs = pd.read_csv(sample, index_col=0)
        # obs = obs.drop(obs.columns[0], axis=1)
        for fold in range(10):
            filename = os.path.join(file_path, f'{regression_method}/combined_{method}_{norm}_{fold}.csv')
            # filename =  f'{file_path}_{fold}.csv'

            # print("Doing for file: {}".format(filename.split('/')[-1]))
            # Load prediction file
            pred = pd.read_csv(filename)
            # drop the first column
            pred = pred.drop(pred.columns[0], axis=1)
            # Compute mse for each column and store in mse_df
            for col in sample.columns:
                if (is_SOM):
                    mse_df.loc[i,col] = mean_squared_error(sample[col], pred[col], squared=False)
                else:
                    mse_df.loc[i,col] = mean_squared_error(sample[col][num_train:], pred[col][num_train:])
            i+=1

# Print average mse for each column
        avg_mse = mse_df.mean()

        table_string = ''
        for col, mse in avg_mse.items():
            if (abs(mse) < 100):
                # print(f'Column: {col:<15}, Average RMSE: {mse:.2f}')
                table_string += f'{mse:.2f} & '
            else:
                # print(f'Column: {col:<15}, Average RMSE: {mse:.1e}')
                table_string += f'{mse:.2e} & '
        table_string += f'{avg_mse.mean():.2e}'
        print(table_string)