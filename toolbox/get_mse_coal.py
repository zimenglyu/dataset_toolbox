import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import os

is_SOM = True
# file_path = "/Users/zimenglyu/Documents/code/git/susi/SOM_energy_50/energy"
num_train = 53
obs = pd.read_csv("/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/cyclone_10/Cyclone_10_202303_202105_202209_lab_results.csv", index_col=0)
# print("sample columns:")
# print(sample.columns)
# sample.drop(sample.columns[0], axis=1)
# print("sample columns:")
# print(sample.columns)
# Create a dataframe to store mse values

epoch=20000
# for i, file in enumerate(pred_files):

for n in [20]:
    file_path = f'/Users/zimenglyu/Documents/code/git/susi/SOM_Result_NEW/combined/{n}/'
    print("file path is ", file_path)
    for nei in [5]:
        print("--------------------------------------")
        print("doing for n: ", n, "nei: ", nei)
        for norm in ["minmax", "standard", "robust"]:
            print("doing for norm: ", norm)
            for method in ["weighted"]: 
                i=0
                mse_df = pd.DataFrame(columns=obs.columns)
                for fold in range(5):
                    filename = os.path.join(file_path, f'combined_{n}_{epoch}_{method}_neighbor_{nei}_{norm}_{fold}.csv')
                    # filename =  f'{file_path}_{fold}.csv'

                    # print("Doing for file: {}".format(filename))
                    # Load prediction file
                    pred = pd.read_csv(filename)
                    # drop the first column
                    pred = pred.drop(pred.columns[0], axis=1)
                    # Compute mse for each column and store in mse_df
                    for col in obs.columns:
                        if (is_SOM):
                            mse_df.loc[i, col] = mean_squared_error(obs[col], pred[col], squared=False)
                        else:
                            mse_df.loc[i, col] = mean_squared_error(obs[col][num_train:], pred[col][num_train:])
                    i += 1

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

# table_string