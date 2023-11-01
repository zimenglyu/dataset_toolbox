import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import os

is_SOM = True
file_size = 50 
# file_path = "/Users/zimenglyu/Documents/code/git/susi/SOM_energy_50/energy"

sample = pd.read_csv("/Users/zimenglyu/Documents/datasets/regression/50/energydata_0_label.csv", index_col=0)
print("sample columns:")
print(sample.columns)
# sample.drop(sample.columns[0], axis=1)
# print("sample columns:")
# print(sample.columns)
# Create a dataframe to store mse values

epoch=20000
# for i, file in enumerate(pred_files):

for n in [10]:
    file_path = f'/Users/zimenglyu/Documents/code/git/susi/SOM_energy_FINAL/filesize_{file_size}/som_{n}'
    for nei in [5]:
        print("--------------------------------------")
        print("doing for n: ", n, "nei: ", nei)
        for norm in ["minmax", "standard", "robust"]:
            print("doing for norm: ", norm)
            for method in ["weighted"]: 
                i=0
                mse_df = pd.DataFrame(columns=sample.columns)
                for file in range(10):
                    label_path = f'/Users/zimenglyu/Documents/datasets/regression/{file_size}/energydata_{file}_label.csv'
                    obs = pd.read_csv(label_path, index_col=0)
                    # obs = obs.drop(obs.columns[0], axis=1)
                    for fold in range(1):
                        filename = os.path.join(file_path, f'{file}/energy_{n}_{epoch}_{method}_neighbor_{nei}_{norm}_{fold}.csv')
                        # filename =  f'{file_path}_{fold}.csv'

                        # print("Doing for file: {}".format(filename.split('/')[-1]))
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