import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


num_train = 53
is_SOM = False
# Load true observations
obs = pd.read_csv('/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209_lab_results.csv')
# drop the first column
obs = obs.drop(obs.columns[0], axis=1)
# File name prefix and suffix
file_prefix = '/Users/zimenglyu/Documents/code/git/dataset_toolbox/toolbox/results/dnn/combined_2021-2023_dnn_minmax_'
file_suffix = '.csv'
print("Doing for file: {}".format(file_prefix.split('/')[-1]))
# Generate file names
pred_files = [file_prefix + str(i) + file_suffix for i in range(5)]

# Create a dataframe to store mse values
mse_df = pd.DataFrame(columns=obs.columns)

for i, file in enumerate(pred_files):
    # Load prediction file
    pred = pd.read_csv(file)
    # drop the first column
    pred = pred.drop(pred.columns[0], axis=1)
    # Compute mse for each column and store in mse_df
    for col in obs.columns:
        if (is_SOM):
            mse_df.loc[i, col] = mean_squared_error(obs[col], pred[col], squared=False)
        else:
            mse_df.loc[i, col] = mean_squared_error(obs[col][num_train:], pred[col][num_train:])

# Print average mse for each column
avg_mse = mse_df.mean()
table_string = ''
for col, mse in avg_mse.items():
    if (abs(mse) < 100):
        print(f'Column: {col:<15}, Average RMSE: {mse:.2f}')
        table_string += f'{mse:.2f} & '
    else:
        print(f'Column: {col:<15}, Average RMSE: {mse:.1e}')
        table_string += f'{mse:.1e} & '
table_string += f'{avg_mse.mean():.1e}'
print(table_string)
# table_string