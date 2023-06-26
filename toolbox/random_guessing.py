import pandas as pd
import numpy as np

# Load true observations
obs = pd.read_csv('/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209_lab_results.csv')

# File name prefix and suffix
file_prefix = '/Users/zimenglyu/Documents/code/git/dataset_toolbox/toolbox/results/random/combined_2021-2023_random_'
file_suffix = '.csv'

for method in ["uniform", "normal"]:
    
    # Generate and save 10 random prediction files
    for i in range(10):
        pred = pd.DataFrame()
        # copy the first column
        pred[obs.columns[0]] = obs[obs.columns[0]]
        if method == "uniform":
        # For each column, generate random values within the column's range
            for col in obs.columns[1:]:
                min_val = obs[col].min()
                max_val = obs[col].max()
                pred[col] = np.random.uniform(min_val, max_val, size=len(obs))
            
            # Save the predictions to a csv file
        elif method == "normal":
            for col in obs.columns[1:]:
                mean_val = obs[col].mean()
                std_val = obs[col].std()
                pred[col] = np.random.normal(mean_val, std_val, size=len(obs))
        pred.to_csv(file_prefix + method + "_" + str(i) + file_suffix, index=False)
