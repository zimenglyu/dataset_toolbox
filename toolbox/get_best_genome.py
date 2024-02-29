import pandas as pd
import os
from glob import glob
import shutil

result_path = '/Users/zimenglyu/Documents/cluster_results/SingleStock'
genome_path = "/Users/zimenglyu/Documents/cluster_results/0216"

best_mse = 100
if not os.path.exists(genome_path):
    os.makedirs(genome_path)
    print("Created directory: {}".format(genome_path))

for stock in ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DOW', 'DIS', 'WBA', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'TRV', 'UNH', 'RTX', 'VZ', 'V', 'WMT', 'XOM']:
    for i in range(15):
        filepath = "{}/{}/lr_0.0001/max_genome_20000/island_10/{}".format(result_path, stock, i)
        df = pd.read_csv("{}/fitness_log.csv".format(filepath))
        mse = df[' Best Val. MSE'].iloc[-1]
        if mse < best_mse:
            best_mse = mse
            best_genome_path = glob("{}/*.bin".format(filepath))[0]
        new_genome_path = os.path.join(genome_path, "{}.bin".format(stock))
        shutil.copy(best_genome_path, new_genome_path)

