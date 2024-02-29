import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


num_example = 10
result_path = "/Users/zimenglyu/Documents/code/git/susi/SOM_Flight_{}/flight/".format(num_example)

label_path = "/Users/zimenglyu/Documents/datasets/SOM_flight/flight_53438_labelled_{}.csv".format(num_example)
fold = 5

vote_array = []
wavg_array = []
# label_array = []
df_label = pd.read_csv(label_path)
for som_size in [5,7,10, 15]:
    for num_neighbor in [3,5,7,10]:
        for norm in ["minmax", "standard", "robust"]:
            for i in range(fold):
                vote_name = result_path + "{}/flight_{}_20000_vote_neighbor_{}_{}_{}.csv".format(som_size, som_size, num_neighbor, norm, i)
                wavg_name = result_path + "{}/flight_{}_20000_weighted_neighbor_{}_{}_{}.csv".format(som_size, som_size, num_neighbor, norm, i)
                
                df_vote = pd.read_csv(vote_name)
                df_wavg = pd.read_csv(wavg_name)

                # Extract second column of df_vote and convert to numpy array
                vote_array.append(np.transpose(df_vote.iloc[:, 1].to_numpy()))
                wavg_array.append(np.transpose(df_wavg.iloc[:, 1].to_numpy()))
            plt.figure(figsize=(10, 5))
            plt.plot(np.mean(vote_array, axis=0), label="Vote")
            plt.plot(np.mean(wavg_array, axis=0), label="WAVG")
            plt.plot(df_label.iloc[:, 1].to_numpy(), label="Label")
            plt.legend()
            plt.title("SOM size: {}x{}, neighbor: {}, {}".format(som_size, som_size, num_neighbor, norm))
            plt.savefig(result_path + "flight_{}_nei_{}_{}.png".format(som_size, num_neighbor, norm))

    