
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from gaussian_process import
from sklearn.metrics import mean_squared_error

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

def get_norm_method(norm_name):
    if (norm_name == "minmax"):
        print("minmax")
        norm_method = MinMaxScaler()
    elif (norm_name == "standard"):
        print("standard")
        norm_method = StandardScaler()
    return norm_method

# def do_pca(X, portion=0.8):
#     if (portion == 1.0):
#         print("No PCA is performed")
#         return X
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

if __name__ == '__main__':

    
    X_path = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209_lab_results_spectra.csv"

    norm_name = "minmax"
    norm_method = get_norm_method(norm_name)

    X_data = pd.read_csv(X_path)

    scaler_X = norm_method
    X_scaled = scaler_X.fit_transform(X_data.iloc[:, 1:])
    print(X_scaled.shape)
    # Conduct PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    # Calculate cumulative variance
    var = pca.explained_variance_ratio_
    print(var.shape)
    cumulative_var = np.cumsum(np.round(var, decimals=3)*100)

    # Calculate the number of components required for 80% and 90% variance
    num_comp_80 = np.where(cumulative_var >= 80)[0][0] + 1
    num_comp_90 = np.where(cumulative_var >= 90)[0][0] + 1

    # Create a range of number of components
    num_comp = range(1, len(cumulative_var)+1)

    # Plot cumulative variance
    plt.figure(figsize=(5, 5))
    plt.plot(num_comp, cumulative_var)
    plt.scatter(num_comp_80, 80, color='r')
    plt.scatter(num_comp_90, 90, color='b')

    # Add dotted line at the points where 80% and 90% of variance is achieved
    # plt.axhline(y=80, color='r', linestyle='dotted')
    # plt.axvline(x=num_comp_80, color='r', linestyle='dotted')

    # plt.axhline(y=90, color='b', linestyle='dotted')
    # plt.axvline(x=num_comp_90, color='b', linestyle='dotted')

    plt.title('Cumulative Explained Variance in PCA')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance (%)')

    # Mark the points where 80% and 90% of variance is achieved
    plt.xticks([num_comp_80, num_comp_90])

    # Display the plot
    # plt.show()
    plt.savefig("PCA_" + norm_name + ".png")