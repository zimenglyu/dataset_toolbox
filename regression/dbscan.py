import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from regression_main import do_pca
# from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
from collections import Counter
from matplotlib import colors


X_path = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209.csv"
X_data = pd.read_csv(X_path)
min_pt = 2
# n_pc = 20
min_eps = 10  # example value
max_eps = 35  # example value
pca_scale = 1
eps_values = np.linspace(min_eps, max_eps, 5)

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_data.iloc[:, 1:])
data_pca = do_pca(X_scaled, pca_scale)

# list to store results
results = []

for eps in eps_values:
    # Create a DBSCAN model
    db = DBSCAN(eps=eps, min_samples=5)
    # Fit and predict
    y_db = db.fit_predict(data_pca)
    
    # Count clusters (ignoring noise if present)
    n_clusters = len(set(y_db)) - (1 if -1 in y_db else 0)
    
    # Find size of largest cluster, smallest cluster, average size and number of outliers
    sizes = [list(y_db).count(c) for c in set(y_db) if c != -1]
    n_outliers = list(y_db).count(-1)
    max_cluster_size = max(sizes) if sizes else 0
    min_cluster_size = min(sizes) if sizes else 0
    avg_cluster_size = sum(sizes)/len(sizes) if sizes else 0
    
    # Append results
    results.append({'eps': eps, 'n_clusters': n_clusters, 'n_outliers': n_outliers,
                    'max_cluster_size': max_cluster_size, 'min_cluster_size': min_cluster_size,
                    'avg_cluster_size': avg_cluster_size})

# Convert results to DataFrame
results_df = pd.DataFrame(results)

fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Plot eps vs n_clusters
axs[0].plot(results_df['eps'], results_df['n_clusters'])
axs[0].axvline(x=results_df['eps'][results_df['n_clusters'].idxmax()], color="0.7", linestyle='--')
axs[0].axvline(x=19, color="0.7", linestyle='--')
axs[0].set_title('eps vs n_clusters')
axs[0].set_xlabel('eps')
axs[0].set_ylabel('n_clusters')

# Plot eps vs various metrics
axs[1].plot(results_df['eps'], results_df['n_outliers'], label='Number of outliers')
axs[1].plot(results_df['eps'], results_df['max_cluster_size'], label='Max cluster size')
axs[1].plot(results_df['eps'], results_df['min_cluster_size'], label='Min cluster size')
axs[1].plot(results_df['eps'], results_df['avg_cluster_size'], label='Average cluster size')
axs[1].axvline(x=results_df['eps'][results_df['n_clusters'].idxmax()], color="0.7", linestyle='--')
axs[1].axvline(x=19, color="0.7", linestyle='--')
axs[1].set_title('eps vs metrics')
axs[1].set_xlabel('eps')
axs[1].set_ylabel('metrics')
axs[1].legend()

# Adjust layout for better appearance
plt.tight_layout()

# Save the figure
plt.savefig('DBSCAN_metrics_pca.png', dpi=300)

# Display the figure
plt.show()