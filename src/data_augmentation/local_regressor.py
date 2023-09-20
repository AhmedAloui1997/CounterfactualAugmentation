import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,Matern,DotProduct,RationalQuadratic,ExpSineSquared
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
import torch

def impute_missing_values_embeddings(embeddings, data_factual, k=5, distance_threshold=1.0, local_regressor="gp", gp_kernel="RBF"):
    # put none for the non observed potential outcomes
    data = np.column_stack((data_factual[:,:-1], np.nan * np.ones(data_factual.shape[0]), np.nan * np.ones(data_factual.shape[0])))
    for i in range(data.shape[0]):
        if data[i,-3] == 1:
            data[i,-1] = data_factual[i,-1]
        else:
            data[i,-2] = data_factual[i,-1]
    imputed_data = data.copy()
    X = data[:, :-3]
    X_embeddings = embeddings.detach().numpy()

    # Find nearest neighbors based on X
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="minkowski", p=2).fit(X_embeddings)

    def impute_column(column_idx, mask):
        missing_indices = np.argwhere(mask)[:, 0]
        observed_indices = np.argwhere(~mask)[:, 0]
        distances, indices = nbrs.kneighbors(X_embeddings[missing_indices])

        if local_regressor == "gp":
            if gp_kernel == "RBF":
                kernel = RBF(length_scale=0.01)
            elif gp_kernel == "Matern":
                kernel = Matern(nu=0.5)
            elif gp_kernel == "DotProduct":
                kernel = DotProduct(sigma_0=0.01)
            elif gp_kernel == "RationalQuadratic":
                kernel = RationalQuadratic(length_scale=0.01, alpha=0.1)
            elif gp_kernel == "ExpSineSquared":
                kernel = ExpSineSquared(length_scale=0.01, periodicity=0.1)
            else:
                raise ValueError("Unsupported kernel type")
            
            regressor = GaussianProcessRegressor(kernel=kernel, alpha=1e-1)
        elif local_regressor == "linear":
            regressor = LinearRegression()
        else:
            raise ValueError("Unsupported local regressor type")

        for missing_idx, neighbor_distances, neighbor_indices in zip(missing_indices, distances, indices):
            observed_neighbor_indices = [i for i in neighbor_indices if i in observed_indices]
            if np.all(neighbor_distances < distance_threshold) and len(observed_neighbor_indices) > 0:
                regressor.fit(X[observed_neighbor_indices], data[observed_neighbor_indices, column_idx])
                imputed_data[missing_idx, column_idx] = regressor.predict([X[missing_idx]])[0]

    # Impute missing values
    Y0_mask = np.isnan(data[:, -1])
    if np.any(Y0_mask):
        impute_column(-1, Y0_mask)

    Y1_mask = np.isnan(data[:, -2])
    if np.any(Y1_mask):
        impute_column(-2, Y1_mask)

    return imputed_data


#this function will output the imputed data into a data with only the new imputed factual outcomes.
def data_preprocessing(dataset,imputed_data):  
  rows = dataset
  for row in imputed_data:
      x = row[:-3]
      t, y1, y0 = row[-3:]

      if not np.isnan(y1) and not np.isnan(y0):
        if t==1:
          rows = np.vstack([rows,np.hstack((x, [0], [y0]))])
        if t==0:
          rows = np.vstack([rows,(np.hstack((x, [1], [y1])))])
  imputed_dataset = rows
  # Convert NumPy array to PyTorch tensor
  imputed_dataset_tensor = torch.from_numpy(imputed_dataset)
  imputed_dataset = imputed_dataset_tensor.float()
  return imputed_dataset

