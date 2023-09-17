import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,Matern
from sklearn.neighbors import NearestNeighbors
import torch


#this function will impute the missing values for a data with the form (X,T,Y0,Y1)
def impute_missing_values_embeddings(model,data, k=5, distance_threshold=1.0,gp_kernel = "RBF"):
    imputed_data = data.copy()
    X = data[:, :-3]
    Y0_mask = np.isnan(data[:, -1])
    Y1_mask = np.isnan(data[:, -2])
    X_tensor = torch.tensor(X).float()
    X_embeddings = model(X_tensor).detach().numpy()
    # Find nearest neighbors based on X
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="minkowski",p=2).fit(X_embeddings)
    
    # Impute missing values in Y0
    if np.any(Y0_mask):
        Y0_missing_indices = np.argwhere(Y0_mask)[:, 0]
        Y0_observed_indices = np.argwhere(~Y0_mask)[:, 0]
        distances, indices = nbrs.kneighbors(X_embeddings[Y0_missing_indices])
        if gp_kernel == "RBF":
            kernel = RBF(length_scale=0.01)
        else:
            kernel = Matern(nu=0.5)
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0)

        for missing_idx, neighbor_distances, neighbor_indices in zip(Y0_missing_indices, distances, indices):
            observed_neighbor_indices = [i for i in neighbor_indices if i in Y0_observed_indices]
            if np.all(neighbor_distances < distance_threshold) and len(observed_neighbor_indices) > 0:
                gpr.fit(X[observed_neighbor_indices], data[observed_neighbor_indices, -1])
                imputed_data[missing_idx, -1] = gpr.predict([X[missing_idx]])[0]

    # Impute missing values in Y1
    if np.any(Y1_mask):
        Y1_missing_indices = np.argwhere(Y1_mask)[:, 0]
        Y1_observed_indices = np.argwhere(~Y1_mask)[:, 0]
        distances, indices = nbrs.kneighbors(X_embeddings[Y1_missing_indices])
        if gp_kernel == "RBF":
            kernel = RBF(length_scale=0.01)
        else:
            kernel = Matern(nu=0.5)
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0)

        for missing_idx, neighbor_distances, neighbor_indices in zip(Y1_missing_indices, distances, indices):
            observed_neighbor_indices = [i for i in neighbor_indices if i in Y1_observed_indices]
            if np.all(neighbor_distances < distance_threshold) and len(observed_neighbor_indices) > 0:
                gpr.fit(X[observed_neighbor_indices], data[observed_neighbor_indices, -2])
                imputed_data[missing_idx, -2] = gpr.predict([X[missing_idx]])[0]

    return imputed_data

#this function will output the imputed data into a data with only the new imputed factual outcomes.
def data_preprocessing(dataset,imputed_data):  
  rows = dataset.numpy()
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

