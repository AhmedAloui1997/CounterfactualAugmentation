import numpy as np
import pandas as pd
import torch



def generate_linear(n_samples=1500,dim=10):
    # Generate a dataset with 4-dimensional features and 1000 samples
    X = np.random.randn(n_samples, dim)

    # Create treatment assignment (t) with some bias
    treatment_prob = 1 / (1 + np.exp(-(X[:, 0])-(X[:, 1])))
    t = (np.random.rand(n_samples) < treatment_prob).astype(int)

    # Generate potential outcomes using a linear function
    coef_y0 = 0.5 * np.ones(dim)
    coef_y1 = 0.3* np.ones(dim)
    mu0 = X @ coef_y0
    mu1 = X @ coef_y1

    y0 = mu0 + 0.01* np.random.randn(n_samples)
    y1 = mu1 + 0.01* np.random.randn(n_samples)
    y = np.zeros_like(y0)
    y[t==0] = y0[t==0]
    y[t==1] = y1[t==1]

    X = torch.tensor(X,dtype=torch.float32)
    t = torch.tensor(t,dtype=torch.float32)
    y = torch.tensor(y,dtype=torch.float32)
    y0 = torch.tensor(y0,dtype=torch.float32)
    y1 = torch.tensor(y1,dtype=torch.float32)
    mu0 = torch.tensor(mu0,dtype=torch.float32)
    mu1 = torch.tensor(mu1,dtype=torch.float32)
    return X,t,y,y0,y1,mu0,mu1




def generate_non_linear(n_samples=1500,dim=10):
    # Generate a dataset with 4-dimensional features and 1000 samples
    X = np.random.randn(n_samples, dim)

    # Create treatment assignment (t) with some bias
    treatment_prob = 1 / (1 + np.exp(-(X[:, 0])-(X[:, 1])))
    t = (np.random.rand(n_samples) < treatment_prob).astype(int)

    # Generate potential outcomes using a linear function
    coef_y0 = 0.5 * np.ones(dim)
    coef_y1 = 0.2* np.ones(dim)
    mu0 = np.exp(X @ coef_y0)
    mu1 = np.exp(X @ coef_y1)
    
    y0 = mu0 + 0.01* np.random.randn(n_samples)
    y1 = mu1 + 0.01* np.random.randn(n_samples)
    y = np.zeros_like(y0)
    y[t==0] = y0[t==0]
    y[t==1] = y1[t==1]

    X = torch.tensor(X,dtype=torch.float32)
    t = torch.tensor(t,dtype=torch.float32)
    y = torch.tensor(y,dtype=torch.float32)
    y0 = torch.tensor(y0,dtype=torch.float32)
    y1 = torch.tensor(y1,dtype=torch.float32)
    mu0 = torch.tensor(mu0,dtype=torch.float32)
    mu1 = torch.tensor(mu1,dtype=torch.float32)
    return X,t,y,y0,y1,mu0,mu1