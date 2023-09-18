import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils.losses import *
import tqdm
import torch.optim as optim

class SLearner(nn.Module):
    def __init__(self, input_dim, output_dim, hyp_dim=100):
        super().__init__()
        func = [nn.Linear(input_dim + 1, hyp_dim),  # +1 for treatment
                nn.ELU(),
                nn.Linear(hyp_dim, hyp_dim),
                nn.ELU(),
                nn.Linear(hyp_dim, output_dim)]
        self.func = nn.Sequential(*func)

    def forward(self, X, t):
        in_ = torch.cat((X, t.unsqueeze(-1)), dim=1)  # Ensuring t is a column tensor
        Y = self.func(in_)
        return Y

    def fit(self, X, treatment, y_factual, epochs=1000, batch=128, lr=1e-3, decay=0):
        X_tensor = torch.from_numpy(X).float()
        treatment_tensor = torch.from_numpy(treatment).float()
        y_factual_tensor = torch.from_numpy(y_factual).float()

        dataset = torch.utils.data.TensorDataset(X_tensor, treatment_tensor, y_factual_tensor)
        loader = DataLoader(dataset, batch_size=batch, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=decay)
        mse = nn.MSELoss()

        tqdm_epoch = tqdm.trange(epochs)
        for _ in tqdm_epoch:
            for (batch_X, batch_t, batch_y) in loader:
                y_pred = self(batch_X, batch_t)
                loss = mse(y_pred, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tqdm_epoch.set_description('Total Loss: {:3f}'.format(loss.item()))

    def predict(self, X):
        X_tensor = torch.from_numpy(X).float()
        y0 = self(X_tensor, torch.zeros(X.shape[0]).float())
        y1 = self(X_tensor, torch.ones(X.shape[0]).float())
        ite_pred = (y1 - y0).numpy()
        return ite_pred
