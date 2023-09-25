import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import torch.optim as optim

class SLearner(nn.Module):
    def __init__(self, input_dim, output_dim, hyp_dim=100):
        super(SLearner, self).__init__()

        func = [
            nn.Linear(input_dim , hyp_dim),  
            nn.ELU(),
            nn.Linear(hyp_dim, hyp_dim),
            nn.ELU(),
            nn.Linear(hyp_dim, output_dim)
        ]
        self.func = nn.Sequential(*func)

    def forward(self, X, t):
        in_ = torch.cat((X, t), dim=1)  # Ensuring t is a column tensor
        Y = self.func(in_)
        return Y

    def fit(self, X, treatment, y_factual, epochs=100, batch=128, lr=1e-3, decay=0):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)  # Move model to GPU if CUDA is available
        data = torch.cat((X, treatment.unsqueeze(-1), y_factual.unsqueeze(-1)), dim=1)
        dim = data.shape[1] - 2

        mse = nn.MSELoss()
        tqdm_epoch = tqdm.trange(epochs)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=decay)
        loader = DataLoader(data, batch_size=batch, shuffle=True)
        
        for _ in tqdm_epoch:
            for tr in loader:
                tr = tr.to(device)
                train_t = tr[:, dim:dim+1]
                train_X = tr[:, 0:dim]
                train_y = tr[:, dim + 1:dim + 2]
                y_hat = self(train_X, train_t)
                optimizer.zero_grad()
                loss = mse(y_hat, train_y)
                loss.backward()
                optimizer.step()
                
                tqdm_epoch.set_description('Total Loss: {:3f}'.format(loss.item()))
        
        return self

    def predict(self, X):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        X = X.to(device)
        with torch.no_grad():
            y0 = self(X, torch.zeros(X.shape[0], 1).to(device))
            y1 = self(X, torch.ones(X.shape[0], 1).to(device))
        ite_pred = (y1 - y0)
        return ite_pred

