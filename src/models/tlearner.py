import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import torch.optim as optim

class TLearner(nn.Module):
    def __init__(self, input_dim, output_dim, hyp_dim=100):
        super().__init__()

        # Potential outcome y0
        func0 = [nn.Linear(input_dim, hyp_dim),
                 nn.ReLU(),
                 nn.Linear(hyp_dim, hyp_dim),
                 nn.ReLU(),
                 nn.Linear(hyp_dim, output_dim)]
        self.func0 = nn.Sequential(*func0)

        # Potential outcome y1
        func1 = [nn.Linear(input_dim, hyp_dim),
                 nn.ReLU(),
                 nn.Linear(hyp_dim, hyp_dim),
                 nn.ReLU(),
                 nn.Linear(hyp_dim, output_dim)]
        self.func1 = nn.Sequential(*func1)

    def forward(self, X):
        Y0 = self.func0(X)
        Y1 = self.func1(X)
        return Y0, Y1

    def predict(self, X):
        with torch.no_grad():
            Y0, Y1 = self(X)
        return (Y1 - Y0)

    def fit(self, X, treatment, y_factual, epochs=500, batch=256, lr=1e-3, decay=0):
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
                train_t = tr[:, dim].int()
                train_X = tr[:, 0:dim]
                train_y = tr[:, dim + 1:dim + 2]
                y0_hat, y1_hat = self(train_X)
                optimizer.zero_grad()
                loss = (mse(y0_hat[train_t == 0], train_y[train_t == 0]) + 
                        mse(y1_hat[train_t == 1], train_y[train_t == 1]))
                loss.backward()
                optimizer.step()
                
                tqdm_epoch.set_description('Total Loss: {:3f}'.format(loss.item()))
        
        return self

