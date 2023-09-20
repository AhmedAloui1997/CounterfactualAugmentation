import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils.losses import CFRLoss  # Assuming CFRLoss is defined in utils.losses
import tqdm
import torch.optim as optim

class CFR(nn.Module):
    def __init__(self, input_dim, output_dim, rep_dim=800, hyp_dim=400):
        super(CFR, self).__init__()

        # Representation layer
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, rep_dim),
            nn.ELU(),
            nn.Linear(rep_dim, rep_dim),
            nn.ELU(),
            nn.Linear(rep_dim, rep_dim),
            nn.ELU()
        )

        # Potential outcome for control (y0)
        self.func0 = nn.Sequential(
            nn.Linear(rep_dim, hyp_dim),
            nn.ELU(),
            nn.Linear(hyp_dim, hyp_dim),
            nn.ELU(),
            nn.Linear(hyp_dim, output_dim)
        )

        # Potential outcome for treated (y1)
        self.func1 = nn.Sequential(
            nn.Linear(rep_dim, hyp_dim),
            nn.ELU(),
            nn.Linear(hyp_dim, hyp_dim),
            nn.ELU(),
            nn.Linear(hyp_dim, output_dim)
        )

    def forward(self, X):
        Phi = self.encoder(X)
        Y0 = self.func0(Phi)
        Y1 = self.func1(Phi)
        return Phi, Y0, Y1

    def fit(self, X, treatment, y_factual, epochs=100, batch=128, lr=1e-3, decay=0, alpha=3, metric="W1"):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device) # Move model to GPU if CUDA is available
        data = torch.cat((X, treatment.unsqueeze(-1), y_factual.unsqueeze(-1)), dim=1)
        dim = data.shape[1] - 2
        cfr_loss = CFRLoss(alpha=alpha, metric=metric)
        tqdm_epoch = tqdm.trange(epochs)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=decay)
        mse = nn.MSELoss()
        loader = DataLoader(data, batch_size=batch, shuffle=True)
        
        for _ in tqdm_epoch:
            for tr in loader:
                tr = tr.to(device)
                train_t = tr[:, dim].int()
                train_X = tr[:, 0:dim]
                train_y = tr[:, dim + 1:dim + 2]
                phi, y0_hat, y1_hat = self(train_X)
                optimizer.zero_grad()
                loss = cfr_loss(y1_hat, y0_hat, train_y[train_t == 1], train_y[train_t == 0], train_t, phi)
                loss.backward()
                optimizer.step()
                
                tqdm_epoch.set_description('Total Loss: {:3f} --- Factual Loss for Control: {:3f}, Factual Loss for Treated: {:3f}'.format(
                    loss.item(), 
                    mse(y0_hat[train_t == 0], train_y[train_t == 0]).item(),
                    mse(y1_hat[train_t == 1], train_y[train_t == 1]).item()))
        
        return self

    def predict(self, X):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        X = X.to(device)
        with torch.no_grad():
            _, Y0, Y1 = self(X)
        return Y1 - Y0
