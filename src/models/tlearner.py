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
        # Add batch normalization (you mentioned it, but didn't include in the model)

    def forward(self, X):
        Y0 = self.func0(X)
        Y1 = self.func1(X)
        return Y0, Y1

    def predict(self, X):
        X_tensor = torch.from_numpy(X).float()
        with torch.no_grad():
            Y0, Y1 = self(X_tensor)
        return (Y1 - Y0).numpy()

    def fit(self, X, treatment, y_factual, epochs=500, batch=256, lr=1e-3, decay=0):
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
                y0, y1 = self(batch_X)
                optimizer.zero_grad()
                loss = (mse(y0[batch_t == 0], batch_y[batch_t == 0]) + 
                        mse(y1[batch_t == 1], batch_y[batch_t == 1]))
                loss.backward()
                optimizer.step()
            tqdm_epoch.set_description(
                'Total Loss: {:3f}'.format(loss.item())
            )
        return self
