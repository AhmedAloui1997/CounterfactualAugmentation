# Reference: Künzel, Sören R., et al. "Metalearners for estimating heterogeneous treatment effects 
# using machine learning." Proceedings of the national academy of sciences 116.10 (2019): 4156-4165.
# Link: https://www.pnas.org/doi/10.1073/pnas.1804597116



import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# inmplementation of the causal forests method
class X_Learner_RF:

    def __init__(self, random_state: int = 0):
        self.M1 = RandomForestRegressor(random_state=random_state)
        self.M2 = RandomForestRegressor(random_state=random_state)
        self.M3 = RandomForestRegressor(random_state=random_state)
        self.M4 = RandomForestRegressor(random_state=random_state)
        self.g = LogisticRegression(max_iter=2000, random_state=random_state)
    
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        X0 = X[(T == 0).ravel()]
        X1 = X[(T == 1).ravel()]
        Y0 = Y[(T == 0).ravel()].ravel()
        Y1 = Y[(T == 1).ravel()].ravel()
        self.M1.fit(X0, Y0)
        self.M2.fit(X1, Y1)
        self.g.fit(X, T.ravel())
        D_hat = np.where((T == 0).ravel(), self.M2.predict(X) - Y.ravel(), Y.ravel() - self.M1.predict(X))
        self.M3.fit(X0, D_hat[(T == 0).ravel()])
        self.M4.fit(X1, D_hat[(T == 1).ravel()])

    def predict(self, X):
        return self.g.predict_proba(X)[:,0] * self.M3.predict(X) + self.g.predict_proba(X)[:,1] * self.M4.predict(X)

# implementation of bart
class X_Learner_BART(X_Learner_RF):

    def __init__(self, n_trees: int = 100, random_state: int = 0):
        self.M1 = RandomForestRegressor(n_estimators=n_trees, random_state=random_state)
        self.M2 = RandomForestRegressor(n_estimators=n_trees, random_state=random_state)
        self.M3 = RandomForestRegressor(n_estimators=n_trees, random_state=random_state)
        self.M4 = RandomForestRegressor(n_estimators=n_trees, random_state=random_state)
        self.g = LogisticRegression(max_iter=2000, random_state=random_state)


class Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.layer(x)

class X_Learner_NN:
    def __init__(self, input_size, hidden_size):
        self.M1 = Net(input_size, hidden_size)
        self.M2 = Net(input_size, hidden_size)
        self.M3 = Net(input_size, hidden_size)
        self.M4 = Net(input_size, hidden_size)
        self.g = LogisticRegression(max_iter=2000)

    def fit(self, X, T, Y, epochs=100, batch_size=64, lr=0.01):
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = torch.tensor(X, dtype=torch.float)
        T = torch.tensor(T, dtype=torch.float).view(-1, 1)
        Y = torch.tensor(Y, dtype=torch.float).view(-1, 1)

        # Step 1: Estimate the response
        self._train(self.M1, X[T.squeeze() == 0], Y[T.squeeze() == 0], epochs, batch_size, lr)
        self._train(self.M2, X[T.squeeze() == 1], Y[T.squeeze() == 1], epochs, batch_size, lr)
        
        # Step 2: Impute the treatment effect and estimate the CATE function
        D_hat = torch.where(T == 0, self.M2(X) - Y, Y - self.M1(X))
        self._train(self.M3, X[T.squeeze() == 0], D_hat[T.squeeze() == 0], epochs, batch_size, lr)
        self._train(self.M4, X[T.squeeze() == 1], D_hat[T.squeeze() == 1], epochs, batch_size, lr)

        # Propensity score
        self.g.fit(X.detach().numpy(), T.detach().numpy().ravel())

    def _train(self, model, X, Y, epochs, batch_size, lr):
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            permutation = torch.randperm(X.size()[0])
            for i in range(0, X.size()[0], batch_size):
                optimizer.zero_grad()
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X[indices], Y[indices]
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float)
        propensity_scores = self.g.predict_proba(X.detach().numpy())[:,1]
        return propensity_scores * self.M3(X).detach().numpy() + (1-propensity_scores) * self.M4(X).detach().numpy()


