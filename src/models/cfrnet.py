import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.losses import *
import tqdm
import torch.optim as optim

class CFR(nn.Module):
  def __init__(self, input_dim, output_dim, rep_dim=200,hyp_dim=100):
    '''
    input_dim: dimension of the input features, treatment is not considered to be a feature 
    output_dim: dimension of the output data, each potential outcome
    rep_dim: dimension of the representation layers, assumed to be the same
    hyp_dim: dimension of the hypothese layers for each potential outcome
    '''
    super().__init__()
    
    #representation layer
    encoder = [nn.Linear(input_dim,rep_dim),nn.ELU(),nn.Linear(rep_dim,rep_dim),nn.ELU(),nn.Linear(rep_dim,rep_dim),nn.ELU()]
    self.encoder = nn.Sequential(*encoder)

    #Potential outcome y0
    func0 = [nn.Linear(rep_dim,hyp_dim),nn.ELU(),nn.Linear(hyp_dim, hyp_dim),nn.ELU(),nn.Linear(hyp_dim, output_dim)]
    self.func0 = nn.Sequential(*func0)
    #potential outcome y1
    func1 = [nn.Linear(rep_dim,hyp_dim),nn.ELU(),nn.Linear(hyp_dim, hyp_dim),nn.ELU(),nn.Linear(hyp_dim, output_dim)]
    self.func1 = nn.Sequential(*func1)
    #add batch normalization
    
  def forward(self,X):
    
    # The input features (covariates) are first mapped in a hidden representation space
    # to measure the distance between Z0 and Z1

    Phi = self.encoder(X)
    
    # Pass the transformed features through potential outcomes predicting networks
    Y0 = self.func0(Phi)
    Y1 = self.func1(Phi)

    return Phi, Y0, Y1
  



  def fit(net, data,epochs=1000, batch=128, lr=1e-3, decay=0, alpha=3,metric="W1"):
      """
      Train a Counterfactual Regression (CFR) network.

      Parameters:
      - net: The CFR network to be trained.
      - data: The training data. Expected format: [features, treatment, outcome].
      - epochs: Number of epochs to train for.
      - batch: Batch size for training.
      - lr: Learning rate for the optimizer.
      - decay: Weight decay for the optimizer.
      - alpha: The balance factor between the factual loss and counterfactual regularizer in the loss function.
      - metric: either Wassertein1 "W1", Wassertein2 "W2", or MMD for "MMD"
      """
      dim = data.shape[1]-2

      # Initialize the Wassertein Loss, which will be used as a counterfactual regularizer
      cfr_loss = CFRLoss(alpha=alpha,metric=metric)

      # Initialize progress bar for training
      tqdm_epoch = tqdm.trange(epochs)

      # Initialize the optimizer
      optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=decay)

      # MSE
      mse = nn.MSELoss()


      # Initialize the data loader
      loader = DataLoader(data, batch_size=batch, shuffle=True)

      # Loop over epochs
      for _ in tqdm_epoch:
          # Loop over batches
          for tr in loader:
              # Extract data for this batch
              train_t = tr[:, dim]
              train_X = tr[:, 0:dim]
              train_y = tr[:, dim + 1:dim + 2]
              train_Y0 = train_y[train_t == 0]
              train_Y1 = train_y[train_t == 1]

              # Forward pass: compute predicted Y and phi
              phi, y0_hat, y1_hat = net(train_X)

              # Zero the gradients before backward pass
              optimizer.zero_grad()

              # Compute the loss: factual loss + alpha * counterfactual regularizer
              loss = cfr_loss(y1_hat,y0_hat,train_Y1,train_Y0,train_t,phi)
              #loss = wc * mse(y0[train_t == 0], train_Y0) + wt * mse(y1[train_t == 1], train_Y1) + alpha * wass(phi[train_t == 0], phi[train_t == 1])

              # Backward pass: compute gradient of the loss with respect to model parameters
              loss.backward()

              # Perform a single optimization step (parameter update)
              optimizer.step()

              # Update progress bar
              tqdm_epoch.set_description('Total Loss: {:3f} --- Factual Loss for Control: {:3f}, Factual Loss for Treated: {:3f}'.format(
                  loss.cpu().detach().numpy(), 
                  mse(y0_hat[train_t == 0], train_Y0).cpu().detach().numpy(),
                  mse(y1_hat[train_t == 1], train_Y1).cpu().detach().numpy()))
      return net

  def predict(self,X):
    with torch.no_grad():
       Phi, Y0, Y1 = self.forward(X)
    return Y1 - Y0