import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.losses import *
import tqdm
import torch.optim as optim

class TLearner(nn.Module):
  def __init__(self, input_dim, output_dim,hyp_dim=100):
    '''
    input_dim: dimension of the input features, treatment is not considered to be a feature 
    output_dim: dimension of the output data, each potential outcome
    hyp_dim: dimension of the hypothese layers for each potential outcome
    '''
    super().__init__()

    #Potential outcome y0
    func0 = [nn.Linear(input_dim,hyp_dim),nn.ReLU(),nn.Linear(hyp_dim, hyp_dim),nn.ReLU(),nn.Linear(hyp_dim, output_dim)]
    self.func0 = nn.Sequential(*func0)
    #potential outcome y1
    func1 = [nn.Linear(input_dim,hyp_dim),nn.ReLU(),nn.Linear(hyp_dim, hyp_dim),nn.ReLU(),nn.Linear(hyp_dim, output_dim)]
    self.func1 = nn.Sequential(*func1)
    #add batch normalization
    
  def forward(self,X):
  
    
    # Pass the transformed features through potential outcomes predicting networks
    Y0 = self.func0(X)
    Y1 = self.func1(X)

    return Y0, Y1


def train_tlearner(net, data,epochs=500, batch=256, lr=1e-3, decay=0):
  tqdm_epoch = tqdm.trange(epochs)
  optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=decay)
  mse = nn.MSELoss()

  #u = torch.mean(t)
  dim = data.shape[1]-2
  wt = 1.0
  wc = 1.0
  loader = DataLoader(data, batch_size=batch, shuffle=True)
  for _ in tqdm_epoch:
    for tr in loader:
        train_t = tr[:,dim]
        train_X = tr[:,0:dim]      
        train_y = tr[:,dim+1:dim+2]
        train_Y0 = train_y[train_t==0]
        train_Y1 = train_y[train_t==1]
        y0, y1 = net(train_X)
        optimizer.zero_grad()
        loss = wc * mse(y0[train_t==0],train_Y0) + wt * mse(y1[train_t==1],train_Y1) 
        loss.backward()
        optimizer.step()
    tqdm_epoch.set_description('Total Loss: {:3f} --- FL0 = {:3f}, FL1 = {:3f}'.format(loss.cpu().detach().numpy(),mse(y0[train_t==0],train_Y0).cpu().detach().numpy(),mse(y1[train_t==1],train_Y1).cpu().detach().numpy()))
  return net