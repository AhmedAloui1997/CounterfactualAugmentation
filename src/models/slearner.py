import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.losses import *
import tqdm
import torch.optim as optim

class SLearner(nn.Module):

  def __init__(self, input_dim, output_dim,hyp_dim=100):
    '''
    input_dim: dimension of the input features, treatment is not considered to be a feature 
    output_dim: dimension of the output data, each potential outcome
    hyp_dim: dimension of the hypothese layers for each potential outcome
    '''
    super().__init__()

    #Potential outcomes
    func = [nn.Linear(input_dim,hyp_dim),nn.ELU(),nn.Linear(hyp_dim, hyp_dim),nn.ELU(),nn.Linear(hyp_dim, output_dim)]
    self.func = nn.Sequential(*func)
    
  def forward(self,X,t):

    # Pass the transformed features through potential outcomes predicting networks
    in_ = torch.cat((X,t.reshape(len(t),1)),dim=1)
    Y = self.func(in_)

    return Y



  def fit(net, data,epochs=1000, batch=128, lr=1e-3, decay=0):
    tqdm_epoch = tqdm.trange(epochs)
    optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=decay)
    mse = nn.MSELoss()
    dim = data.shape[1]-2
    loader = DataLoader(data, batch_size=batch, shuffle=True)
    for _ in tqdm_epoch:
      for tr in loader:
          train_X = tr[:,0:dim] 
          train_t =  tr[:,dim:dim+1]     
          train_y = tr[:,dim+1:dim+2]
          y = net(train_X,train_t)
          optimizer.zero_grad()
          loss = mse(y,train_y)
          loss.backward()
          optimizer.step()
      tqdm_epoch.set_description('Total Loss: {:3f}'.format(loss.cpu().detach().numpy()))
    return net
  
  def predict(self,X):
    # no gradient computation
    with torch.no_grad():
      t0 = torch.zeros(X.shape[0])
      t1 = torch.ones(X.shape[0])
      y0 = torch.cat((X,t0.reshape(len(t0),1)),dim=1)
      y1 = torch.cat((X,t1.reshape(len(t1),1)),dim=1)
    
    return y1 - y0