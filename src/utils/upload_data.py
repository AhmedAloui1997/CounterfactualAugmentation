import pandas as pd
import torch




def upload_ihdp(dir,device):
    # IHDP
    names=['treatment', 'y_factual', 'y_cfactual', 'mu0', 'mu1', 'x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21','x22','x23','x24','x25']
    features_ihdp = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21','x22','x23','x24','x25']
    ihdp = pd.read_csv(dir,header=0,names=names)
    ihdp_t = torch.tensor(ihdp['treatment'].values,dtype=torch.float32,device =device)
    ihdp_y = torch.tensor(ihdp['y_factual'].values,dtype=torch.float32,device = device)
    ihdp_ycf = torch.tensor(ihdp['y_cfactual'].values,dtype=torch.float32,device= device)
    ihdp_mu0 = torch.tensor(ihdp['mu0'].values,dtype=torch.float32, device= device)
    ihdp_mu1 = torch.tensor(ihdp['mu1'].values,dtype=torch.float32, device=device)
    ihdp_X = torch.tensor(ihdp[features_ihdp].values,dtype=torch.float32,device=device)


    ihdp_y1 = torch.zeros_like(ihdp_y)
    ihdp_y1[ihdp_t==1] = ihdp_y[ihdp_t==1]
    ihdp_y1[ihdp_t==0] = ihdp_ycf[ihdp_t==0]

    ihdp_y0 = torch.zeros_like(ihdp_y)
    ihdp_y0[ihdp_t==1] = ihdp_ycf[ihdp_t==1]
    ihdp_y0[ihdp_t==0] = ihdp_y[ihdp_t==0]
    return ihdp_X,ihdp_t,ihdp_y,ihdp_y0,ihdp_y1,ihdp_mu0,ihdp_mu1