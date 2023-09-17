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



import pandas as pd
import torch
from scipy.sparse import coo_matrix

def upload_news(dir_x, dir_y, device):
    # Read .y file directly as pandas DataFrame
    names_y = ['treatment', 'y_factual', 'y_cfactual', 'mu0', 'mu1']
    y_data = pd.read_csv(dir_y, header=0, names=names_y)

    # Read the .x file and convert it to dense format
    with open(dir_x, 'r') as f:
        # Read matrix dimensions
        rows, cols, _ = map(int, f.readline().strip().split(','))

        # Extract data for sparse matrix
        i_values, j_values, values = [], [], []
        for line in f:
            i, j, v = map(int, line.strip().split(','))
            i_values.append(i-1)  # Adjusting for 0-based index
            j_values.append(j-1)
            values.append(v)

    # Construct sparse matrix and then convert to dense
    x_sparse = coo_matrix((values, (i_values, j_values)), shape=(rows, cols))
    x_data = x_sparse.todense()

    # Convert data to torch tensors
    treatment = torch.tensor(y_data['treatment'].values, dtype=torch.float32, device=device)
    y_factual = torch.tensor(y_data['y_factual'].values, dtype=torch.float32, device=device)
    y_cfactual = torch.tensor(y_data['y_cfactual'].values, dtype=torch.float32, device=device)
    mu0 = torch.tensor(y_data['mu0'].values, dtype=torch.float32, device=device)
    mu1 = torch.tensor(y_data['mu1'].values, dtype=torch.float32, device=device)
    X = torch.tensor(x_data, dtype=torch.float32, device=device)

    # Calculate y0 and y1 similar to IHDP
    y1 = torch.zeros_like(y_factual)
    y1[treatment == 1] = y_factual[treatment == 1]
    y1[treatment == 0] = y_cfactual[treatment == 0]

    y0 = torch.zeros_like(y_factual)
    y0[treatment == 1] = y_cfactual[treatment == 1]
    y0[treatment == 0] = y_factual[treatment == 0]

    return X, treatment, y_factual, y0, y1, mu0, mu1
