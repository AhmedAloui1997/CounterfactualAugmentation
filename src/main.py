import os
import pandas as pd
from sklearn.model_selection import ParameterGrid
from utils.config import load_config
from models.xlearner import *
from models.slearner import *
from models.tlearner import *
from models.cfrnet import *

from data_augmentation import contrastive_learning 
from utils.upload_data import *
import numpy as np
from data_augmentation.gp_counterfactual import *
from utils.perf import *
from utils.generate_synthetic_data import *
#from utils import generate_linear

import pickle
# To make this notebook's output stable across runs
np.random.seed(2023)
torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
torch.cuda.manual_seed_all(2023)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load the configuration
    config = load_config('configs/config.yaml')


    # Load the dataset
    #if config['data']['type'] == "semi-synthetic":
    #    data = upload_ihdp('/hpc/home/aa671/phd/counterfactual_sampling/src/ihdp/ihdp_npci_1.csv',device='cpu')
    #    print('Dataset Uploaded!')
    #    lr =1e-4
    #    decay = 0.3#0.001
    #    batch = 256
    #    EPOCHS = 500

    data = generate_non_linear()

    #hyperparameters for the contrastive learning network
    epsilon_contrastive = config['contrastive_learning']['epsilon_contrastive']
    embedding_dim = config['contrastive_learning']['embedding_dim']
    batch_size = config['contrastive_learning']['batch_size']
    num_epochs = config['contrastive_learning']['num_epochs']
    learning_rate = config['contrastive_learning']['learning_rate']
    margin = config['contrastive_learning']['margin']
    print(data[0].shape)
    #train the contrastive learning network
    print("-------- Training Contrastive Learning ----------")

    model_path = 'model_contrastive_non_linear'
    # Replace `ModelClass` with the actual class of your model
    contrastive_model = contrastive_learning.ContrastiveLearningModel(data[0].shape[1],embedding_dim)

    # Check if the model exists
    if os.path.isfile(model_path):
        # Load the trained weights
        contrastive_model.load_state_dict(torch.load(model_path))
        print("Model loaded")
    else:
        # Train the model
        contrastive_model = contrastive_learning.train_model(data[0], 
                                                data[1], 
                                                data[2], 
                                                epsilon_contrastive,
                                                data[0].shape[1],
                                                embedding_dim,
                                                batch_size,
                                                num_epochs,
                                                learning_rate,
                                                margin)
        # Save the model
        torch.save(contrastive_model.state_dict(), model_path)
        print("Model trained and saved")
    #change this to an If statement later.

    print("-------- Finished Training ----------")
        # Assume contrastive_model is your trained model
    #torch.save(contrastive_model.state_dict(), )
    # Create a new DataFrame with columns: X, T, Y1, and Y0
    column_names = ["x" + str(i+1) for i in range(data[0].shape[1])]
    partially_observed_df = pd.DataFrame(data[0].numpy(),columns = column_names)
    partially_observed_df['treatment'] = data[1].numpy()
    partially_observed_df['Y0'] = np.where(data[1] == 0, data[4], None)
    partially_observed_df['Y1'] = np.where(data[1] == 1, data[5], None)

    data_factual = torch.cat((data[0],data[1].reshape(len(data[1]),1),data[2].reshape(len(data[2]),1)),dim=1)
    # Define the hyperparameters for the ablation study
    hyperparameters = {
        'num_neighbors': [2,5,10,15,20],
        'distance_threshold': [0,0.02,0.04,0.06,0.08,0.1],
    }

    # Create a grid of hyperparameters
    grid = ParameterGrid(hyperparameters)

    
    results_bart = {}
    results_rf ={}
    results_tlearner = {}
    results_slearner = {}
    results_tarnet = {}
    results_wass = {}
    resutls_mmd = {}

    lr =5e-3
    decay = 0.3#0.001
    batch = 259
    EPOCHS = 250
    dim = data[0].shape[1]
    # Loop over the hyperparameters
    for params in grid:
        k = params['num_neighbors']
        distance_threshold = params['distance_threshold']
        # Perform data augmentation
        dataset = partially_observed_df.to_numpy(dtype=np.float32)
        imputed_data = impute_missing_values_embeddings(contrastive_model,dataset, k, distance_threshold)
        augmented_data = data_preprocessing(data_factual,imputed_data) 
        print(augmented_data.shape)
        # Train the model on the augmented data
        augmented_data = augmented_data.to(device)
        per = ((augmented_data.shape[0] - len(data[1]))/len(data[1]))
        net = CFR(dim,1,128,64).to(device)
        per = ((augmented_data.shape[0] - len(data[1]))/len(data[1]))
        net = train_cfr(net,augmented_data,EPOCHS,batch,lr,decay,alpha=0,metric="W1")
        net.eval()
        _,y0_hat,y1_hat = net(data[0].to(device))
        predictions = y1_hat - y0_hat
        e_pehe,e_ate = perf_epehe_e_ate(data[5],data[6],predictions.cpu())
        results_tarnet[(k, distance_threshold)] = {'per': per, 'e_pehe': e_pehe, 'e_ate': e_ate}
        print('TARNet: (K,eps,e_pehe,e_ate,per) = ',k,distance_threshold,e_pehe,e_ate,per)

        #Train cfrnet-wass
        #net = CFR(dim,1,128,64).to(device)
        #per = ((augmented_data.shape[0] - len(data[1]))/len(data[1]))
        #net = train_cfr(net,augmented_data,EPOCHS,batch,lr,decay,alpha=3.0,metric="W1")
        #net.eval()
        #_,y0_hat,y1_hat = net(data[0].to(device))
        #predictions = y1_hat - y0_hat
        #e_pehe,e_ate = perf_epehe_e_ate(data[5],data[6],predictions.cpu())
        #results_wass[(k, distance_threshold)] = {'per': per, 'e_pehe': e_pehe, 'e_ate': e_ate}
        #print('CFR-Wass: (K,eps,e_pehe,e_ate,per) = ',k,distance_threshold,e_pehe,e_ate,per)
        #Train cfrnet-mmd
        #net = CFR(dim,1,128,64).to(device)
        #per = ((augmented_data.shape[0] - len(data[1]))/len(data[1]))
        #net = train_cfr(net,augmented_data,EPOCHS,batch,lr,decay,alpha=3.0,metric="MMD")
        #net.eval()
        #_,y0_hat,y1_hat = net(data[0].to(device))
        #predictions = y1_hat - y0_hat
        #e_pehe,e_ate = perf_epehe_e_ate(data[5],data[6],predictions.cpu())
        #results_mmd[(k, distance_threshold)] = {'per': per, 'e_pehe': e_pehe, 'e_ate': e_ate}
        #print('CFR-MMD: (K,eps,e_pehe,e_ate,per) = ',k,distance_threshold,e_pehe,e_ate,per)
        # Initialize and train the model
        #xlearner_rf = X_Learner_RF(random_state=2)#.to(deivce)
        #xlearner_rf.fit(augmented_data[:,:-2].cpu().numpy(), augmented_data[:,-2].cpu().numpy(),augmented_data[:,-1].cpu().numpy())
        # Make predictions
        #predictions = torch.tensor(xlearner_rf.predict(data[0]))
        #e_pehe,e_ate = perf_epehe_e_ate(data[5],data[6],predictions)
        # Store the results in the dictionary
        #results_rf[(k, distance_threshold)] = {'per': per, 'e_pehe': e_pehe, 'e_ate': e_ate}
        # Save the results dictionary into a pickle file
        #print('RF: (K,eps,e_pehe,e_ate,per) = ',k,distance_threshold,e_pehe,e_ate,per)
        #xlearner_bart = X_Learner_BART(n_trees=10, random_state=2)
        #xlearner_bart.fit(augmented_data[:,:-2].cpu().numpy(), augmented_data[:,-2].cpu().numpy(),augmented_data[:,-1].cpu().numpy())
        #predictions = torch.tensor(xlearner_bart.predict(data[0]))
        #e_pehe,e_ate = torch.tensor(perf_epehe_e_ate(data[5],data[6],predictions))
        #results_bart[(k, distance_threshold)] = {'per': per, 'e_pehe': e_pehe, 'e_ate': e_ate}
        #print('BART: (K,eps,e_pehe,e_ate,per) = ',k,distance_threshold,e_pehe,e_ate,per)
        #net = TLearner(dim,1,64).to(device)
        #net = train_tlearner(net,augmented_data)
        #net.eval()
        #y0_hat,y1_hat = net(data[0].to(device))
        #predictions = y1_hat - y0_hat
        #e_pehe,e_ate = perf_epehe_e_ate(data[5],data[6],predictions.cpu())
        #results_tlearner[(k, distance_threshold)] = {'per': per, 'e_pehe': e_pehe, 'e_ate': e_ate}
        #print('T-Learner: (K,eps,e_pehe,e_ate,per) = ',k,distance_threshold,e_pehe,e_ate,per)
        s_learner = SLearner(dim+1,1,64).to(device)
        s_learner = train_slearner(s_learner,augmented_data,EPOCHS,batch,lr,decay)
        s_learner.eval()
        y1_hat =s_learner(data[0].to(device),torch.ones_like(data[1]).to(device))
        y0_hat =s_learner(data[0].to(device),torch.zeros_like(data[1]).to(device))
        predictions = y1_hat - y0_hat
        e_pehe,e_ate = perf_epehe_e_ate(data[5],data[6],predictions.cpu())
        results_slearner[(k, distance_threshold)] = {'per': per, 'e_pehe': e_pehe, 'e_ate': e_ate}
        print('S-Learner: (K,eps,e_pehe,e_ate,per) = ',k,distance_threshold,e_pehe,e_ate,per)

    #with open('results_linear_bart.pickle', 'wb') as handle:
    #    pickle.dump(results_bart, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('results_non_linear_slearner.pickle', 'wb') as handle:
        pickle.dump(results_slearner, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #with open('results_linear_rf.pickle', 'wb') as handle:
    #    pickle.dump(results_rf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #with open('results_linear_tlearner.pickle', 'wb') as handle:
    #    pickle.dump(results_tlearner, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #with open('results_linear_tarnet.pickle', 'wb') as handle:
    #    pickle.dump(results_tarnet, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #with open('results_linear_cfrwass.pickle', 'wb') as handle:
    #    pickle.dump(results_wass, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #with open('results_linear_cfrmmd.pickle', 'wb') as handle:
    #    pickle.dump(results_mmd, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
