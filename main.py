import os
import pickle
import pandas as pd
import numpy as np
import torch
import time

from sklearn.model_selection import ParameterGrid
from src.utils.config import load_config
from src.models.xlearner import X_Learner_BART
from src.models.xlearner import X_Learner_RF
from src.models.slearner import SLearner
from src.models.tlearner import TLearner
from src.models.cfrnet import CFR

from src.data_augmentation.local_regressor import impute_missing_values_embeddings, data_preprocessing
from src.utils.upload_data import load_data
from src.utils.perf import perf_epehe_e_ate
from src.utils.generate_synthetic_data import generate_linear, generate_non_linear
from src.data_augmentation import contrastive_learning 
from src.data_augmentation import similarity_measure
from src.data_augmentation import local_regressor

from sklearn.model_selection import train_test_split

# Seeds for reproducibility 
np.random.seed(2023)
torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
torch.cuda.manual_seed_all(2023)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main(k=None,r=None):

    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("The device is: ", device)
    # Load Config and Data
    config = load_config('configs/config.yaml')
    dataset_name = config['dataset_name']
    params = config['params']

    k = k if k is not None else params['num_neighbors']
    r = r if r is not None else params['distance_threshold']

    X, treatment, y_factual, y0,y1, mu0, mu1 = load_data(dataset_name)

    # Split into training and test sets
    # The random seed won't be fixed for the splitting in order to have variance in the results and estimate this variance
    # sample an integer randomly bewteen 0 and 1000
     # Generate a random seed for the split, based on the current time
    time_seed = int(time.time()) % 1000  # Ensures the seed is within the valid range for randint
    print("The time seed is: ", time_seed)
    

    # The only variance in the results should come from the random seed for the splitting
    X_train, X_test, treatment_train, treatment_test, y_factual_train, y_factual_test, y0_train, y0_test, y1_train, y1_test, mu0_train, mu0_test, mu1_train, mu1_test = train_test_split(
        X, treatment, y_factual, y0, y1, mu0, mu1, test_size=0.2,random_state=time_seed)  # 20% of the data as the test set

    # Seeds for reproducibility 
    np.random.seed(2023)

    # Data Augmentation
    similarity = config['similarity_measure']
    local_regressor = config['local_regressor']
    gp_kernel = config['gp_kernel']
    contrastive_trained = config['contrastive_trained']


    sim_measure = similarity_measure.SimilarityMeasures(similarity)
    # add the dataeset name to the model path
    model_path = 'model_{}.pt'.format(dataset_name)
    embeddings = sim_measure.compute_similarity(X_train,treatment_train,y_factual_train,contrastive_trained=contrastive_trained,model_path=model_path)
    
    # impute the data
    imputed_data = impute_missing_values_embeddings(embeddings, 
                                                    np.column_stack((X_train, treatment_train, y_factual_train)), 
                                                    k = k,
                                                    distance_threshold = r,
                                                    local_regressor=local_regressor, 
                                                    gp_kernel=gp_kernel)
    
    dataset = np.column_stack((X_train, treatment_train, y_factual_train))
    print("The shape of the original dataset is: ", dataset.shape)
    print("The shape of the imputed dataset is: ", imputed_data.shape)
    
    print("The distance threshold is {}".format(params['distance_threshold']))

    imputed_data = data_preprocessing(dataset,imputed_data)
    print("After preprocessing, the shape of the data is: ")
    print(imputed_data.shape)
    # Ther percentage of imputed values is
    print("The percentage of imputed values is: ", (imputed_data.shape[0]-dataset.shape[0])/imputed_data.shape[0])


    # Extract Augmented Data
    X_augmented = imputed_data[:, :-2]
    treatment_augmented = imputed_data[:, -2]
    y_factual_augmented = imputed_data[:, -1]

    #print("The shape of the augmented data is: ", X_augmented.shape)
    #print("The shape of the treatment augmented data is: ", treatment_augmented.shape)
    #print("The shape of the y_factual augmented data is: ", y_factual_augmented.shape)
    # Training Causal Inference Model
    print("-----Training Causal Inference Model-----")
    model_name = config['model_name']
    if model_name == 'causal_forest':
        model = X_Learner_RF()
        # check if cpu is necessary (I think u can do everything directly on gpu)
        model.fit(X_augmented.cpu().numpy(), treatment_augmented.cpu().numpy(), y_factual_augmented.cpu().numpy())
         # test the peroforamnce of the model on the test data
        ite_pred_test = model.predict(X_test)
    elif model_name == 'bart':
        # check if cpu is necessary (I think u can do everything directly on gpu)
        model = X_Learner_BART(n_trees=100, random_state=2)
        model.fit(X_augmented.cpu().numpy(), treatment_augmented.cpu().numpy(), y_factual_augmented.cpu().numpy())
         # test the peroforamnce of the model on the test data
        ite_pred_test = model.predict(X_test)
    elif model_name == 'slearner':
        slearner_params = config['slearner_params']
        model = SLearner(input_dim=X.shape[1]+1, output_dim=1, hyp_dim=slearner_params['hyp_dim'])
        model.fit(X_augmented, treatment_augmented, y_factual_augmented, epochs=slearner_params['epochs'], batch=slearner_params['batch'], lr=slearner_params['lr'], decay=slearner_params['decay'])
         # test the peroforamnce of the model on the test data
        ite_pred_test = model.predict(X_test.to(device)).cpu()
    elif model_name == 'cfrnet':
        cfrnet_params = config['cfrnet_params']
        # to fill
        model = CFR(input_dim=X.shape[1], output_dim=1, rep_dim=cfrnet_params['rep_dim'], hyp_dim=cfrnet_params['hyp_dim'])
        model.fit(X_augmented, treatment_augmented, y_factual_augmented, epochs=cfrnet_params['epochs'], batch=cfrnet_params['batch'], lr=cfrnet_params['lr'], decay=cfrnet_params['decay'], alpha=cfrnet_params['alpha'], metric=cfrnet_params['metric'])
         # test the peroforamnce of the model on the test data
        ite_pred_test = model.predict(X_test.to(device)).cpu()
    elif model_name == 'tlearner':
        tlearner_params = config['tlearner_params']
        # to fill
        model = TLearner(input_dim=X.shape[1], output_dim=1, hyp_dim=tlearner_params['hyp_dim'])
        model.fit(X_augmented, treatment_augmented, y_factual_augmented, epochs=tlearner_params['epochs'], batch=tlearner_params['batch'], lr=tlearner_params['lr'], decay=tlearner_params['decay'])
         # test the peroforamnce of the model on the test data
        ite_pred_test = model.predict(X_test.to(device)).cpu()
       
    else:
        raise ValueError("Unsupported model type")
    
    
    #results = perf_epehe_e_ate(mu0,mu1, ite_pred)
    results_test = perf_epehe_e_ate(mu0_test, mu1_test, ite_pred_test)

    results_directory = 'result/result_single_runs'  # Specifying the directory where the results should be saved
    os.makedirs(results_directory, exist_ok=True) # Create the directory if it does not exist

    print(f"e_pehe and e_ate for model {model_name}:")
    print(results_test)
    # Save results as a pickle file
    results_filename = os.path.join(results_directory, 'experiment_results.pkl')  # Creating the full path for the results file

    
    if os.path.exists(results_filename):
        with open(results_filename, 'rb') as file:
            results_list = pickle.load(file)
    else:
        results_list = []

    results_list.append({
        'Dataset': dataset_name,
        'Model': model_name,
        'Similarity Measure': similarity,
        'Local Regressor': local_regressor,
        'GP Kernel': gp_kernel,
        'e_pehe': results_test['e_pehe'].item(),
        'e_ate': results_test['e_ate'].item()
    })
    # Saving the updated results list back to the file
    with open(results_filename, 'wb') as file:  # Opening the results file in write-binary mode
        pickle.dump(results_list, file)

    return results_test
if __name__ == "__main__":
    main()




