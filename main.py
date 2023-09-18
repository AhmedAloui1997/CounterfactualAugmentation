import os
import pandas as pd
import numpy as np
import torch


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


# Seed for reproducibility
np.random.seed(2023)
torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
torch.cuda.manual_seed_all(2023)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load Config and Data
    config = load_config('configs/config.yaml')
    dataset_name = config['dataset_name']
    params = config['params']
    X, treatment, y_factual, y0,y1, mu0, mu1 = load_data(dataset_name)

    # Split into training and test sets
    X_train, X_test, treatment_train, treatment_test, y_factual_train, y_factual_test, y0_train, y0_test, y1_train, y1_test, mu0_train, mu0_test, mu1_train, mu1_test = train_test_split(
        X, treatment, y_factual, y0, y1, mu0, mu1, test_size=0.2, random_state=42)  # 20% of the data as the test set



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
                                                    k = params['num_neighbors'],
                                                    distance_threshold = params['distance_threshold'],
                                                    local_regressor=local_regressor, 
                                                    gp_kernel=gp_kernel)
    
    dataset = np.column_stack((X_train, treatment_train, y_factual_train))
    print(dataset.shape)
    print(imputed_data.shape)

    imputed_data = data_preprocessing(dataset,imputed_data)
    print("After preprocessing, the shape of the data is: ")
    print(imputed_data.shape)

    # Extract Augmented Data
    X_augmented = imputed_data[:, :-2]
    treatment_augmented = imputed_data[:, -2]
    y_factual_augmented = imputed_data[:, -1]
    # Training Causal Inference Model
    model_name = config['model_name']
    if model_name == 'causal_forest':
        model = X_Learner_RF()
        # check if cpu is necessary (I think u can do everything directly on gpu)
        model.fit(X_augmented.cpu().numpy(), treatment_augmented.cpu().numpy(), y_factual_augmented.cpu().numpy())
    elif model_name == 'bart':
        # check if cpu is necessary (I think u can do everything directly on gpu)
        model = X_Learner_BART(n_trees=100, random_state=2)
        model.fit(X_augmented.cpu().numpy(), treatment_augmented.cpu().numpy(), y_factual_augmented.cpu().numpy())
    elif model_name == 'slearner':
        # to fill
        model = SLearner()
    elif model_name == 'cfrnet':
        # to fill
        model = CFR(input_dim=X.shape[1], output_dim=1)
        model.fit(X_augmented, treatment_augmented, y_factual_augmented)
    elif model_name == 'tlearner':
        # to fill
        model = TLearner()
    else:
        raise ValueError("Unsupported model type")

    # test the peroforamnce of the model on the test data
    ite_pred_test = model.predict(X_test)

    #results = perf_epehe_e_ate(mu0,mu1, ite_pred)
    results_test = perf_epehe_e_ate(mu0_test, mu1_test, ite_pred_test)

    print(f"Results for model {model_name}:")
    print(results_test)

if __name__ == "__main__":
    main()




