import os
import pandas as pd
import numpy as np
import torch


from sklearn.model_selection import ParameterGrid
from utils.config import load_config
from models.xlearner import X_Learner_BART
from models.xlearner import X_Learner_RF
from models.slearner import SLearner
from models.tlearner import TLearner
from models.cfrnet import CFR

from CounterfactualAugmentation.src.data_augmentation.local_regressor import impute_missing_values_embeddings
from utils.upload_data import load_data
from utils.perf import perf_epehe_e_ate
from utils.generate_synthetic_data import generate_linear, generate_non_linear
from data_augmentation import contrastive_learning 
from data_augmentation import similarity_measure
from data_augmentation import local_regressor
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
    config = load_config('config.yaml')
    dataset_name = config['dataset_name']
    params = config['params']
    X, treatment, y_factual, y0,y1, mu0, mu1 = load_data(dataset_name)


    # Data Augmentation
    similarity = config['similarity_measure']
    local_regressor = config['local_regressor']
    gp_kernel = config['gp_kernel']
    
    sim_measure = similarity_measure.SimilarityMeasures(similarity)
    embeddings = sim_measure.compute_similarity(X,treatment,y_factual)
    
    # impute the data
    imputed_data = impute_missing_values_embeddings(embeddings, 
                                                    np.column_stack((X, treatment, y_factual)), 
                                                    k = params['num_neighbors'],
                                                    distance_threshold = params['distance_threshold'],
                                                    local_regressor=local_regressor, 
                                                    gp_kernel=gp_kernel)
    imputed_data = local_regressor.data_preprocessing(X,imputed_data)

    # Extract Augmented Data
    X_augmented = imputed_data[:, :-3]
    treatment_augmented = imputed_data[:, -2]
    y_factual_augmented = imputed_data[:, -1]

    # Training Causal Inference Model
    model_name = config['model_name']
    if model_name == 'causal_forest':
        model = X_Learner_RF()
    elif model_name == 'bart':
        model = X_Learner_BART(n_trees=10, random_state=2)
    elif model_name == 'slearner':
        model = SLearner()
    elif model_name == 'cfrnet':
        model = CFR()
    elif model_name == 'tlearner':
        model = TLearner()

    model.fit(X_augmented.cup().numpy(), treatment_augmented.cpu().numpy(), y_factual_augmented.cpu().numpy())
    ite_pred = model.predict(X.cpu().numpy())
    results = perf_epehe_e_ate(mu0,mu1, ite_pred.cpu().numpy())
    print(f"Results for model {model_name}:")
    print(results)

if __name__ == "__main__":
    main()




