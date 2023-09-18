import torch
import numpy as np
from contrastive_learning import *
from CounterfactualAugmentation.src.data_augmentation.local_regressor import *
from sklearn.linear_model import LogisticRegression


# define the class for the data augmentation
class SimilarityMeasures:
    def __init__(self, measure_type):
        self.measure_type = measure_type

    # the compute similarity class with return the embeddings of the X data
    def compute_similarity(self,X,T,Y,epsilon=0.01):
        if self.measure_type == 'contrastive':
            lr =1e-3

            epsilon = 0.1
            input_dim =25
            embedding_dim = 32
            batch_size = 200
            num_epochs = 100
            learning_rate = 1e-2
            margin = 1.0   
            model = train_model(X, T, Y, epsilon, input_dim, embedding_dim, batch_size, num_epochs, learning_rate, margin)        
            embeddings = model(X)
            return embeddings
        
        elif self.measure_type == 'euclidean':
            return X
        
        # For propensity, it's assumed x1 and x2 are propensity scores
        elif self.measure_type == 'propensity':
            # train a logistic regression model to predict propensity scores
            propensity_model = LogisticRegression().fit(X, T)
            propensity_scores = propensity_model.predict_proba(X)[:, 1]
            return propensity_scores.reshape(-1, 1)
        else:
            raise ValueError("Unsupported similarity measure")



