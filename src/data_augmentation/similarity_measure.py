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
    def compute_similarity(self,X,T,Y,epsilon=0.01,contrastive_trained=0,model_path='model.pt'):
        if self.measure_type == 'contrastive':
            if contrastive_trained == 0:
                # train the contrastive learning model
                # hyperparameters for the contrastive learning model
                batch_size = 200
                epsilon = 0.1
                input_dim =25
                embedding_dim = 32
                batch_size = 200
                num_epochs = 100
                learning_rate = 1e-2
                margin = 1.0   
                model = train_model(X, T, Y, epsilon, input_dim, embedding_dim, batch_size, num_epochs, learning_rate, margin)        
                embeddings = model(X)
                torch.save(model.state_dict(), model_path)
                print("Model trained and saved")
                return embeddings
            else:
                # load the trained model
                model = ContrastiveLearningModel(X.shape[1],32)
                model.load_state_dict(torch.load(model_path))
                print("Model loaded")
                embeddings = model(X)
                return embeddings   
        
        elif self.measure_type == 'euclidean':
            return X
        
        # we will use the logistic regression model to predict the propensity scores
        elif self.measure_type == 'propensity':
            # train a logistic regression model to predict propensity scores
            propensity_model = LogisticRegression().fit(X, T)
            propensity_scores = propensity_model.predict_proba(X)[:, 1]
            return propensity_scores.reshape(-1, 1)
        else:
            raise ValueError("Unsupported similarity measure")



