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



