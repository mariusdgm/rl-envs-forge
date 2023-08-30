import random
import numpy as np

def set_random_seeds(seed=None):
    random.seed(seed)
    np.random.seed(seed)