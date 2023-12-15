import os
import random
import numpy as np
import tensorflow as tf



def set_global_seed(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)   