import os
import random

import numpy as np
import tensorflow as tf

from load_data import load_data
from preprocess import preprocess
from config import Config
from train import run_training
import json

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CUR_DIR, 'output', 'parquet')

def set_seed(seed):
    tf.random.set_seed(seed)
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == '__main__':
    set_seed(42)
    config = Config(batch_size=512, learning_rate=0.01, nb_epochs=3)
    data = load_data()
    preprocessed_data = preprocess(data, config.batch_size)
    model, history = run_training(preprocessed_data, config)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'history.json'), 'w') as f:
        json.dump(history.history, f)
    model.save(os.path.join(OUTPUT_DIR, 'model'))
