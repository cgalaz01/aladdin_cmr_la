import os
from typing import List

import tensorflow as tf
from tensorflow import keras

from tensorboard.plugins.hparams import api as hp

from tf.callbacks.weights import IncreaseLROnPlateau, LoadBestWeightsOnNaN



def get_reg_callbacks(checkpoint_directory: str, folder_name: str, hparams) -> List[tf.keras.callbacks.Callback]:
    metric_monitor = 'loss'
    mode = 'min'
        
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_directory,
        save_weights_only=False,
        monitor=metric_monitor,
        mode=mode,
        save_best_only=True)
    
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor=metric_monitor,
        min_delta=0,
        patience=50,
        mode=mode,
        restore_best_weights=False)
    
    log_dir = os.path.join('..', 'logs', 'fit', folder_name) + '/'
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    hparams_callback = hp.KerasCallback(log_dir, hparams)
    
    weight_reload = LoadBestWeightsOnNaN()
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=metric_monitor,
        factor=0.98,
        min_delta=1e-6,
        patience=5,
        mode=mode,
        min_lr=1e-8)
    
    increase_lr = IncreaseLROnPlateau(
        monitor=metric_monitor,
        factor=10.0,
        min_delta=1e-6,
        patience=20,
        mode=mode,
        max_lr=1.0)
    
    return [model_checkpoint_callback,
            weight_reload,
            reduce_lr,
            increase_lr,
            early_stopping_callback,
            tensorboard_callback,
            hparams_callback]