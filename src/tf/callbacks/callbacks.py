import os
from typing import Tuple

import tensorflow as tf
from tensorflow import keras

from tensorboard.plugins.hparams import api as hp

from tf.callbacks.weights import IncreaseLROnPlateau, LoadBestWeightsOnNaN



def get_seg_callbacks(checkpoint_directory: str, folder_name: str, hparams) -> Tuple[tf.keras.callbacks]:
    metric_monitor = 'loss'
    mode = 'min'
    
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_directory,
        save_weights_only=False,
        monitor=metric_monitor,
        mode=mode,
        save_best_only=True)
    
    log_dir = os.path.join('..', 'logs', 'fit', folder_name) + '/'
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    hparams_callback = hp.KerasCallback(log_dir, hparams)
    
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor=metric_monitor,
        min_delta=0,
        patience=100,
        mode=mode,
        restore_best_weights=False)
    
    weight_reload = LoadBestWeightsOnNaN()
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=metric_monitor,
        factor=0.95,
        min_delta=1e-4,
        patience=20,
        mode=mode,
        min_lr=1e-9)
    
    
    return [model_checkpoint_callback,
            tensorboard_callback,
            hparams_callback,
            early_stopping_callback,
            weight_reload,
            reduce_lr]


def get_reg_callbacks(checkpoint_directory: str, folder_name: str, hparams) -> Tuple[tf.keras.callbacks]:
    metric_monitor = 'loss'
    mode = 'min'
        
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_directory,
        save_weights_only=False,
        monitor=metric_monitor,
        mode=mode,
        save_best_only=True)
    
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=0,
        patience=20,
        restore_best_weights=False)
    
    log_dir = os.path.join('..', 'logs', 'fit', folder_name) + '/'
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    hparams_callback = hp.KerasCallback(log_dir, hparams)
    
    weight_reload = LoadBestWeightsOnNaN()
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=metric_monitor,
        factor=0.98,
        min_delta=1e-5,
        patience=5,
        mode=mode,
        min_lr=1e-8)
    
    increase_lr = IncreaseLROnPlateau(
        monitor=metric_monitor,
        factor=10.0,
        min_delta=1e-5,
        patience=10,
        mode=mode,
        max_lr=1e-8)
    
    return [model_checkpoint_callback,
            weight_reload,
            reduce_lr,
            increase_lr,
            early_stopping_callback,
            tensorboard_callback,
            hparams_callback]