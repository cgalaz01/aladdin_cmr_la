import os
from typing import Any

import tensorflow as tf

from loader.data_generator import (BaseDataLoader, VoxelmorphDataLoader,
                                   VoxelmorphSegDataLoader)
from tf.metrics.metrics import dice_score
from tf.losses.loss import l1_loss, l2_loss
from tf.losses.deform import BendingEnergy, GradientNorm
from tf.models import aladdin_r

import voxelmorph as vxm


def load_model(model: str, patient: str) -> tf.keras.Model:
    """
    Loads the appropriate image registration model from the 'chekpoint' folder.

    Parameters
    ----------
    model : str
        The model's folder name.
    patient : str
        The patient's name.

    Returns
    -------
    model : tf.keras.Model
        The loaded image registration model.

    """
    if os.path.isdir(os.path.join('..', 'checkpoint', model, patient)):
        model_path = os.path.join('..', 'checkpoint', model, patient, 'model.h5')
    else:
        model_path = os.path.join('..', 'checkpoint', model, 'model.h5')
        
    model = tf.keras.models.load_model(model_path, custom_objects={
        'VecInt': aladdin_r.get_vector_integration(),
        'SpatialTransformer': aladdin_r.get_spatial_transformer(),
        'Negate': aladdin_r.get_flow_negate(),
        'loss': tf.keras.losses.MeanSquaredError(),
        'BendingEnergy': BendingEnergy,
        'displacement_losses': tf.keras.losses.MeanSquaredError(),
        'l1_loss': l1_loss,
        'l2_loss': l2_loss,
        'dice_score': dice_score,
        'GradientNorm': GradientNorm,
        'VxmDense': vxm.networks.VxmDense,
        'VxmDenseSemiSupervisedSeg': vxm.networks.VxmDenseSemiSupervisedSeg})

    return model


def get_data_loader(model_type: str) -> Any:
    """
    Loads the appropriate data loader based on the model type.

    Parameters
    ----------
    model_type : str
        For which models to obtain the data loader for. Valid parameters are
        'aladdin_r', 'vxm', 'vxmseg'

    Returns
    -------
    data_loader : Any
        Returns the data loader suitable for the model type.

    """
    if model_type == 'aladdin_r':
        return BaseDataLoader(memory_cache=False, disk_cache=False)
    elif model_type == 'vxm':
        return VoxelmorphDataLoader(memory_cache=False, disk_cache=False)
    elif model_type == 'vxmseg':
        return VoxelmorphSegDataLoader(memory_cache=False, disk_cache=False)