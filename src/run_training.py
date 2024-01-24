import os
import datetime
from typing import Any, Callable, Dict, Optional, Tuple

import tensorflow as tf
from tensorflow import keras


from loader.tf_generator import (TensorFlowDataGenerator,
                                 TensorFlowVxmDataGenerator,
                                 TensorFlowVxmSegDataGenerator)
from configuration.configuration_aladdin_r import HyperParameters

from tf.callbacks.callbacks import get_reg_callbacks
from tf.metrics import dice_score
from tf.models import aladdin_r, voxelmorph
from tf.losses.loss import l1_loss, l2_loss
from tf.losses.deform import BendingEnergy, GradientNorm
from tf.utils.seed import set_global_seed

import run_output_generation

import voxelmorph as vxm

    

def get_data_generator(model_type: str, patient_case: str, 
                       dilation_radius: Optional[float] = None) -> Tuple[tf.data.Dataset, Any]:
    """
    Returns the data generator for the given model type.

    Parameters
    ----------
    model_type : str
        The model type. Can be 'aladdin_r', 'vxm' or 'vxmseg'.
    patient_case: str
        The patient to train for.
    dilation_radius : float, optional
        The dilation radius of the contour. The default is None.

    Returns
    -------
    train_gen : tf.data.Dataset
        The tensorflow training data generator.
    data_gen : Any
        The data generator.

    """
    if model_type == 'aladdin_r':
        tf_data_generator = TensorFlowDataGenerator()
    elif model_type == 'vxm':
        tf_data_generator = TensorFlowVxmDataGenerator()
    elif model_type == 'vxmseg':
        tf_data_generator = TensorFlowVxmSegDataGenerator()
    else:
        return None
    
    train_gen, data_gen = tf_data_generator.get_generators(batch_size=batch_size,
                                                           max_buffer_size=None,
                                                           memory_cache=True,
                                                           disk_cache=True,
                                                           dilation_radius=dilation_radius,
                                                           patient_case=patient_case,
                                                           translation_alignment=True,
                                                           data_type=data_type)
    
    return train_gen, data_gen
    

def get_model(model_type: str, generator: Any) -> keras.Model:
    """
    Returns the model for the given model type.

    Parameters
    ----------
    model_type : str
        The desired model. Can be 'aladdin_r', 'vxm' or 'vxmseg'.
    generator : Any
        The respective data generator of the model.

    Returns
    -------
    model : tf.keras.Model
        The specified model.

    """
    if model_type == 'aladdin_r':
        return aladdin_r.get_model_3d(generator.image_size)
    elif model_type == 'vxm':
        return voxelmorph.get_model(generator.image_size[:-1])
    elif model_type == 'vxmseg':
        return voxelmorph.get_model_seg(generator.image_size[:-1])
        
    return None


def get_flow_loss(flow_loss_type: str, flow_loss_lambda: float) -> Tuple[Callable, float]:
    """
    Reutnr the flow loss function used to regulate the displacement vector field
    and the lambda value.

    Parameters
    ----------
    flow_loss_type : str
        The flow loss type. Can be 'l1', 'l2', 'energy', 'gradl1', 'gradl2' or
        'nan'.
    flow_loss_lambda : float
        The weight value of the flow loss.

    Returns
    -------
    flow_loss : Callable
        The flow loss function.
    flow_loss_lambda : Float
        The weight value of the flow loss as given. If flow_loss_type is 'nan'
        then 0 is returned.

    """
    if flow_loss_type == 'l1':
        return l1_loss, flow_loss_lambda
    elif flow_loss_type == 'l2':
        return l2_loss, flow_loss_lambda
    elif flow_loss_type == 'energy':
        return BendingEnergy(), flow_loss_lambda
    elif flow_loss_type == 'gradl1':
        return GradientNorm(l1=True), flow_loss_lambda
    elif flow_loss_type == 'gradl2':
        return GradientNorm(l1=False), flow_loss_lambda
    elif flow_loss_type == 'nan':
        return l1_loss, 0 # Any loss will do as we set lambda to 0
    else:
        return None


def get_model_losses(model_type: str, flow_loss_type: str,
                     flow_loss_lambda: float) -> Tuple[Dict[str, Any]]:
    """
    Returns the loss, loss weights and metrics for the given model type.

    Parameters
    ----------
    model_type : str
        The model type. Can be 'aladdin_r', 'vxm' or 'vxmseg'.
    flow_loss_type : str
        The flow loss type. Can be 'l1', 'l2', 'energy', 'gradl1', 'gradl2' or
        'nan'.
    flow_loss_lambda : float
        The weight value of the flow loss.

    Returns
    -------
    loss : Dict[str, Any]
        The loss function.
    loss_weights : Dict[str, Any]
        The respective loss weights.
    metrics : Dict[str, Any]
        The metrics.

    """
    img_loss = vxm.losses.MutualInformation(nb_bins=128).loss
    seg_loss = vxm.losses.Dice().loss
    flow_loss, flow_loss_lambda = get_flow_loss(flow_loss_type, flow_loss_lambda)
    
    if model_type == 'vxm':
        loss = {'vxm_transformer': img_loss,
                'vxm_flow': flow_loss}
        loss_weights={'vxm_transformer': 1.0,
                      'vxm_flow': flow_loss_lambda}
        metrics = {}
    elif model_type == 'vxmseg':
        loss = {'vxm_dense_transformer': img_loss,
                'vxm_seg_transformer': seg_loss,
                'vxm_dense_flow': flow_loss}
        loss_weights={'vxm_dense_transformer': 1.0,
                      'vxm_seg_transformer': 1.0,
                      'vxm_dense_flow': flow_loss_lambda}
        metrics = {}
    elif model_type == 'aladdin-r':
        loss = {'output_fixed': img_loss,
                'output_moving': img_loss,
                'flow_params': flow_loss}
        loss_weights={'output_fixed': 1.0,
                      'output_moving': 1.0,
                      'flow_params': flow_loss_lambda}
        metrics={'output_fixed_seg': dice_score}
    
    return loss, loss_weights, metrics
    


if __name__ == '__main__':
    hyper_parameters = HyperParameters('grid')
    
    date_str = datetime.datetime.now().strftime('_%Y%m%d-%H%M%S')
    models_to_evaluate = {}
    
    for hparams in hyper_parameters:
        keras.backend.clear_session()
        set_global_seed(hparams[hyper_parameters.HP_SEED])
    
        model_type = hparams[hyper_parameters.HP_MODEL]
        batch_size = hparams[hyper_parameters.HP_BATCH_SIZE]
        
        patient_case = hparams[hyper_parameters.HP_PATIENT]
        if patient_case == False:
           patient_case = None
        
        if hparams[hyper_parameters.HP_C_DILATION]:
            dilation_radius = hparams[hyper_parameters.HP_C_DILATION]
        else:
            dilation_radius = None
        
        if hparams[hyper_parameters.HP_DATA_TYPE]:
            data_type = hparams[hyper_parameters.HP_DATA_TYPE]
        else:
            data_type = None
        
        # Prepare data and model
        train_gen, data_gen = get_data_generator(model_type, patient_case, dilation_radius)
        model = get_model(model_type, data_gen)
        # Prepare optimizer
        flow_loss_type = hparams[hyper_parameters.HP_FLOW_LOSS]
        flow_loss_lambda = hparams[hyper_parameters.HP_FLOW_LAMBDA]
        loss, loss_weights, metrics = get_model_losses(model_type, flow_loss_type, flow_loss_lambda)
        learning_rate = hparams[hyper_parameters.HP_LEANRING_RATE]
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate,
                                          clipvalue=1.0)
        
        model.compile(optimizer=optimizer,
                      loss=loss,
                      loss_weights=loss_weights,
                      metrics=metrics)
            
        prefix = model_type
        if data_type:
            prefix += '_' + data_type
            
        if dilation_radius is not None:
            name = '_contour_' + str(dilation_radius)
            prefix += name
            
        if flow_loss_type:
            prefix += '_' + flow_loss_type + '_' + str(flow_loss_lambda)
            
            
        folder_name = prefix + date_str
        models_to_evaluate[folder_name] = data_gen
        if patient_case:
            folder_name = os.path.join(folder_name, patient_case)
        checkpoint_path = os.path.join('..', 'checkpoint', folder_name)
        checkpoint_model_path = os.path.join(checkpoint_path, 'model.h5')
        epochs = hparams[hyper_parameters.HP_EPOCHS]
        model.fit(x=train_gen,
                  epochs=epochs,
                  callbacks=get_reg_callbacks(checkpoint_model_path, folder_name, hparams),
                  verbose=1)
    
    # Save outputs
    for model_path, data_gen in models_to_evaluate.items():
        run_output_generation.main(model_path, data_gen.cache_directory, model_type,
                                   False, 'nifti')
