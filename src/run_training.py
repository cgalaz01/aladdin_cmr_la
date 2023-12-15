import os
import datetime

from tensorflow import keras


from loader.tf_generator import (TensorFlowDataGenerator, TensorFlowVxmDataGenerator,
                                 TensorFlowVxmOverlayDataGenerator,
                                 TensorFlowVxmSegDataGenerator)
from configuration import HyperParameters

from tf.callbacks.callbacks import get_reg_callbacks
from tf.metrics import dice_score
from tf.models import aladdin_r, voxelmorph
from tf.losses.loss import l1_loss, l2_loss
from tf.losses.deform import BendingEnergy, GradientNorm
from tf.utils.seed import set_global_seed

import run_segmentation_evaluation
import run_displacement_field_evaluation
import run_output_generation

import voxelmorph as vxm

    

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
        translation_alignment = hparams[hyper_parameters.HP_ALIGN]
        
        
        if hparams[hyper_parameters.HP_CONTOUR]:
            contour_properties = {}
            contour_properties['dilation_radius'] = hparams[hyper_parameters.HP_C_DILATION]
        else:
            contour_properties = None
        
        if hparams[hyper_parameters.HP_DATA_TYPE]:
            data_type = hparams[hyper_parameters.HP_DATA_TYPE]
        else:
            data_type = None
        
        # TODO: as function call
        if model_type == 'vxm':
            tf_data_generator = TensorFlowVxmDataGenerator()
        elif model_type == 'vxmoverlay':
            tf_data_generator = TensorFlowVxmOverlayDataGenerator()
        elif model_type == 'vxmseg':
            tf_data_generator = TensorFlowVxmSegDataGenerator()
        else:
            tf_data_generator = TensorFlowDataGenerator()
        
        (train_gen, data_gen) = tf_data_generator.get_generators(batch_size=batch_size,
                                                                 max_buffer_size=None,
                                                                 memory_cache=True,
                                                                 disk_cache=True,
                                                                 contour=contour_properties,
                                                                 patient_case=patient_case,
                                                                 translation_alignment=translation_alignment,
                                                                 data_type=data_type)
        
        # TODO: as function call
        if model_type == 'aladdin_r':
            model = aladdin_r.get_model_3d(data_gen.image_size)
        elif model_type == 'vxm' or model_type == 'vxmoverlay':
            model = voxelmorph.get_model(data_gen.image_size[:-1])
        elif model_type == 'vxmseg':
            model = voxelmorph.get_model_seg(data_gen.image_size[:-1])


        flow_loss_type = hparams[hyper_parameters.HP_FLOW_LOSS]
        flow_loss_lambda = hparams[hyper_parameters.HP_FLOW_LAMBDA]
        # TODO: as function call
        if flow_loss_type == 'l1':
            flow_loss = l1_loss
        elif flow_loss_type == 'l2':
            flow_loss = l2_loss
        elif flow_loss_type == 'energy':
            flow_loss = BendingEnergy()
        elif flow_loss_type == 'gradl1':
            flow_loss = GradientNorm(l1=True)
        elif flow_loss_type == 'gradl2':
            flow_loss = GradientNorm(l1=False)
        elif flow_loss_type == 'nan':
            flow_loss = l1_loss # Any loss will do as we set lambda to 0
            flow_loss_lambda = 0
        
        # TODO: as function call
        img_loss = vxm.losses.MutualInformation(nb_bins=128).loss
        seg_loss = vxm.losses.Dice().loss
        
        if model_type == 'vxm' or model_type == 'vxmoverlay':
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
            

        learning_rate = hparams[hyper_parameters.HP_LEANRING_RATE]
        if hparams[hyper_parameters.HP_OPTIMISER] == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate,
                                              clipvalue=1.0)

        model.compile(optimizer=optimizer,
                      loss=loss,
                      loss_weights=loss_weights,
                      metrics=metrics)
            
        prefix = model_type
        if data_type:
            prefix += '_' + data_type
            
        if contour_properties is not None:
            name = '_contour_' + str(contour_properties['dilation_radius'])
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
