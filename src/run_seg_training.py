import os
import datetime

from tensorflow import keras

from loader.tf_generator_seg import TensorFlowDataGenerator
from tf.callbacks.callbacks import get_seg_callbacks
from tf.utils.seed import set_global_seed
from configuration.configuration_aladdin_s import HyperParameters
from tf.models import aladdin_s

import run_output_seg_generation

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
        
        (train_gen, validation_gen,
         test_gen, data_gen) = TensorFlowDataGenerator.get_generators(batch_size=batch_size,
                                                                      max_buffer_size=None,
                                                                      memory_cache=True,
                                                                      disk_cache=True,
                                                                      patient_case=patient_case)
        
        if model_type == 'base':
            model = aladdin_s.get_model_3d(data_gen.image_size)
        
        loss = {'output_seg': vxm.losses.Dice().loss}
            

        learning_rate = hparams[hyper_parameters.HP_LEANRING_RATE]
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate,
                                          clipvalue=1.0)

        model.compile(optimizer=optimizer,
                      loss=loss)
            
            
        folder_name = 'aladdin_s' + date_str
        models_to_evaluate[folder_name] = data_gen
        if patient_case:
            folder_name = os.path.join(folder_name, patient_case)
        checkpoint_path = os.path.join('..', 'checkpoint', folder_name)
        checkpoint_model_path = os.path.join(checkpoint_path, 'model.h5')
        epochs = hparams[hyper_parameters.HP_EPOCHS]
        model.fit(x=train_gen,
                  epochs=epochs,
                  callbacks=get_seg_callbacks(checkpoint_model_path, folder_name, hparams),
                  verbose=1,)

    
    # Save outputs
    for model_path, data_gen in models_to_evaluate.items():
        run_output_seg_generation.main(model_path, data_gen.cache_directory, 'nifti')
    
    
