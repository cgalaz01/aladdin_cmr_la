from typing import Dict, Tuple, Optional

import tensorflow as tf

from loader.data_generator_seg import SegDataLoader


class TensorFlowDataGenerator():
    
    @staticmethod
    def _prepare_generators(dg: SegDataLoader, batch_size: int,
                            output_shapes: Tuple[Dict[str, tf.TensorShape]],
                            output_types: Tuple[Dict[str, tf.dtypes.DType]],
                            max_buffer_size: Optional[int] = None) -> Tuple[tf.data.Dataset]:
        """
        Prepare the data generators for training.

        Parameters
        ----------
        dg : SegDataLoader
            The data loader object.
        batch_size : int
            The batch size for the data generator.
        output_shapes : Tuple[Dict[str, tf.TensorShape]]
            The shapes of the output tensors.
        output_types : Tuple[Dict[str, tf.dtypes.DType]]
            The data types of the output tensors.
        max_buffer_size : int, optional
            The maximum buffer size for shuffling the data. Default is None.

        Returns
        -------
        train_generator : tf.data.Dataset
            The prepared training data generator.
        validation_generator : tf.data.Dataset
            The prepared validation data generator.
        test_generator : tf.data.Dataset
            The prepared test data generator.
        dg : SegDataLoader
            The data loader object.

        """
        
        buffer_size = len(dg.train_list) * 3
        if max_buffer_size is not None:
            buffer_size = min(buffer_size, max_buffer_size)    

        generator_type = dg.train_generator
        train_generator = tf.data.Dataset.from_generator(generator_type,
                                                         output_types=output_types,
                                                         output_shapes=output_shapes)
        train_generator = train_generator.shuffle(buffer_size=buffer_size,
                                                  seed=4875,
                                                  reshuffle_each_iteration=True
                                                  ).batch(batch_size).prefetch(2)
        
        generator_type = dg.validation_generator
        validation_generator = tf.data.Dataset.from_generator(generator_type,
                                                              output_types=output_types,
                                                              output_shapes=output_shapes)
        validation_generator = validation_generator.batch(batch_size)
        
        generator_type = dg.test_generator
        test_generator = tf.data.Dataset.from_generator(generator_type,
                                                        output_types=output_types,
                                                        output_shapes=output_shapes)
        test_generator = test_generator.batch(batch_size)
        
        return train_generator, validation_generator, test_generator, dg


    @staticmethod
    def get_generators(batch_size: int, max_buffer_size: Optional[int] = None,
                       memory_cache: bool = False, disk_cache: bool = True,
                       patient_case: Optional[str] = None) -> Tuple[tf.data.Dataset]:
        """
        Get the data generators for training.

        Parameters
        ----------
        batch_size : int
            The batch size for the data generator.
        max_buffer_size : int, optional
            The maximum buffer size for shuffling the data. Default is None.
        memory_cache : bool, optional
            Whether to use memory caching. Default is False.
        disk_cache : bool, optional
            Whether to use disk caching. Default is True.
        patient_case : str, optional
            The patient case. If none then all cases are used. Default is None.

        Returns
        -------
        train_generator : tf.data.Dataset
            The prepared training data generator.
        validation_generator : tf.data.Dataset
            The prepared validation data generator.
        test_generator : tf.data.Dataset
            The prepared test data generator.
        dg : SegDataLoader
            The data loader object.

        """
        dg = SegDataLoader(memory_cache=memory_cache,
                           disk_cache=disk_cache,
                           patient_case=patient_case)
        
        output_shapes = ({'input_img': tf.TensorShape(dg.image_size)},
                         {'output_seg': tf.TensorShape(dg.image_size)})
            
        data_type = tf.float32
        output_types = ({'input_img': data_type},
                        {'output_seg': data_type})


        return TensorFlowDataGenerator._prepare_generators(dg, batch_size,
                                                           output_shapes,
                                                           output_types,
                                                           max_buffer_size)

