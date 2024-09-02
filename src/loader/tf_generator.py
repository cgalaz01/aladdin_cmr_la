from typing import Any, Dict, Tuple, Optional

import tensorflow as tf

from loader.data_generator import (BaseDataLoader,
                                   VoxelmorphDataLoader,
                                   VoxelmorphSegDataLoader)



class TensorFlowDataGenerator():
    
    @staticmethod
    def _prepare_generators(dg: BaseDataLoader, batch_size: int,
                            output_shapes: Tuple[Dict[str, tf.TensorShape]],
                            output_types: Tuple[Dict[str, tf.dtypes.DType]],
                            max_buffer_size: Optional[int] = None) -> Tuple[tf.data.Dataset, BaseDataLoader]:
        """
        Prepare the data generators for training.

        Parameters
        ----------
        dg : BaseDataLoader
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
        dg : BaseDataLoader
            The data loader object.

        """
        buffer_size = len(dg.train_list) * 20
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
        
        return train_generator, dg



    @staticmethod
    def get_generators(batch_size: int, max_buffer_size: Optional[int] = None,
                       memory_cache: bool = False, disk_cache: bool = True,
                       dilation_radius: Optional[float] = None, patient_case: Optional[str] = None,
                       translation_alignment: bool = False, data_type: Optional[str] = None) -> \
                    Tuple[tf.data.Dataset, BaseDataLoader]:
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
        dilation_radius : float, optional
            The dilation radius of the contour. Default is None.
        patient_case : str, optional
            The patient case. If none then all cases are used. Default is None.
        translation_alignment : bool, optional
            Whether to use translation alignment. Default is False.
        data_type : str, optional
            The data type. Default is None.

        Returns
        -------
        train_generator : tf.data.Dataset
            The prepared training data generator.
        dg : BaseDataLoader
            The data loader object.

        """
        dg = BaseDataLoader(memory_cache=memory_cache,
                            disk_cache=disk_cache,
                            dilation_radius=dilation_radius,
                            patient_case=patient_case,
                            translation_alignment=translation_alignment,
                            data_type=data_type)
        
        output_shapes = ({'input_moving': tf.TensorShape(dg.image_size),
                          'input_fixed': tf.TensorShape(dg.image_size),
                          'input_moving_seg': tf.TensorShape(dg.image_size)},
                         {'output_fixed': tf.TensorShape(dg.image_size),
                          'output_fixed_seg': tf.TensorShape(dg.image_size),
                          'output_moving': tf.TensorShape(dg.image_size),
                          'flow_params': tf.TensorShape(dg.flow_size),
                          'output_flow': tf.TensorShape(dg.flow_size)})
            
        data_type = tf.float32
        output_types = ({'input_moving': data_type,
                         'input_fixed': data_type,
                         'input_moving_seg': data_type},
                        {'output_fixed': data_type,
                         'output_fixed_seg': data_type,
                         'output_moving': data_type,
                         'flow_params': data_type,
                         'output_flow': data_type})

        return TensorFlowDataGenerator._prepare_generators(dg, batch_size,
                                                           output_shapes,
                                                           output_types,
                                                           max_buffer_size)



class TensorFlowVxmDataGenerator():
    
    @staticmethod
    def get_generators(batch_size: int, max_buffer_size: Optional[int] = None,
                       memory_cache: bool = False, disk_cache: bool = True,
                       dilation_radius: Optional[float] = None,
                       patient_case: Optional[str] = None, translation_alignment: bool = False,
                       data_type: Optional[str] = None) -> Tuple[tf.data.Dataset, VoxelmorphDataLoader]:
        """
        Get the generators for the Voxelmorph overlay data.

        Parameters
        ----------
        batch_size : int
            The batch size for the data generator.
        max_buffer_size : int, optional
            The maximum buffer size for the data generator. Default is None.
        memory_cache : bool, optional
            Whether to enable memory caching. Default is False.
        disk_cache : bool, optional
            Whether to enable disk caching. Default is True.
        dilation_radius : float, optional
            Value ignored. Default is None.
        patient_case : str, optional
            The patient case for the data generator. Default is None.
        translation_alignment : bool, optional
            Whether to enable translation alignment. Default is False.
        data_type : str, optional
            The data type for the data generator. Default is None.

        Returns
        -------
        train_generator : tf.data.Dataset
            The prepared training data generator.
        dg : VoxelmorphDataLoader
            The data loader object.

        """
        dg = VoxelmorphDataLoader(memory_cache=memory_cache,
                                  disk_cache=disk_cache,
                                  patient_case=patient_case,
                                  translation_alignment=translation_alignment,
                                  data_type=data_type)
        
        output_shapes = ({'vxm_source_input': tf.TensorShape(dg.image_size),
                          'vxm_target_input': tf.TensorShape(dg.image_size)},
                         {'vxm_transformer': tf.TensorShape(dg.image_size),
                          'vxm_flow': tf.TensorShape(dg.flow_size)})
        
        data_type = tf.float32
        output_types = ({'vxm_source_input': data_type,
                         'vxm_target_input': data_type,},
                        {'vxm_transformer': data_type,
                         'vxm_flow': data_type})


        return TensorFlowDataGenerator._prepare_generators(dg, batch_size,
                                                           output_shapes,
                                                           output_types,
                                                           max_buffer_size)
    
    
    
class TensorFlowVxmSegDataGenerator():
    
    @staticmethod
    def get_generators(batch_size: int, max_buffer_size: Optional[int] = None,
                       memory_cache: bool = False, disk_cache: bool = True,
                       dilation_radius: Optional[float] = None,
                       patient_case: Optional[str] = None, translation_alignment: bool = False,
                       data_type: Optional[str] = None) -> Tuple[tf.data.Dataset, VoxelmorphSegDataLoader]:
        """
        Get the generators for the Voxelmorph segmentation data.

        Parameters
        ----------
        batch_size : int
            The batch size for the data generator.
        max_buffer_size : int, optional
            The maximum buffer size for the data generator. Default is None.
        memory_cache : bool, optional
            Whether to use memory caching. Default is False.
        disk_cache : bool, optional
            Whether to use disk caching. Default is True.
        dilation_radius : float, optional
            The dilation radius of the contour. Default is None.
        patient_case : str, optional
            The patient case for the data generator. Default is None.
        translation_alignment : bool, optional
            Whether to perform translation alignment. Default is False.
        data_type : str, optional
            The data type for the data generator. Default is None.

        Returns
        -------
        train_generator : tf.data.Dataset
            The prepared training data generator.
        dg : VoxelmorphSegDataLoader
            The data loader object.

        """
        dg = VoxelmorphSegDataLoader(memory_cache=memory_cache,
                                    disk_cache=disk_cache,
                                    patient_case=patient_case,
                                    translation_alignment=translation_alignment,
                                    data_type=data_type)
        
        output_shapes = ({'vxm_dense_source_input': tf.TensorShape(dg.image_size),
                          'vxm_dense_target_input': tf.TensorShape(dg.image_size),
                          'vxm_source_seg': tf.TensorShape(dg.image_size)},
                         {'vxm_dense_transformer': tf.TensorShape(dg.image_size),
                          'vxm_seg_transformer': tf.TensorShape(dg.image_size),
                          'vxm_dense_flow': tf.TensorShape(dg.flow_size)})
        
        data_type = tf.float32
        output_types = ({'vxm_dense_source_input': data_type,
                         'vxm_dense_target_input': data_type,
                         'vxm_source_seg': data_type},
                        {'vxm_dense_transformer': data_type,
                         'vxm_seg_transformer': data_type,
                         'vxm_dense_flow': data_type})


        return TensorFlowDataGenerator._prepare_generators(dg, batch_size,
                                                           output_shapes,
                                                           output_types,
                                                           max_buffer_size)
