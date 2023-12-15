from typing import Tuple

import tensorflow as tf

import voxelmorph as vxm


def get_model(image_shape: Tuple[int]) -> tf.keras.Model:
    """
    Get the VoxelMorph model for image registration.

    Parameters
    ----------
    image_shape : Tuple[int]
        The shape of the input image.

    Returns
    -------
    voxelmorph_model : tf.keras.Model
        The VoxelMorph model for image registration.

    """
    nb_unet_features = [[32, 64, 128], [256, 128, 64, 32]]
    voxelmorph_model = vxm.networks.VxmDense(inshape=image_shape,
                                             nb_unet_features=nb_unet_features,
                                             nb_unet_levels=None,
                                             unet_feat_mult=1,
                                             int_resolution=1,
                                             int_steps=0,
                                             reg_field='warp',
                                             name='vxm')
    
    return voxelmorph_model



def get_model_seg(image_shape: Tuple[int]) -> tf.keras.Model:
    """
    Get the VoxelMorph model that combines the segmentation map for image registration.

    Parameters
    ----------
    image_shape : Tuple[int]
        The shape of the input image.

    Returns
    -------
    voxelmorph_model : tf.keras.Model
        The VoxelMorph-Seg model for image registration.

    """
    nb_unet_features = [[32, 64, 128], [256, 128, 64, 32]]
    voxelmorph_model = vxm.networks.VxmDenseSemiSupervisedSeg(inshape=image_shape,
                                                              nb_labels=1,
                                                              nb_unet_features=nb_unet_features,
                                                              int_steps=0,
                                                              int_resolution=1,
                                                              seg_resolution=1,
                                                              reg_field='warp',
                                                              name='vxm')
    
    return voxelmorph_model


if __name__ == '__main__':
    tf.keras.backend.clear_session()
    image_shape = (96, 96, 36)
    model = get_model_seg(image_shape)
    print(model.summary())