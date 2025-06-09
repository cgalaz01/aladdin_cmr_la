from typing import Optional, Tuple
from unittest.mock import patch

import tensorflow as tf

import voxelmorph as vxm


def _patch_upsample_block(x: tf.Tensor, connection: tf.Tensor, factor: int = 2,
                          name: Optional[str] = None) -> tf.Tensor:
    """
    Patched upsampling and concatenation block for U-Net architectures.
    Handles arbitrary input sizes by applying dynamic zero-padding to align shapes.
    
    Parameters
    ----------
    x : tf.Tensor
        The input tensor to be upsampled. Must be rank 3 to 5 (e.g., (B, H, W, C) or (B, D, H, W, C)).
    connection : tf.Tensor
        The skip-connection tensor to concatenate with after upsampling. Must be the same rank as `x`.
    factor : int, optional
        The upsampling factor for each spatial dimension. Default is 2.
    name : str, optional
        Base name for layers created inside this block. Layer names will be suffixed accordingly.
    
    Returns
    -------
    tf.Tensor
        A tensor resulting from upsampling `x`, zero-padding to match `connection`, and concatenating them.
    
    Raises
    ------
    AssertionError
        If the number of spatial dimensions inferred from `x` is not 1, 2, or 3.
    """
    ndims = len(x.get_shape()) - 2
    assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
    UpSampling = getattr(tf.keras.layers, 'UpSampling%dD' % ndims)

    size = (factor,) * ndims if ndims > 1 else factor
    upsampled = UpSampling(size=size, name=name)(x)
    pad_range = ((0, connection.shape[1] - upsampled.shape[1]),
                 (0, connection.shape[2] - upsampled.shape[2]),
                 (0, connection.shape[3] - upsampled.shape[3])) 
    upsampled = tf.keras.layers.ZeroPadding3D(padding=pad_range)(upsampled)
    name = name + '_concat' if name else None
    return tf.keras.layers.concatenate([upsampled, connection], name=name)


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
    with patch('voxelmorph.tf.networks._upsample_block', new=_patch_upsample_block):
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
    with patch('voxelmorph.tf.networks._upsample_block', new=_patch_upsample_block):
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
    model = get_model(image_shape)
    print(model.summary())