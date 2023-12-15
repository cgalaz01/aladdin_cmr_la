from typing import Tuple, Union

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from keras.engine.keras_tensor import KerasTensor


#########################################
# 3D Aladdin-S
#########################################


def downsample_layer_3d(x: KerasTensor, num_filters: int, activation: Union[str, layers.Activation],
                        kernel_initializer: str, index_str: str) -> Tuple[KerasTensor, KerasTensor]:
    """
    Downsample layer for 3D Aladdin-S.

    Parameters
    ----------
    x : KerasTensor
        Input tensor.
    num_filters : int
        Number of filters in the convolutional layer.
    activation : Union[str, layers.Activation]
        Activation function to be applied.
    kernel_initializer : str
        Initialization method for the kernel weights.
    index_str : str
        Index string used for naming the layers.

    Returns
    -------
    output : KerasTensor
        Downsampled tensor.
    output_skip : KerasTensor
        Tensor before downsampling, used for skip connections.

    """
    x = layers.Convolution3D(num_filters, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                             kernel_initializer=kernel_initializer, padding='same',
                             name='conv3d_' + index_str + '_1')(x)
    x = layers.Activation(activation, name='activation_' + index_str + '_2')(x)
    
    x_s = x
    
    x = layers.MaxPool3D(pool_size=(2, 2, 2), strides=None, padding='same',
                         name='maxpool_' + index_str + '_3')(x)
    
    return x, x_s
    

def upsample_layer_3d(x: KerasTensor, x_skip: KerasTensor, num_filters: int,
                      activation: Union[str, layers.Activation], kernel_initializer: str,
                      index_str: str) -> KerasTensor:
    """
    Upsample layer for 3D Aladdin-S.

    Parameters
    ----------
    x : KerasTensor
        Input tensor.
    x_skip : KerasTensor
        Skip connection tensor.
    num_filters : int
        Number of filters in the convolutional layer.
    activation : Union[str, layers.Activation]
        Activation function to be applied.
    kernel_initializer : str
        Initialization method for the kernel weights.
    index_str : str
        Index string used for naming the layers.

    Returns
    -------
    output : KerasTensor
        Upsampled tensor.

    """
    x = layers.Conv3DTranspose(num_filters, kernel_size=(3, 3, 3), strides=(2, 2, 2),
                               padding='same', kernel_initializer=kernel_initializer,
                               name='convtransp3d_' + index_str + '_1')(x)
    crop_range = ((0, x.shape[1] - x_skip.shape[1]),
                  (0, x.shape[2] - x_skip.shape[2]),
                  (0, x.shape[3] - x_skip.shape[3])) 
    x = layers.Cropping3D(cropping=crop_range, name='crop3d_' + index_str + '_2')(x)
    
    x = layers.Concatenate(axis=-1, name='concatenate3d_' + index_str + '_3')([x, x_skip])
    
    x = layers.Convolution3D(num_filters, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                             padding='same', kernel_initializer=kernel_initializer,
                             name='conv3d_' + index_str + '_6')(x)
    x = layers.Activation(activation, name='activation_' + index_str + '_7')(x)
    
    return x


def latent_layer_3d(x: KerasTensor, num_filters: int, activation: Union[str, layers.Activation],
                    kernel_initializer: str, index_str: str) -> KerasTensor:  
    """
    Latent layer for 3D Aladdin-S.

    Parameters
    ----------
    x : KerasTensor
        Input tensor.
    num_filters : int
        Number of filters in the convolutional layer.
    activation : Union[str, layers.Activation]
        Activation function to be applied.
    kernel_initializer : str
        Initialization method for the kernel weights.
    index_str : str
        Index string used for naming the layers.

    Returns
    -------
    output : KerasTensor
        Latent tensor.

    """
    x = layers.Convolution3D(num_filters, kernel_size=(1, 1, 1), strides=(1, 1, 1),
                              kernel_initializer=kernel_initializer, padding='same',
                              name='latent_conv3d_' + index_str + '_1')(x)
    x = layers.Activation(activation, name='latent_activation_' + index_str + '_2')(x)
    
    return x
    

def skip_layer_3d(x: KerasTensor, num_filters: int, ratio: int, kernel_initializer: str,
                  activation: Union[str, layers.Activation], index_str: str) -> KerasTensor:
    """
    Skip layer for 3D Aladdin-S.

    Parameters
    ----------
    x : KerasTensor
        Input tensor.
    num_filters : int
        Number of filters in the convolutional layer.
    ratio : int
        Ratio used in the Squeeze and Excitation block.
    kernel_initializer : str
        Initialization method for the kernel weights.
    activation : Union[str, layers.Activation]
        Activation function to be applied.
    index_str : str
        Index string used for naming the layers.

    Returns
    -------
    output : KerasTensor
        Output tensor after applying the skip layer.

    """
    # Squeeze and Excitation block
    se_x = layers.GlobalAveragePooling3D(name='skip_squeeze_excite_globalaveragepooling3d_1_' + index_str)(x)
    se_x = layers.Dense(num_filters // ratio, use_bias=False, kernel_initializer=kernel_initializer,
                        name='skip_squeeze_excite_dense_2_' + index_str)(se_x)
    se_x = layers.Activation(activation, name='skip_squeeze_excite_activation_3_' + index_str)(se_x)
    se_x = layers.Dense(num_filters, use_bias=False, kernel_initializer=kernel_initializer,
                        name='skip_squeeze_excite_dense_4_' + index_str)(se_x)
    se_x = layers.Activation('sigmoid', name='skip_squeeze_excite_activation_5_' + index_str)(se_x)
    se_x = layers.Reshape((1, 1, 1, num_filters), name='skip_squeeze_excite_reshape_6_' + index_str)(se_x)
    x = layers.Multiply(name='skip_squeeze_excite_multiply_7_' + index_str)([x, se_x])
    
    return x
    

def get_model_3d(input_shape: Tuple[int]) -> keras.Model:
    """
    Get the 3D Aladdin-S model for segmentation.

    Parameters
    ----------
    input_shape : Tuple[int]
        The shape of the input images.

    Returns
    -------
    model : keras.Model
        The 3D Aladdin-S model for image segmentation.

    """
    num_layers = 3
    num_filters_seq = [32, 64, 128]
    activation = tf.keras.layers.LeakyReLU(alpha=0.3)
    kernel_initializer = 'he_normal'
    
    input_img = keras.Input(shape=input_shape, name='input_img')   
    
    x = input_img
    
    x_skip = []
    # Encoder
    for i_e in range(num_layers):
        x, x_s = downsample_layer_3d(x, num_filters_seq[i_e], activation,
                                     kernel_initializer, str(i_e + 1))
        x_skip.append(x_s)
    
    # Latent space
    x = latent_layer_3d(x, num_filters_seq[-1], activation, kernel_initializer,
                        str(num_layers + 1))
    
    # Skip layer
    for i in range(len(x_skip)):
        x_skip[i] = skip_layer_3d(x_skip[i], num_filters_seq[i], 8, 'he_normal',
                                  activation, str(i+1))
    
    x_skip.reverse()
    num_filters_seq.reverse()
    # Decoder
    for i_d in range(num_layers):
        x = upsample_layer_3d(x, x_skip[i_d], num_filters_seq[i_d], activation,
                              kernel_initializer, str(i_d + num_layers + 2))
        
    
    output_seg = layers.Conv3D(1, kernel_size=(3, 3, 3), strides=1, padding='same',
                               kernel_initializer=kernel_initializer,
                               activation='sigmoid',
                               use_bias=True, name='output_seg')(x)
    

    model = keras.Model([input_img], [output_seg])
    
    
    return model


if __name__ == '__main__':
    keras_model = get_model_3d((96, 96, 36, 1))
    print(keras_model.summary())
    