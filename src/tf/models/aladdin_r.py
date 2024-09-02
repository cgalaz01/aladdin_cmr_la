from typing import Tuple, Union

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import neurite as ne
import voxelmorph as vxm

#########################################
# 3D Aladdin-R
#########################################


def downsample_layer_3d(x, num_filters: int, activation: Union[str, layers.Activation],
                        kernel_initializer: str, index_str: str):
    """
    Downsample layer for 3D Aladdin-R.

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

    

def upsample_layer_3d(x, x_skip, num_filters: int,
                      activation: Union[str, layers.Activation], kernel_initializer: str,
                      index_str: str):
    """
    Upsample layer for 3D Aladdin-R.

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


def latent_layer_3d(x, num_filters: int, activation: Union[str, layers.Activation],
                    kernel_initializer: str, index_str: str):  
    """
    Latent layer for 3D Aladdin-R.

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
    

def skip_layer_3d(x, num_filters: int, ratio: int, kernel_initializer: str,
                  activation: Union[str, layers.Activation], index_str: str):
    """
    Skip layer for 3D Aladdin-R.

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


def get_flow_negate() -> tf.keras.layers.Layer:
    """
    Get the flow negation layer.

    Returns
    -------
    flow_negation : tf.keras.layers.Layer
        The flow negation layer.

    """
    flow_negation = ne.layers.Negate(name='flow_negate')
    
    return flow_negation

    
def get_vector_integration(int_steps: int = 7) -> tf.keras.layers.Layer:
    """
    Get the vector integration layer.

    Parameters
    ----------
    int_steps : int, optional
        The number of integration steps. Default is 7.

    Returns
    -------
    vector_integration : tf.keras.layers.Layer
        The vector integration layer.

    """
    vector_integration = vxm.layers.VecInt(
        indexing='ij',
        method='quadrature',
        int_steps=int_steps,
        out_time_pt=1,
        name='output_flow')

    return vector_integration


def get_spatial_transformer(name: str = 'output_fixed') -> tf.keras.layers.Layer:
    """
    Get the spatial transformer layer.

    Parameters
    ----------
    name : str, optional
        The name of the layer. Default is 'output_fixed'.

    Returns
    -------
    spatial_transformer : tf.keras.layers.Layer
        The spatial transformer layer.

    """
    spatial_transformer = vxm.layers.SpatialTransformer(interp_method='linear',
                                                 indexing='ij',
                                                 shift_center=False,
                                                 single_transform=False,
                                                 fill_value=None,
                                                 name=name)
    
    return spatial_transformer
    

def get_model_3d(input_shape: Tuple[int]) -> keras.Model:
    """
    Get the 3D Aladdin-R model for image registation.

    Parameters
    ----------
    input_shape : Tuple[int]
        The shape of the input images.

    Returns
    -------
    model : keras.Model
        The 3D Aladdin-R model for image registation.

    """
    num_layers = 3
    num_filters_seq = [32, 64, 128]
    activation = tf.keras.layers.LeakyReLU(alpha=0.3)
    kernel_initializer = 'he_normal'
    
    
    input_moving = keras.Input(shape=input_shape, name='input_moving')
    input_fixed = keras.Input(shape=input_shape, name='input_fixed')
    input_moving_seg = keras.Input(shape=input_shape, name='input_moving_seg')    
    
    x_moving = input_moving
    x_fixed = input_fixed
    x_moving_seg = input_moving_seg
    
    x = layers.Concatenate(axis=-1)([x_moving, x_fixed])
    
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
        
    
    flow_params = layers.Conv3D(3, kernel_size=(3, 3, 3), strides=1, padding='same',
                                kernel_initializer=kernel_initializer,
                                use_bias=True, name='flow_params')(x)
    
    
    # integrate to produce diffeomorphic warp (i.e. treat flow as a stationary velocity field)
    vector_integration = get_vector_integration(int_steps=7)
    output_flow = vector_integration(flow_params)

    
    # Obtain the model inputs and outputs
    model_input = [input_moving, input_fixed, input_moving_seg]
              
    spatial_transformer = get_spatial_transformer(name='output_fixed')
    output_fixed = spatial_transformer([x_moving, output_flow])
    spatial_transformer = get_spatial_transformer(name='output_fixed_seg')
    output_fixed_seg = spatial_transformer([x_moving_seg, output_flow])

    flow_negation = get_flow_negate()
    output_neg_flow = flow_negation(output_flow)
    
    spatial_transformer = get_spatial_transformer(name='output_moving')
    output_moving = spatial_transformer([x_fixed, output_neg_flow])
    
    model_output = [output_fixed, output_moving, output_fixed_seg, flow_params, output_flow]
    
    model = keras.Model(model_input, model_output)
    
    return model


if __name__ == '__main__':
    keras_model = get_model_3d((96, 96, 36, 1))
    print(keras_model.summary())
    