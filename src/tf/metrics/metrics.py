from typing import Callable, Optional

import tensorflow as tf
from tensorflow.keras import backend as K


@tf.autograph.experimental.do_not_convert
def dice_score(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate the dice score for binary segmentation.

    Parameters
    ----------
    y_true : tf.Tensor
        The ground truth segmentation mask.
    y_pred : tf.Tensor
        The predicted segmentation mask.

    Returns
    -------
    dice_coef : tf.Tensor
        The dice score.

    """
    epsilon = 1e-7
    
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred >= 0.5, tf.float32)
    
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    dice_coef = (2. * intersection + epsilon) / (K.sum(y_true) + K.sum(y_pred) + epsilon)

    return dice_coef


@tf.autograph.experimental.do_not_convert
def soft_dice_score(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate the soft dice score for binary segmentation.

    Parameters
    ----------
    y_true : tf.Tensor
        The ground truth segmentation mask.
    y_pred : tf.Tensor
        The predicted segmentation mask.

    Returns
    -------
    dice_coef : tf.Tensor
        The soft dice score.

    """
    epsilon = 1e-7
    
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)            
    
    # Expected y_pred to be 'logits'
    #y_pred = tf.sigmoid(y_pred)
    
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    dice_coef = (2. * intersection + epsilon) / (K.sum(y_true) + K.sum(y_pred) + epsilon)

    return dice_coef


@tf.autograph.experimental.do_not_convert
def soft_dice_class_score(index: Optional[int] = None) -> Callable:
    """
    Calculate the soft dice score for multi-class segmentation.

    Parameters
    ----------
    index : Optional[int], optional
        The index of the class to calculate the score for. If None, the score
        is calculated for all classes.

    Returns
    -------
    fn : Callable
        A callable function that calculates the soft dice score for the specified
        class index.

    """
    def soft_dice_index_score(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculate the soft dice score for a specific class.

        Parameters
        ----------
        y_true : tf.Tensor
            The ground truth segmentation mask.
        y_pred : tf.Tensor
            The predicted segmentation mask.

        Returns
        -------
        dice_coef : tf.Tensor
            The soft dice score for the specified class.

        """
        if index:
            y_true = y_true[..., index]
            y_pred = y_pred[..., index]
        else:
            # Remove background
            y_true = y_true[..., 1:]
            y_pred = y_pred[..., 1:]
        
        return soft_dice_score(y_true, y_pred)
    
    if index:
        soft_dice_index_score.__name__ = 'soft_dice_class_{}'.format(index)
    else:
        soft_dice_index_score.__name__ = 'soft_dice_class'
        
    return soft_dice_index_score