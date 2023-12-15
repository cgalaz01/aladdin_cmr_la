import tensorflow as tf


def l1_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate the L1 loss between the true and predicted values.

    Parameters
    ----------
    y_true : tf.Tensor
        The true values.
    y_pred : tf.Tensor
        The predicted values.

    Returns
    -------
    l1_loss : tf.Tensor
        The L1 loss.

    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    y_diff = y_pred - y_true
    l1_loss = tf.reduce_sum(tf.abs(y_diff))
    
    return l1_loss


def l2_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate the L2 loss between the true and predicted values.

    Parameters
    ----------
    y_true : tf.Tensor
        The true values.
    y_pred : tf.Tensor
        The predicted values.

    Returns
    -------
    l2_loss : tf.Tensor
        The L2 loss.

    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    y_diff = y_pred - y_true
    l2_loss = tf.reduce_sum(tf.square(y_diff))
    
    return l2_loss


def l1_l2_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate the L1 + L2 loss between the true and predicted values.

    Parameters
    ----------
    y_true : tf.Tensor
        The true values.
    y_pred : tf.Tensor
        The predicted values.

    Returns
    -------
    l1_l2_loss : tf.Tensor
        The L1 + L2 loss.

    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    y_diff = y_pred - y_true
    l1_loss = tf.reduce_sum(tf.abs(y_diff))
    l2_loss = tf.reduce_sum(tf.square(y_diff))    
    
    return l1_loss + l2_loss


    