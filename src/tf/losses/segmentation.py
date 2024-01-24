import tensorflow as tf



class SoftDiceLoss(tf.keras.losses.Loss):
    """Implements the Dice loss for classification problems.
    """

    def __init__(self,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name=None):
        """Initializes `DiceLoss`.
        Args:
          reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used with
            `tf.distribute.Strategy`, outside of built-in training loops such as
            `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
            will raise an error. Please see this custom training [tutorial](
              https://www.tensorflow.org/tutorials/distribute/custom_training) for
                more details.
          name: Optional name for the op.
        """
        super(SoftDiceLoss, self).__init__(reduction=reduction, name=name)
  
  
    def call(self, y_true, y_pred):
        """Invokes the `DiceLoss`.
        Args:
          y_true: A tensor of size [batch, ..., num_classes]
          y_pred: A tensor of size [batch, ..., num_classes]
        Returns:
          Summed loss float `Tensor`.
        """
        with tf.name_scope('dice_loss'):
            epsilon = 1e-7
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            # Remove background - expects to be first one-hot encoding
            remove_background = False
            if remove_background:
                y_true = y_true[..., 1:]
                y_pred = y_pred[..., 1:]
            
            #y_pred = tf.sigmoid(y_pred)
            
            # Reduce along all axis except batch and class axis
            dimensions = len(y_true.shape)
            reduction_axis = list(range(1, dimensions - 1))
            
            numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=reduction_axis) + epsilon
            denominator = tf.reduce_sum(y_true + y_pred, axis=reduction_axis) + epsilon
        
            dice_score = numerator / denominator
            mean_class_dice_score = tf.math.reduce_mean(dice_score, axis=-1)
            loss = 1 - mean_class_dice_score
        
        return loss
  
  
    def get_config(self):
        base_config = super(SoftDiceLoss, self).get_config()
        return base_config
