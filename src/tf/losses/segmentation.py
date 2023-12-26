import tensorflow as tf


# Loss taken from here:
#    https://github.com/tensorflow/models/blob/master/official/vision/keras_cv/losses/focal_loss.py
class FocalLoss(tf.keras.losses.Loss):
  """Implements a Focal loss for classification problems.
  Reference:
    [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).
  """

  def __init__(self,
               alpha,
               gamma,
               reduction=tf.keras.losses.Reduction.AUTO,
               name=None):
    """Initializes `FocalLoss`.
    Args:
      alpha: The `alpha` weight factor for binary class imbalance.
      gamma: The `gamma` focusing parameter to re-weight loss.
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the op. Defaults to 'retinanet_class_loss'.
    """
    self._alpha = alpha
    self._gamma = gamma
    super(FocalLoss, self).__init__(reduction=reduction, name=name)


  def call(self, y_true, y_pred):
    """Invokes the `FocalLoss`.
    Args:
      y_true: A tensor of size [batch, num_anchors, num_classes]
      y_pred: A tensor of size [batch, num_anchors, num_classes]
    Returns:
      Summed loss float `Tensor`.
    """
    with tf.name_scope('focal_loss'):
      y_true = tf.cast(y_true, dtype=tf.float32)
      y_pred = tf.cast(y_pred, dtype=tf.float32)
      #y_true = y_true[..., 1:]
      #y_pred = y_pred[..., 1:]
      positive_label_mask = tf.equal(y_true, 1.0)
      cross_entropy = (
          tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
      probs = tf.math.sigmoid(y_pred)
      probs_gt = tf.where(positive_label_mask, probs, 1.0 - probs)
      # With small gamma, the implementation could produce NaN during back prop.
      modulator = tf.pow(1.0 - probs_gt, self._gamma)
      loss = modulator * cross_entropy
      weighted_loss = tf.where(positive_label_mask, self._alpha * loss,
                               (1.0 - self._alpha) * loss)

    return weighted_loss


  def get_config(self):
    config = {
        'alpha': self._alpha,
        'gamma': self._gamma,
    }
    base_config = super(FocalLoss, self).get_config()
    return dict(list(base_config.items()) + list(config.items())) 



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
    

    
class SparseSoftDiceLoss(tf.keras.losses.Loss):
    """Implements the Sparse Dice loss for classification problems.
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
        super(SparseSoftDiceLoss, self).__init__(reduction=reduction, name=name)
  
  
    def call(self, y_true, y_pred):
        """Invokes the `DiceLoss`.
        Args:
          y_true: A tensor of size [batch, ..., num_classes]
          y_pred: A tensor of size [batch, ..., num_classes]
        Returns:
          Summed loss float `Tensor`.
        """
        with tf.name_scope('sparse_dice_loss'):
            epsilon = 1e-7
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            
            # Flatten inputs
            y_true = tf.keras.layers.Flatten()(y_true) #tf.reshape(y_true, [tf.shape(y_true)[0], -1])
            y_pred = tf.keras.layers.Flatten()(y_pred) #tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
            
            # Get unique classes in the tensors per batch
            #unique_labels, _ = tf.map_fn(tf.unique, y_true, infer_shape=True)
            unique_labels, _ = tf.unique(tf.reshape(y_true, [-1])) # TODO: Make it per batch
            
            class_dice_scores = []

            class_dice_scores = tf.map_fn(fn=lambda label: self.dice_score_func(y_true, y_pred, label), elems=unique_labels)
            # Transpose so that batch axis is in axis position 0
            class_dice_scores = tf.transpose(class_dice_scores)
            
            mean_class_dice_score = tf.math.reduce_mean(class_dice_scores, axis=-1)
            loss = 1 - mean_class_dice_score
        
        return loss
  
  
    def get_config(self):
        base_config = super(SparseSoftDiceLoss, self).get_config()
        return base_config

    
    @tf.function
    def dice_score_func(self, y_true, y_pred, unique_label):
        y_true_label = tf.cond(unique_label == 0.0,
                               lambda: tf.where(y_true == unique_label, y_true, 0.0),
                               lambda: tf.where(y_true == unique_label, y_true / unique_label, 0.0))
        y_pred_label = tf.cond(unique_label == 0.0,
                               lambda: tf.where(y_true == unique_label, y_pred, 0.0),
                               lambda: tf.where(y_true == unique_label, y_pred / unique_label, 0.0))
        
        #y_pred = tf.sigmoid(y_pred)
        epsilon = 1e-7
        numerator = 2 * tf.reduce_sum(y_true_label * y_pred_label, axis=-1) + epsilon
        denominator = tf.reduce_sum(y_true_label + y_pred_label, axis=-1) + epsilon
        dice_score = numerator / denominator
        
        return dice_score


    
def combined_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
            
    focal_loss = FocalLoss(alpha=0.25,
                           gamma=2.0,
                           reduction=tf.keras.losses.Reduction.AUTO)
    dice_loss = SoftDiceLoss(reduction=tf.keras.losses.Reduction.AUTO)
    
    total_loss = 0.1 * focal_loss(y_true, y_pred) + 0.9 * dice_loss(y_true, y_pred)
    
    return total_loss


if __name__ == '__main__':
    
    a = tf.constant([[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]])
    b = tf.constant([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]])
    
    loss =SparseSoftDiceLoss()
    print(loss(a, b))