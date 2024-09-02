from typing import Dict

import numpy as np

from tensorflow import keras

from tensorflow.python.platform import tf_logging as logging
from keras import backend
from tensorflow.python.keras.utils import tf_utils




class LoadBestWeightsOnNaN(keras.callbacks.Callback):
    """Callback that loades previous weights when a NaN loss is encountered or
    when the loss increases above a point."""

    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    
    def on_train_begin(self, logs: Dict = None):
        """
        Called at the beginning of training.

        Parameters
        ----------
        logs : Dict, optional
            Dictionary of logs. The default is None.

        Returns
        -------
        None.
        
        """
        self.previous_weights = None
        
        
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """
        Called at the end of each epoch. Checks whether the loss is NaN or
        infinite and if so, loads the previous weights. Also checks if the loss
        is too high and if so, loads the previous weights.
        The previous weights are stored at the end of each epoch.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        logs : Dict, optional
            Dictionary of logs. The default is None.

        Returns
        -------
        None.

        """
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None and self.previous_weights is not None:
            loss = tf_utils.sync_to_numpy_or_python_type(loss)
            if np.isnan(loss) or np.isinf(loss) or np.isclose(loss, 0.0, rtol=0, atol=1e-08):
                self.model.set_weights(self.previous_weights)
            elif epoch > 5 and loss > 1e4:
                self.model.set_weights(self.previous_weights)

        self.previous_weights = self.model.get_weights()
        


class IncreaseLROnPlateau(keras.callbacks.Callback):
    """Increase learning rate when a metric has stopped improving."""

    def __init__(
        self,
        monitor: str = 'val_loss',
        factor: float = 10.0,
        patience: int = 10,
        verbose: int = 0,
        mode: str = 'auto',
        min_delta: float = 1e-4,
        cooldown: int = 0,
        max_lr: float = 1.0,
        **kwargs,
    ):
        """
        Increase learning rate when a metric has stopped improving. The learning
        rate will be increased by a factor of `factor` when no improvement is
        seen for `patience` epochs.
                

        Parameters
        ----------
        monitor : str, optional
            Quantity to be monitored. The default is 'val_loss'.
        factor : float, optional
            Factor by which the learning rate will be increased. The default
            is 10.0. `new_lr = lr * factor`.
        patience : int, optional
            Number of epochs with no improvement after which learning rate will
            be reduced. The default is 10.
        verbose : int, optional
            0: quiet, 1: update messages.. The default is 0.
        mode : str, optional
            One of `{'auto', 'min', 'max'}`. In `'min'` mode, the learning rate
            will be reduced when the quantity monitored has stopped decreasing;
            in `'max'` mode it will be reduced when the quantity monitored has
            stopped increasing; in `'auto'` mode, the direction is automatically
            inferred from the name of the monitored quantity. The default is 'auto'.
        min_delta : float, optional
            Threshold for measuring the new optimum, to only focus on significant
            changes. The default is 1e-4.
        cooldown : int, optional
            Number of epochs to wait before resuming normal operation after lr
            has been increased.. The default is 0.
        max_lr : float, optional
            Upper bound on the learning rate.. The default is 1.0.

        Raises
        ------
        ValueError
            If factor smaller than 1.0.

        Returns
        -------
        None.

        """
        super().__init__()

        self.monitor = monitor
        if factor < 1.0:
            raise ValueError(
                f'IncreaseLROnPlateau does not support '
                f'a factor < 1.0. Got {factor}'
            )
        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')
            logging.warning(
                "`epsilon` argument is deprecated and "
                'will be removed, use `min_delta` instead.'
            )
        self.factor = factor
        self.max_lr = max_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0 
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """
        Resets wait counter and cooldown counter.

        Returns
        -------
        None.

        """
        if self.mode not in ['auto', 'min', 'max']:
            logging.warning(
                'Learning rate reduction mode %s is unknown, '
                'fallback to auto mode.',
                self.mode,
            )
            self.mode = 'auto'
        if self.mode == 'min' or (
            self.mode == 'auto' and 'acc' not in self.monitor
        ):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs: Dict = None):
        """
        Callback method called at the beginning of training.

        Parameters
        ----------
        logs : Dict, optional
            Dictionary of logs. The default is None.

        Returns
        -------
        None.

        """
        self._reset()

    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """
        Callback method called at the end of each epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        logs : Dict, optional
            Dictionary of logs. The default is None.

        Returns
        -------
        None.

        """
        logs = logs or {}
        logs['lr'] = backend.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            logging.warning(
                "Learning rate reduction is conditioned on metric `%s` "
                'which is not available. Available metrics are: %s',
                self.monitor,
                ','.join(list(logs.keys())),
            )

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = backend.get_value(self.model.optimizer.lr)
                    if old_lr < np.float32(self.max_lr):
                        new_lr = old_lr * self.factor
                        new_lr = min(new_lr, self.max_lr)
                        backend.set_value(self.model.optimizer.lr, new_lr)
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    def in_cooldown(self):
        """
        Check if the callback is in cooldown period.

        Returns
        -------
        is_cooldown : bool
            True if the callback is in cooldown period, False otherwise.

        """
        return self.cooldown_counter > 0