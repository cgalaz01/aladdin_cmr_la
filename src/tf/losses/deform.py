# Adapted from DeepReg: https://github.com/DeepRegNet/DeepReg/blob/main/deepreg/loss/deform.py
"""Provide regularization functions and classes for ddf."""
from typing import Callable, Dict

import tensorflow as tf



def gradient_dx(fx: tf.Tensor) -> tf.Tensor:
    """
    Calculate gradients on x-axis of a 3D tensor using central finite difference.
    It moves the tensor along axis 1 to calculate the approximate gradient, the x axis,
    dx[i] = (x[i+1] - x[i-1]) / 2.

    Parameters
    ----------
    fx : tf.Tensor
        Tensor with a shape of (batch, m_dim1, m_dim2, m_dim3).

    Returns
    -------
    gradient_x : tf.Tensor
        Tensor representing the gradients along the x-axis.

    """
    return (fx[:, 2:, 1:-1, 1:-1] - fx[:, :-2, 1:-1, 1:-1]) / 2


def gradient_dy(fy: tf.Tensor) -> tf.Tensor:
    """
    Calculate gradients on y-axis of a 3D tensor using central finite difference.
    It moves the tensor along axis 2 to calculate the approximate gradient, the y axis,
    dy[i] = (y[i+1] - y[i-1]) / 2.

    Parameters
    ----------
    fy : tf.Tensor
        Tensor with a shape of (batch, m_dim1, m_dim2, m_dim3).

    Returns
    -------
    gradient_y : tf.Tensor
        Tensor representing the gradients along the y-axis.

    """
    return (fy[:, 1:-1, 2:, 1:-1] - fy[:, 1:-1, :-2, 1:-1]) / 2


def gradient_dz(fz: tf.Tensor) -> tf.Tensor:
    """
    Calculate gradients on z-axis of a 3D tensor using central finite difference.
    It moves the tensor along axis 3 to calculate the approximate gradient, the z axis,
    dz[i] = (z[i+1] - z[i-1]) / 2.

    Parameters
    ----------
    fz : tf.Tensor
        Tensor with a shape of (batch, m_dim1, m_dim2, m_dim3).

    Returns
    -------
    gradient_z : tf.Tensor
        Tensor representing the gradients along the z-axis.

    """
    return (fz[:, 1:-1, 1:-1, 2:] - fz[:, 1:-1, 1:-1, :-2]) / 2


def gradient_dxyz(fxyz: tf.Tensor, fn: Callable) -> tf.Tensor:
    """
    Calculate gradients on x,y,z-axis of a tensor using central finite difference.
    The gradients are calculated along x, y, z separately then stacked together.

    Parameters
    ----------
    fxyz : tf.Tensor
        A tensor that has 3 channels on the last axis.
    fn : Callable
        Function to call over each channels.

    Returns
    -------
    gradient_xyz : tf.Tensor
        Tensor representing the gradients along the x-, y- and z-axis.

    """
    return tf.stack([fn(fxyz[..., i]) for i in [0, 1, 2]], axis=4)



class GradientNorm(tf.keras.losses.Loss):
    """
    Calculate the L1/L2 norm of ddf using central finite difference.
    y_true and y_pred have to be at least 5d tensor, including batch axis.
    """

    def __init__(self, reduction = tf.keras.losses.Reduction.AUTO,
                 l1: bool = False, name: str = 'GradientNorm', **kwargs: Dict):
        """
        Initialize the GradientNorm loss.

        Parameters
        ----------
        reduction : tf.keras.losses.Reduction, optional
            Specifies the type of reduction to apply to the loss. The default is tf.keras.losses.Reduction.AUTO.
        l1 : bool, optional
            If True, calculates the L1 norm. If False, calculates the L2 norm. The default is False.
        name : str, optional
            Name of the loss. The default is 'GradientNorm'.
        **kwargs : Dict
            Additional arguments.

        Returns
        -------
        None.

        """
        super().__init__(reduction=reduction, name=name)
        self.l1 = l1

    def call(self, _: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculate the L1/L2 norm of ddf using central finite difference.
        The gradients are calculated along x, y, z separately then stacked together.

        Parameters
        ----------
        _ : tf.Tensor
            y_true tensors (ignored).
        y_pred : tf.Tensor
            The predicted results tensor with a shape of (batch, m_dim1, m_dim2, m_dim3, 3).

        Returns
        -------
        gradient_norm : tf.Tensor
            Returns the L1/L2 norm of ddf with a shape of (batch,).

        """
        assert len(y_pred.shape) == 5
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        ddf = y_pred
        # first order gradient
        # (batch, m_dim1-2, m_dim2-2, m_dim3-2, 3)
        dfdx = gradient_dxyz(ddf, gradient_dx)
        dfdy = gradient_dxyz(ddf, gradient_dy)
        dfdz = gradient_dxyz(ddf, gradient_dz)
        if self.l1:
            norms = tf.abs(dfdx) + tf.abs(dfdy) + tf.abs(dfdz)
        else:
            norms = dfdx ** 2 + dfdy ** 2 + dfdz ** 2
        return tf.reduce_mean(norms, axis=[1, 2, 3, 4])

    def get_config(self) -> Dict:
        """
        Return the config dictionary for recreating this class.

        Returns
        -------
        config : Dict
            The class configuration dictionary.

        """
        config = super().get_config()
        config['l1'] = self.l1
        return config



class BendingEnergy(tf.keras.losses.Loss):
    """
    Calculate the bending energy of ddf using central finite difference.
    y_true and y_pred have to be at least 5d tensor, including batch axis.
    """

    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO,
                 name: str = 'BendingEnergy', **kwargs):
        """
        Initialize the BendingEnergy loss.

        Parameters
        ----------
        reduction : tf.keras.losses.Reduction, optional
            The reduction method to apply to the loss. The default is tf.keras.losses.Reduction.AUTO.
        name : str, optional
            The name of the loss. The default is 'BendingEnergy'.
        **kwargs : Dict
            Additional arguments.

        Returns
        -------
        None.

        """
        super().__init__(reduction=reduction, name=name)

    
    def call(self, _: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return the bending energy loss.

        Parameters
        ----------
        _ : tf.Tensor
            y_true tensors (ignored).
        y_pred : tf.Tensor
            The predicted results tensor with a shape of (batch, m_dim1, m_dim2, m_dim3, 3).

        Returns
        -------
        loss: tf.Tensor
            Returns the bending energy loss with a shape of (batch,).

        """
        assert len(y_pred.shape) == 5
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        ddf = y_pred
        # first order gradient
        # (batch, m_dim1-2, m_dim2-2, m_dim3-2, 3)
        dfdx = gradient_dxyz(ddf, gradient_dx)
        dfdy = gradient_dxyz(ddf, gradient_dy)
        dfdz = gradient_dxyz(ddf, gradient_dz)

        # second order gradient
        # (batch, m_dim1-4, m_dim2-4, m_dim3-4, 3)
        dfdxx = gradient_dxyz(dfdx, gradient_dx)
        dfdyy = gradient_dxyz(dfdy, gradient_dy)
        dfdzz = gradient_dxyz(dfdz, gradient_dz)
        dfdxy = gradient_dxyz(dfdx, gradient_dy)
        dfdyz = gradient_dxyz(dfdy, gradient_dz)
        dfdxz = gradient_dxyz(dfdx, gradient_dz)

        # (dx + dy + dz) ** 2 = dxx + dyy + dzz + 2*(dxy + dyz + dzx)
        energy = dfdxx ** 2 + dfdyy ** 2 + dfdzz ** 2
        energy += 2 * dfdxy ** 2 + 2 * dfdxz ** 2 + 2 * dfdyz ** 2
        return tf.reduce_mean(energy, axis=[1, 2, 3, 4])
    
    
def multi_displacement_loss(bending_energy: float, gradient_l1: float, gradient_l2: float,
                            l1: float, l2: float) -> Callable:
    """
    Create a callable function that calculates the total displacement loss.

    Parameters
    ----------
    bending_energy : float
        Weight for the bending energy regularization term.
    gradient_l1 : float
        Weight for the L1 norm of the gradient regularization term.
    gradient_l2 : float
        Weight for the L2 norm of the gradient regularization term.
    l1 : float
        Weight for the L1 regularization term.
    l2 : float
        Weight for the L2 regularization term.

    Returns
    -------
    displacement_losses : Callable
        A callable function that calculates the total displacement loss.

    """
    def displacement_losses(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculate the total displacement loss.

        Parameters
        ----------
        y_true : tf.Tensor
            The true target tensor.
        y_pred : tf.Tensor
            The predicted tensor.

        Returns
        -------
        total_loss : tf.Tensor
            The total displacement loss.

        """
        total_loss = 0
        
        if bending_energy > 0:
            reg = BendingEnergy()
            total_loss += (bending_energy * reg(y_true, y_pred))
            
        if gradient_l1 > 0:
            reg = GradientNorm(l1=True, name='GradientNormL1')
            total_loss += (gradient_l1 * reg(y_true, y_pred))
            
        if gradient_l2 > 0:
            reg = GradientNorm(l1=False, name='GradientNormL2')
            total_loss += (gradient_l2 * reg(y_true, y_pred))
            
        if l1 > 0:
            reg = tf.keras.regularizers.L1(l1)
            total_loss += reg(y_pred)
            
        if l2 > 0:
            reg = tf.keras.regularizers.L2(l2)
            total_loss += reg(y_pred)
            
        return total_loss
    
    return displacement_losses

    