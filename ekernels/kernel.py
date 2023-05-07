"""interface for kernel functions"""

from typing import Any
import numpy as np


class Kernel:
    """kernel function with interfaces to evaluate 
    a kernel function, its hyperparameters, and (possibly)
    its gradient"""
    def k(self, x: np.ndarray, xp: np.ndarray):
        """
        compute the kernel function
        """
        pass

    def kdiag(self, x: np.ndarray):
        """
        The diagonal of the kernel matrix
        """
        pass

    def gradients_x(self, dl_dk, x, xp):
        """
        kernel gradients
        """
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


class ShiftInvariantKernel(Kernel):
    """shift invariant kernel function
    optionally has the n-dim fourier transform"""
    def kr(self, r: float):
        pass

    def fourier_kr(self, r: float):
        pass

    def dk_dr(self, r: float):
        pass


class ExplicitKernel(Kernel):
    """kernel with an explicit feature map"""
    def z(self, x: np.ndarray) -> np.ndarray:
        pass
