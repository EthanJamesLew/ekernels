"""radial basis function kernels"""
from ekernels.kernel import ShiftInvariantKernel
import numpy as np

class RBF(ShiftInvariantKernel):
    """Radial Basis Function Kernel"""
    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
        self.lengthscale = lengthscale
        self.variance = variance

    def kr(self, r: float):
        """radial basis function kernel as a function of r=x-x'"""
        return self.variance * np.exp(-0.5 * r**2 / self.lengthscale**2)
