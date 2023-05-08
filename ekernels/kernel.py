"""interface for kernel functions"""

from typing import Any, Optional
from ekernels.linalg import tdot, diagview
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
    def _unscaled_dist(self, x, xp=None):
        if xp is None:
            Xsq = np.sum(np.square(x),1)
            r2 = -2.*tdot(x) + (Xsq[:,None] + Xsq[None,:])
            diagview(r2)[:,]= 0. # force diagnoal to be zero: sometime numerically a little negative
            r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2)
        else:
            #X2, = self._slice_X(X2)
            X1sq = np.sum(np.square(x),1)
            X2sq = np.sum(np.square(xp),1)
            r2 = -2.*np.dot(x, xp.T) + (X1sq[:,None] + X2sq[None,:])
            r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2)
        
    def _scaled_dist(self, x: np.ndarray, xp: Optional[np.ndarray] = None):
        if self.ard:
            if xp is not None:
                xp = xp / self.lengthscale
            return self._unscaled_dist(x/self.lengthscale, xp)
        else:
            return self._unscaled_dist(x, xp)/self.lengthscale
        
    def k(self, x: np.ndarray, xp: np.ndarray):
        r = self._scaled_dist(x, xp)
        return self.kr(r)
    
    def kr(self, r: float):
        raise NotImplementedError

    def fourier_kr(self, r: float):
        pass

    def dk_dr(self, r: float):
        pass


class ExplicitKernel(Kernel):
    """kernel with an explicit feature map"""
    def z(self, x: np.ndarray) -> np.ndarray:
        pass
