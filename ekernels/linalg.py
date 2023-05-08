import numpy as np
from scipy.linalg import blas


def symmetrify(A, upper=False):
    """
    Take the square matrix A and make it symmetrical by copting elements from
    the lower half to the upper

    works IN PLACE.

    note: tries to use cython, falls back to a slower numpy version
    """
    _symmetrify_numpy(A, upper)


def _symmetrify_numpy(A, upper=False):
    triu = np.triu_indices_from(A,k=1)
    if upper:
        A.T[triu] = A[triu]
    else:
        A[triu] = A.T[triu]


def tdot_blas(mat, out=None):
    """returns np.dot(mat, mat.T), but faster for large 2D arrays of doubles."""
    if (mat.dtype != 'float64') or (len(mat.shape) != 2):
        return np.dot(mat, mat.T)
    nn = mat.shape[0]
    if out is None:
        out = np.zeros((nn, nn))
    else:
        assert(out.dtype == 'float64')
        assert(out.shape == (nn, nn))
        # FIXME: should allow non-contiguous out, and copy output into it:
        assert(8 in out.strides)
        # zeroing needed because of dumb way I copy across triangular answer
        out[:] = 0.0

    # # Call to DSYRK from BLAS
    mat = np.asfortranarray(mat)
    out = blas.dsyrk(alpha=1.0, a=mat, beta=0.0, c=out, overwrite_c=1,
                     trans=0, lower=0)

    symmetrify(out, upper=True)
    return np.ascontiguousarray(out)


def diagview(A, offset=0):
    """
    Get a view on the diagonal elements of a 2D array.

    This is actually a view (!) on the diagonal of the array, so you can
    in-place adjust the view.

    :param :class:`ndarray` A: 2 dimensional numpy array
    :param int offset: view offset to give back (negative entries allowed)
    :rtype: :class:`ndarray` view of diag(A)
    """
    from numpy.lib.stride_tricks import as_strided
    assert A.ndim == 2, "only implemented for 2 dimensions"
    assert A.shape[0] == A.shape[1], "attempting to get the view of non-square matrix?!"
    if offset > 0:
        return as_strided(A[0, offset:], shape=(A.shape[0] - offset, ), strides=((A.shape[0]+1)*A.itemsize, ))
    elif offset < 0:
        return as_strided(A[-offset:, 0], shape=(A.shape[0] + offset, ), strides=((A.shape[0]+1)*A.itemsize, ))
    else:
        return as_strided(A, shape=(A.shape[0], ), strides=((A.shape[0]+1)*A.itemsize, ))
