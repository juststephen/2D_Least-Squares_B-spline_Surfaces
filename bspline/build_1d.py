"""
Original Matlab files by Dr. Amiri Simkooei, Alireza.

Translated into Python by Stephen.
"""

import numpy as np
from scipy.interpolate import BSpline, PPoly

def build_bspline(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Build bspline B matrix for 1D.

    Parameters
    ----------
    x : np.ndarray
        Coords.
    k : np.ndarray
        Knots.

    Returns
    -------
    B : np.ndarray
        1D bspline B matrix.
    """
    m = len(x)
    k_len = len(k)
    c = BSpline.basis_element(k)
    C = PPoly.from_spline(c.tck).c.T[k_len-2:-k_len+2]
    n = C.shape[0]
    B = np.zeros((n, m))

    for i in range(n):
        dx = (x - k[i])
        b = 0
        for j in range(n):
            b += C[i, j] * dx ** ( n - j - 1 )
        B[i] = b.T

    B0 = []
    ix = []
    for i in range(n):
        idx, *_ = np.where(
            ( x >= k[i] ) &
            ( x <= k[-1] if i == n - 1 else x < k[i+1] )
        )
        B0 = [*B0, *B[i, idx]]
        ix.extend(idx)

    B = np.array(B0)[np.argsort(ix)]
    B = B.reshape((m, 1))

    if max(B.shape) != max(B.shape):
        raise(Exception('Dimension problem'))

    return B
