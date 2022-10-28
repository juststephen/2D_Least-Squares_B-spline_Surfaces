"""
Original Matlab files by Dr. Amiri Simkooei, Alireza.

Translated into Python by Stephen.
"""

import numpy as np

from .build_1d import build_bspline

def build_bspline_A_2D(
    x: np.ndarray,
    y: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    degx: int,
    degy: int
) -> np.ndarray:
    """
    Build bspline A matrix for 2D.

    Parameters
    ----------
    x : np.ndarray
        X-coords.
    y : np.ndarray
        Y-coords.
    kx : np.ndarray
        X-knots.
    ky : np.ndarray
        Y-knots.
    degx : np.ndarray
        Degree in X.
    degy : np.ndarray
        Degree in Y.

    Returns
    -------
    A : np.ndarray
        2D bspline A matrix.
    """
    kx0 = kx
    ky0 = ky
    kx_m = np.mean(np.diff(kx))
    ky_m = np.mean(np.diff(ky))
    kx_len = max(kx.shape)
    ky_len = max(ky.shape)
    n_x = ( kx_len - 1 ) + degx
    n_y = ( ky_len - 1 ) + degy
    m = max(x.shape)
    n = n_x * n_y

    # Storage matrix for column indices
    col_i = np.zeros((n_x, n_y), dtype=int)
    A = np.zeros((m, n))

    for i in range(degx):
        kx = [kx[0] - kx_m, *kx, kx[-1] + kx_m]
    for j in range(degy):
        ky = [ky[0] - ky_m, *ky, ky[-1] + ky_m]

    count = 0
    for i in range(degx+1):
        for j in range(degy+1):
            col_i[i, j] = count
            count += 1

    for i in range(degx+1):
        for j in range(degy+1, n_y):
            col_i[i, j] = count
            count += 1

    for i in range(degx+1, n_x):
        for j in range(n_y):
            col_i[i, j] = count
            count += 1

    ix = []
    row_i = [0, 0]
    for i in range(kx_len - 1):
        idxx, *_ = np.where(
            ( x >= kx0[i] ) &
            ( x <= kx0[-1] if i == kx_len - 2 else x < kx0[i+1] )
        )
        for j in range(ky_len - 1):
            idxy, *_ = np.where(
                ( y >= ky0[j] ) &
                ( y <= ky0[-1] if j == ky_len - 2 else y < ky0[j+1] )
            )

            # Get common indices
            idx = np.intersect1d(idxx, idxy)
            # Select coordinates from these indices
            sx = x[idx]
            sy = y[idx]
            # Add them to the main indices
            ix.extend(idx)

            # Indices for matrix building
            row_i = [row_i[1], row_i[1] + len(idx)]
            col_j = col_i[i : i + degx + 1, j : j + degy + 1]

            for Kx in range(degx + 1):
                Bx = build_bspline(sx, kx[i + Kx : i + Kx + degx + 2])
                for Ky in range(degy + 1):
                    By = build_bspline(sy, ky[j + Ky : j + Ky + degy + 2])
                    Ac = Bx * By

                    col_j0 = col_j[Kx, Ky]
                    A[row_i[0] : row_i[1], col_j0] = Ac.reshape(-1)

    return A[np.argsort(ix)]
