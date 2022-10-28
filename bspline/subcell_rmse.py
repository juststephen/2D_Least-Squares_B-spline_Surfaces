import numpy as np

from bspline import build_bspline_A_2D

def subcell_rmse(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    degx: int,
    degy: int,
    x_hat: np.ndarray,
    subcell_dx: float = 1,
    subcell_dy: float = 1
) -> tuple[float, int]:
    """
    Compute the RMSE of the subcells
    on a 2D B-spline surface.

    Parameters
    ----------
    x : np.ndarray
        X-coords.
    y : np.ndarray
        Y-coords.
    z : np.ndarray
        Z-coords.
    kx : np.ndarray
        X-knots.
    ky : np.ndarray
        Y-knots.
    degx : np.ndarray
        Degree in X.
    degy : np.ndarray
        Degree in Y.
    x_hat : np.ndarray
        Least-Squares fit of
        the 2D B-spline surface.
    subcell_dx : float, default: 1
        Subcell width in X.
    subcell_dy : float, default: 1
        Subcell width in Y.

    Returns
    -------
    RMSE : float
        Root-Mean-Square Error.
    nan_count : int
        Amount of subcells
        without any points.
    """
    # Subcell centers grid
    x_grid, y_grid = np.mgrid[
        kx[0] + subcell_dx / 2 : kx[-1] : subcell_dx,
        ky[0] + subcell_dy / 2 : ky[-1] : subcell_dy
    ]

    # Reshaping for building A
    _x = x_grid.reshape(-1)
    _y = y_grid.reshape(-1)

    # Building A
    A = build_bspline_A_2D(_x, _y, kx, ky, degx, degy)

    # Computing Z-coords at the subcell centers
    y_hat = (A.dot(x_hat)).reshape(x_grid.shape)

    # Create array for storing means
    means = np.zeros(x_grid.shape)

    # Computing means from x, y and z data for RMSE computation
    x_grid_len = len(x_grid[:, 0])
    y_grid_len = len(y_grid[0, :])
    for i in range(x_grid_len):
        idxx, *_ = np.where(
            ( x >= x_grid[i, 0] - subcell_dx / 2 ) &
            (
                x <= x_grid[-1, 0] + subcell_dx / 2
                if i == x_grid_len - 1 else
                x < x_grid[i, 0] + subcell_dx / 2
            )
        )
        for j in range(y_grid_len):
            idxy, *_ = np.where(
                ( y >= y_grid[0, j] - subcell_dy / 2) &
                (
                    y <= y_grid[0, -1] + subcell_dy / 2
                    if j == y_grid_len - 1 else
                    y < y_grid[0, j] + subcell_dy / 2
                )
            )
            # Get common indices
            idx = np.intersect1d(idxx, idxy)
            #print(idxx, idxy)

            # Get z coords within and average them.
            means[i, j] = z[idx].mean()

    # Find all non NaN means (any points in the subcell)
    is_nan = np.isnan(means)
    not_nan = ~is_nan
    # Compute RMSE
    rmse = np.linalg.norm(
        means[not_nan] - y_hat[not_nan]
    ) / np.sqrt(not_nan.size)

    return rmse, sum(is_nan)
