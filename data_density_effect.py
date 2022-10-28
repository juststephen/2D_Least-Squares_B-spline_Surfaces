import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from bspline import build_bspline_A_2D, subcell_rmse

# Font size
fs = 24
plt.rcParams.update({'font.size': fs-4})

# Loading from file
x, y, z = np.load('selected_area/100m.npy')

# Boundaries
# Easting
xmin=524000 # [m]
xmax=524100 # [m]
# Northing
ymin=5826000 # [m]
ymax=5826100 # [m]

# Knot arrays
step = 10 # [m]
knotx = np.arange(xmin, xmax + step, step)
knoty = np.arange(ymin, ymax + step, step)

# Bspline degree
degx, degy = 3, 3

# Sigma array
sigmas = []
# RMSEs array
rmses = []

# Baseline A matrix for computing the error
A_base = build_bspline_A_2D(x, y, knotx, knoty, degx, degy)

# B-spline data density iterations
step_range = np.arange(1, 101)

# Generating data
if True:
    for step in step_range:
        # Building A matrix
        A = build_bspline_A_2D(x[::step], y[::step], knotx, knoty, degx, degy)
        m, n = A.shape
        print(m,n)

        # Computing xhat and yhat
        xhat = ( np.linalg.inv(A.T @ A) @ A.T ).dot(z[::step])
        yhat = A_base.dot(xhat)

        # Computing the corresponding error
        ehat = z - yhat
        sigma = np.sqrt(ehat.T @ ehat / (m - n)).flatten()[0]
        print(f'sigma:\t{sigma}')
        sigmas.append(sigma)

        # Computing the RMSE
        rmse, _ = subcell_rmse(
            x, y, z,
            knotx, knoty, degx, degy, xhat
        )
        print(f'RMSE:\t{rmse}')
        rmses.append(rmse)

    np.save('data_density_effect.npy', (sigmas, rmses))
# Loading data
else:
    sigmas, rmses = np.load('data_density_effect.npy')

xticks_locs = (1, 10, 100)
xticks_labels = ('$10^{0}$', '$10^{-1}$', '$10^{-2}$')

# Plotting
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10))

# Sigma
ax0.plot(step_range, sigmas, c='xkcd:deep blue')

ax0.grid()

ax0.set_ylabel('$\sigma$ [$m$]')

ax0.set_xlim((1, 100))
ax0.set_xscale('log')
ax0.invert_xaxis()
ax0.set_xticks(xticks_locs, xticks_labels)

ax0.set_ylim((0, 5))

# RMSE
ax1.plot(step_range, rmses, c='xkcd:brick red')

ax1.grid()

ax1.set_ylabel('RMSE [$m$]')

ax1.set_xlim((1, 100))
ax1.set_xscale('log')
ax1.invert_xaxis()
ax1.set_xticks(xticks_locs, xticks_labels)

ax1.set_ylim((.175, .5))


# Figure
fig.suptitle(
    'Effect of data density on the fit performance'
)
fig.supxlabel('Data density fraction [-]')

fig.savefig('data_density_effect.pdf')
