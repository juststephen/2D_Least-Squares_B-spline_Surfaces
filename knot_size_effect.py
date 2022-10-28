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

# Bspline degree
degx, degy = 3, 3

# Sigma array
sigmas = [] 
# RMSEs array
rmses = []

# B-spline skipped coordinates iterations
step_range = (2, 5, 10, 20, 25, 50, 100)
for step in step_range:
    break
    # Knot arrays
    knotx = np.arange(xmin, xmax+step, step)
    knoty = np.arange(ymin, ymax+step, step)

    # Building A matrix
    A = build_bspline_A_2D(x, y, knotx, knoty, degx, degy)
    m, n = A.shape

    # Computing xhat and yhat
    xhat = ( np.linalg.inv(A.T @ A) @ A.T ).dot(z)
    yhat = A.dot(xhat)

    # Computing the corresponding error
    ehat = z - yhat
    sigma = np.sqrt(ehat.T @ ehat / (m - n)).flatten()[0]
    print(f'sigma:\t{sigma}')
    sigmas.append(sigma)

    # Computing the RMSE
    rmse = subcell_rmse(
        x, y, z,
        knotx, knoty, degx, degy, xhat
    )
    print(f'RMSE:\t{rmse}')
    rmses.append(rmse)

#np.save('knot_size_effect.npy', (sigmas, rmses))
sigmas, rmses = np.load('knot_size_effect.npy')

# Plotting
fig = plt.figure(figsize=(20, 10))

ax = fig.add_subplot(121)

ax.plot(step_range, sigmas, 'o-', c='xkcd:deep blue')

ax.grid()

ax.set_ylabel('$\sigma$ [$m$]')

ax.set_xlim((0, 100))

ax = fig.add_subplot(122)

ax.plot(step_range, rmses, 'o-', c='xkcd:brick red')

ax.grid()

ax.set_ylabel('RMSE [$m$]')

ax.set_xlim((0, 100))

# Figure
fig.suptitle(
    'Effect of the knot step size on the fit performance'
)
fig.supxlabel('Knot Step Size [$m$]')

fig.savefig('knot_size_effect.pdf')
