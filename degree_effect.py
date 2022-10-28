import numpy as np
import matplotlib.pyplot as plt

from bspline import build_bspline_A_2D, subcell_rmse

# Font size
fs = 24
plt.rcParams.update({'font.size': fs-4})

# Loading from file
x, y, z = np.load('selected_area/100m.npy')

# Boundaries
# Easting
xmin=524000
xmax=524100
# Northing
ymin=5826000
ymax=5826100

# Knot arrays
knotx = np.array([xmin + 10 * i for i in range(11)])
knoty = np.array([ymin + 10 * i for i in range(11)])

# Sigma array
sigmas = []
# RMSEs array
rmses = []

# B-spline degree iterations
deg = range(1, 8)
for degx, degy in zip(deg, deg):
    break
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

#np.save('degree_effect.npy', (sigmas, rmses))
sigmas, rmses = np.load('degree_effect.npy')

# Plotting
fig = plt.figure(figsize=(20, 10))

ax = fig.add_subplot(121)

ax.plot(deg, sigmas, 'o-', c='xkcd:deep blue')

plt.grid()

ax.set_ylabel('$\sigma$ [$m$]')

ax.set_xlim((1, 7))

ax = fig.add_subplot(122)

ax.plot(deg, rmses, 'o-', c='xkcd:brick red')

ax.grid()

ax.set_ylabel('RMSE [$m$]')

ax.set_xlim((1, 7))

fig.suptitle(
    'Effect of the B-spline degree on the fit performance'
)
fig.supxlabel('Degree of the B-spline')

fig.savefig('degree_effect.pdf')
