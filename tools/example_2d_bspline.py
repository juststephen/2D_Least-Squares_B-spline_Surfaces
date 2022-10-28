import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from bspline import build_bspline_A_2D

np.random.seed(1337)

# Knot arrays
knotx = np.array([-2 + .4 * i for i in range(11)])
knoty = np.array([-2 + .4 * i for i in range(11)])

# Bspline degree
degx, degy = 3, 3

# Coordinates to fit to
m1 = 20000
# If True it uses matlab's x and y, otherwise generate them
if True:
    x = loadmat('x.mat')['x']
    y = loadmat('y.mat')['y']
else:
    x = 4 * (np.random.rand(m1, 1) - .5)
    y = 4 * (np.random.rand(m1, 1) - .5)
    x[m1-4] = -2; x[m1-3] = 2; x[m1-2] = -2; x[m1-1] = 2
    y[m1-4] = -2; y[m1-3] = 2; y[m1-2] = 2; y[m1-1] = -2
z = x * np.exp( - x ** 2 - y ** 2 )

# Building A matrix
A = build_bspline_A_2D(x, y, knotx, knoty, degx, degy)
m, n = A.shape

# Computing xhat and yhat
xhat = ( np.linalg.inv(A.T @ A) @ A.T ).dot(z)
yhat = A.dot(xhat)

# Computing the corresponding error
ehat = yhat - z
sigma = np.sqrt(ehat.T @ ehat / (m - n)).flatten()[0]
print(f'sigma:\t{sigma}')

# Plotting
fig = plt.figure()
ax0 = fig.add_subplot(121, projection='3d')
ax1 = fig.add_subplot(122)

ax0.scatter(x, y, yhat)

im = ax1.scatter(x, y, c=ehat)
plt.colorbar(im, ax=ax1)

plt.show()
