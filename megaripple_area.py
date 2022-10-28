import numpy as np
from matplotlib import ticker
import matplotlib.pyplot as plt
from scipy.io import loadmat

from bspline import build_bspline_A_2D

# Font size
fs = 20
plt.rcParams.update({'font.size': fs-4, 'axes.formatter.useoffset': True})

# Loading from file
x, y, z = np.load('selected_area/100m.npy')

# Select less points to speed up
x = x[::10]
y = y[::10]
z = z[::10]

# Boundaries
# Easting
xmin=524000 
xmax=524100
# Northing
ymin=5826000
ymax=5826100

# Knot arrays
knotx = np.array([xmin + 5 * i for i in range(41)])
knoty = np.array([ymin + 5 * i for i in range(41)])

# Bspline degree
degx, degy = 3, 3

xhat = np.load('xhat_200m.npy')

# Grid for plotting
x, y = np.mgrid[xmin : xmax : 1, ymin : ymax : 1]
_x = x.reshape(-1)
_y = y.reshape(-1)
A = build_bspline_A_2D(_x, _y, knotx, knoty, degx, degy)
yhat = (A.dot(xhat)).reshape(x.shape)

# Plotting
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')

ax.plot_surface(x, y, yhat, cmap='viridis')

ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

ax.tick_params(axis='both', which='major', labelsize=fs-10)

ax.set_xlabel('Easting [$m$]', labelpad=16)
ax.set_ylabel('Northing [$m$]', labelpad=16)
ax.set_zlabel('Depth [$m$]', labelpad=16)

ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))


ax.view_init(elev=5, azim=0)

plt.savefig('megaripple_area_5_0.pdf')

ax.view_init(elev=35, azim=0)

plt.savefig('megaripple_area_35_0.pdf')

ax.view_init(elev=35, azim=-60)

plt.savefig('megaripple_area_35_-60.pdf')

ax.view_init(elev=35, azim=120)

plt.savefig('megaripple_area_35_120.pdf')

plt.show()
