import gc
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from bspline import build_bspline_A_2D

# Font size
fs = 24
plt.rcParams.update({'font.size': fs-4})

# timings array
timings = []

# B-spline degree iterations
degx, degy = 3, 3

# Iterate over different area sizes
sizes = range(100, 351, 50)
for index, size in enumerate(sizes):
    break
    # Boundaries
    # Easting
    xmin = 524000
    xmax = xmin + size
    # Northing
    ymin = 5826000
    ymax = ymin + size

    # Knot arrays
    knotx = np.array([xmin + 10 * i for i in range(5 * index + 11)])
    knoty = np.array([ymin + 10 * i for i in range(5 * index + 11)])

    # Load x and y values
    x, y, z = np.load(f'selected_area_{size}m.npy')

    # Start time
    start = timer()

    # Building A matrix
    A = build_bspline_A_2D(x, y, knotx, knoty, degx, degy)

    # Computing xhat and yhat
    xhat = ( np.linalg.inv(A.T @ A) @ A.T ).dot(z)
    yhat = A.dot(xhat)

    # End time
    end = timer()
    # Adding to timings
    timings.append(end-start)
    # Clean memory
    gc.collect()

#np.save('area_size_effect.npy', (sizes, timings))
sizes, timings = np.load('area_size_effect.npy')

# Plotting
fig = plt.figure(figsize=(16, 9))

sizes = np.array(sizes)
plt.plot(sizes, timings, 'o-', c='xkcd:royal blue')

plt.xlim((100, 350))
plt.ylim((0, 120))

plt.annotate(
    f'(350 [$m$], {timings[-1]:.0f} [$s$])',
    xy=(310, 120),
    xytext=(225, 101),
    arrowprops={
        'fc': 'k',
        'shrink': .05
    }
)

plt.grid()  

plt.xlabel('Side Length [$m$]')
plt.ylabel('Wall Clock Time [$s$]')

plt.title(
    'Effect of area side length on computation time'
)

fig.savefig('area_size_effect.pdf')
