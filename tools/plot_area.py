import matplotlib.pyplot as plt
import numpy as np

# Loading from file
x, y, z = np.load('uniform_area.npy')

print(x.size)

# Create 3d figure
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(x[::10], y[::10], z[::10])

plt.show()
