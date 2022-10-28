import numpy as np
from matplotlib import ticker
import matplotlib.pyplot as plt

# Font size
fs = 24
plt.rcParams.update({'font.size': fs-4})

# Loading from file
x, y, z = np.load('selected_area/200m.npy')

plt.figure(figsize=(12, 8))

plt.scatter(x[::80], y[::80], c='k')

plt.title(
    'Used points with a data density fraction of 0.0125 [$-$]'
    '\nThe grid cells are $10\\times10$ [$m$]'
)
plt.xlabel('Easting [$m$]')
plt.ylabel('Northing [$m$]')

if True:
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
else:
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter(''))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter(''))

plt.gca().set_aspect('equal')

plt.xlim((524000, 524000+100))
plt.ylim((5826000, 5826000+100))

plt.xticks(range(524000, 524000+100+10, 10), minor=True)
plt.yticks(range(5826000, 5826000+100+10, 10), minor=True)
plt.grid(which='both')

plt.savefig('data_density_display.pdf')
