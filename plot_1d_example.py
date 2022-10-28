import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, splrep, PPoly

np.random.seed(1337)

# Font size
fs = 24
plt.rcParams.update({'font.size': fs-4})

size = 100
x = np.linspace(0, 5 * np.pi, size)
x_10 = np.linspace(0, 5 * np.pi, size * 100)
y = np.cos(x) * np.random.rand(size)
knots = np.linspace(0, 5 * np.pi, 6)

tck = splrep(x=x, y=y, t=knots[1:-1])
yhat_10 = BSpline(*tck)(x_10)

plt.figure(figsize=(16, 9))

plt.plot(x, y, 'o', c='xkcd:burnt orange', label='Scattered Data')

plt.plot(x_10, .5 * np.cos(x_10), 'xkcd:salmon', label='$y=0.5 \cdot \cos\left(x\\right)$')

plt.axvline(knots[0], c='k', ls='--', label='Knots')
for i in knots[1:]:
    plt.axvline(i, c='k', ls='--')

plt.plot(x_10, yhat_10, 'xkcd:cerulean', label='B-spline fit', alpha=.5)

i = 0
tck = splrep(x=x[i*20:(i+1)*20+1], y=y[i*20:(i+1)*20+1], t=())
yhat = BSpline(*tck)(x_10[i*2000:(i+1)*2000+1])
plt.plot(x_10[i*2000:(i+1)*2000+1], yhat, '-.', c='xkcd:green', label='Polynomials')

for i in range(1, 5):
    tck = splrep(x=x[i*20:(i+1)*20+1], y=y[i*20:(i+1)*20+1], t=())
    yhat = BSpline(*tck)(x_10[i*2000:(i+1)*2000+1])
    plt.plot(x_10[i*2000:(i+1)*2000+1], yhat, '-.', c='xkcd:green')

plt.legend()

plt.xlim((0, 5 * np.pi))
plt.ylim((-1, 1))

plt.title('Example of a cubic B-spline Least-Squares fit\n$y=\\alpha\cdot\cos\left(x\\right)$ with randomiser $\\alpha\in\left[0,1\\right)$')
plt.xlabel('$x$-axis')
plt.ylabel('$y$-axis')

plt.savefig('bspline_example.pdf')
