from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plot
import numpy as np
import json
import matplotlib.patheffects as PathEffects
from matplotlib.colors import LogNorm

# Rosenbrock variables
a = 1.0
b = 100.0

fig = plot.figure()
ax = Axes3D(fig)

json_solutions = json.load(open('../solutions.json'))

solutions = np.array(list(
    map(lambda s: [float(s['x']), float(s['y']),
                   float(s['fitness'])], json_solutions['solutions'])
))

max_x, max_y, _ = solutions.max(axis=0)
min_x, min_y, _ = solutions.min(axis=0)
padding = 0
X = np.linspace(min_x - padding, max_x + padding, 40)
Y = np.linspace(min_y - padding, max_y + padding, 40)
X, Y = np.meshgrid(X, Y)


def ackley(X, Y):
    x = np.array([X, Y])
    cost = np.zeros(x.shape[1:])
    a = 20.0
    b = 0.2
    c = 2.0 * np.pi
    cost = -a * np.exp(-b * np.sqrt(0.5 * (x[0]**2 + x[1]**2))) - np.exp(
        0.5 * (np.cos(c * x[0]) + np.cos(c * x[1]))) + a + np.exp(1.0)
    return cost


def zakharov(X, Y):
    x = np.array([X, Y])
    c = np.zeros(x.shape[1:])
    # Calculate Cost
    c = np.sum([x[i]**2 for i in range(0, 2)], axis=0) + \
        np.sum([0.5 * (i + 1) * x[i] for i in range(0, 2)], axis=0)**2 + \
        np.sum([0.5 * (i + 1) * x[i] for i in range(0, 2)], axis=0)**4
    # Return Cost
    return c


def rosenbrock(X, Y):
    return (a - X)**2 + b * (Y-X*X)**2


def plot_solution(x, y, z, iteration):
    ax.plot(
        [x], [y], [z], 'o',
        mew=1, markersize=2, color='white',
        path_effects=[PathEffects.withStroke(
            linewidth=2, foreground='black')]
    )
    txt = ax.text(x, y, z, iteration, color='#eeeeee')
    black_border = PathEffects.withStroke(linewidth=1, foreground='black')
    txt.set_path_effects([black_border])


# Z = rosenbrock(X, Y)
# Z = zakharov(X, Y)
Z = ackley(X, Y)
ax.plot_surface(
    X, Y, Z, rstride=1, cstride=1,
    linewidth=1, edgecolors='#333333',
    cmap=cm.hot,  norm=LogNorm(vmin=Z.min(), vmax=Z.max())
)

for iteration, solution in enumerate(solutions):
    plot_solution(*solution, iteration)

plot.show()
