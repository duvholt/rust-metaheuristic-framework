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

def rosenbrock(x,y):
      return (a-x)**2 + b* ((y-x**2))**2

fig = plot.figure()
ax = Axes3D(fig)

json_solutions = json.load(open('../solutions.json'))

solutions = np.array(list(
    map(lambda s: [float(s['x']), float(s['y'])], json_solutions['solutions'])
))

max_x, max_y = solutions.max(axis=0)
min_x, min_y = solutions.min(axis=0)
padding = 0
X = np.linspace(min_x - padding, max_x + padding, 40)
Y = np.linspace(min_y - padding, max_y + padding, 40)
X, Y = np.meshgrid(X, Y)
    

def plot_rosenbrock():
    Z = (a - X)**2 + b * (Y-X*X)**2
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    linewidth=1, edgecolors='#333333', cmap=cm.hot,  norm=LogNorm(vmin=Z.min(), vmax=Z.max()))

def plot_solution(x, y, iteration):
    z = rosenbrock(x, y)
    point = ax.plot(
        [x], [y], [z], 'o',
        mew=1, markersize=2, color='white',
        path_effects=[PathEffects.withStroke(
            linewidth=2, foreground='black')]
    )
    txt = ax.text(x + 0.01, y + 0.01, z, iteration, color='#eeeeee')
    txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='black')])

plot_rosenbrock()

for iteration, solution in enumerate(solutions):
    plot_solution(*solution, iteration)

plot.show()
