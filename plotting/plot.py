from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plot
import numpy as np
import json

# Rosenbrock variables
a = 1.0
b = 100.0

def rosenbrock(x,y):
      return (a-x)**2 + b* ((y-x**2))**2


fig = plot.figure()
ax = fig.gca(projection='3d')

json_solutions = json.load(open('solutions.json'))

solutions = np.array(list(
    map(lambda s: [float(s['x']), float(s['y'])], json_solutions['solutions'])
))

s = 0.25   # Try s=1, 0.25, 0.1, or 0.05
max_x, max_y = solutions.max(axis=0)
min_x, min_y = solutions.min(axis=0)
padding = 0.5
X = np.arange(min_x - padding, max_x + padding + s, s)   #Could use linspace instead if dividing
Y = np.arange(min_y - padding, max_y + padding + s, s)  # evenly instead of stepping...
X, Y = np.meshgrid(X, Y)
    

def plot_rosenbrock():
    Z = (a - X)**2 + b * (Y-X*X)**2
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)  #Try coolwarm vs jet

def plot_solution(x, y, iteration):
    z = rosenbrock(x, y)
    ax.plot([x], [y], [z], 'x', mew=1, markersize=5, color='red', label='1')
    ax.text(x, y, z, iteration, color='#eeeeee')

plot_rosenbrock()

for iteration, solution in enumerate(solutions):
    plot_solution(*solution, iteration)

plot.show()
