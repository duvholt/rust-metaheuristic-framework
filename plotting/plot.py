from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plot
import numpy as np
import json
import sys
import matplotlib.patheffects as PathEffects
from matplotlib import colors
import test_functions as tf

# Rosenbrock variables
fig = plot.figure()
ax = Axes3D(fig)

json_solutions = json.load(open('../solutions.json'))

# TODO: Plotting for more than two dimenions doesn't work
if len(json_solutions['solutions'][0]['x']) > 2:
    print('WARNING! Solutions with more than two dimensions is not supported!')

solutions = np.array(list(
    map(lambda s: [*s['x'][:2],
                   float(s['fitness'][0])], json_solutions['solutions'])
))

max_x, max_y, _ = solutions.max(axis=0)
min_x, min_y, _ = solutions.min(axis=0)
padding = (max_y - min_y) * 0.1
linspace_size = 50
X = np.linspace(min_x - padding, max_x + padding, linspace_size)
Y = np.linspace(min_y - padding, max_y + padding, linspace_size)
X, Y = np.meshgrid(X, Y)


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


test_function_name = json_solutions['test_function']
if test_function_name == 'rosenbrock':
    test_function = tf.rosenbrock
elif test_function_name == 'ackley':
    test_function = tf.ackley
elif test_function_name == 'zakharov':
    test_function = tf.zakharov
elif test_function_name == 'himmelblau':
    test_function = tf.himmelblau
elif test_function_name == 'hyper-ellipsoid':
    test_function = tf.axis_parallel_hyper_ellipsoid_function
elif test_function_name == 'moved-hyper-ellipsoid':
    test_function = tf.moved_axis_parallel_hyper_ellipsoid_function
elif test_function_name == 'sphere':
    test_function = tf.sphere
elif test_function_name == 'rastrigin':
    test_function = tf.rastrigin

Z = test_function(X, Y)
ax.plot_surface(
    X, Y, Z,
    linewidth=1, edgecolors='#333333',
    cmap=cm.jet,  norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max())
)

for iteration, solution in enumerate(solutions):
    plot_solution(*solution, iteration)

plot.show()
