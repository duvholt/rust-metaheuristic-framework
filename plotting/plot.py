import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import json
import sys
import time
import matplotlib.patheffects as PathEffects
from matplotlib import colors
import test_functions as tf
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from multiprocessing import Process

solutions_file = '../solutions.json'


def plot_json_solutions(json_solutions):
    fig = plot.figure()
    ax = Axes3D(fig)
    solutions = np.array(list(
        map(lambda s: [*s['x'][:2],
                       float(s['fitness'][0])], json_solutions['solutions'])
    ))

    if json_solutions['plot_bounds']:
        max_x = max_y = json_solutions['upper_bound']
        min_x = min_y = json_solutions['lower_bound']
    else:
        max_x, max_y, _ = solutions.max(axis=0)
        min_x, min_y, _ = solutions.min(axis=0)

    padding = (max_y - min_y) * 0.1
    linspace_size = 50
    X = np.linspace(min_x - padding, max_x + padding, linspace_size)
    Y = np.linspace(min_y - padding, max_y + padding, linspace_size)
    X, Y = np.meshgrid(X, Y)

    def plot_solution(x, y, z, iteration):
        if len(solutions) > 300:
            return
        txt = ax.text(x, y, z, iteration, color='#eeeeee', weight='bold')
        black_border = PathEffects.withStroke(linewidth=2, foreground='black')
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
    elif test_function_name == 'katsuura':
        test_function = tf.katsuura

    Z = test_function(X, Y)
    if Z.min() >= 0:
        norm = colors.LogNorm(vmin=Z.min(), vmax=Z.max())
    else:
        norm = colors.Normalize()
    ax.plot_surface(
        X, Y, Z,
        linewidth=1, edgecolors='#333333',
        cmap=cm.jet, norm=norm
    )

    x, y, z = np.transpose(solutions)
    ax.plot(
        x, y, z, 'o',
        mew=1, markersize=2, color='white',
        path_effects=[PathEffects.withStroke(
            linewidth=2, foreground='black')]
    )
    for iteration, solution in enumerate(solutions):
        plot_solution(*solution, iteration)

    plot.show()


def multi_plot(json_solutions):
    solutions = np.array(list(
        map(lambda s: s['fitness'], json_solutions['solutions'])
    ))
    solutions = np.transpose(solutions)

    function_name = json_solutions['test_function']
    plot_data = json.load(open('../optimal_solutions/' + function_name + '-' + str(len(solutions)) + 'd.json'))
    pf_true = np.transpose(np.array(plot_data))

    if len(solutions) == 2:
        ax = plot
    elif len(solutions) == 3:
        fig = plot.figure()
        ax = Axes3D(fig)
    else:
        print('WARNING! Too many objectives to plot!')
        return

    ax.scatter(*solutions, marker='o', s=5)
    ax.scatter(*pf_true, marker='x', s=0.5)
    plot.show()


def read_and_plot():
    json_solutions = json.load(open(solutions_file))
    if len(json_solutions['solutions'][0]['x']) > 2:
        print('WARNING! Solutions with more than two dimensions is not supported!')

    if len(json_solutions['solutions'][0]['fitness']) > 1:
        multi_plot(json_solutions)
    else:
        plot_json_solutions(json_solutions)


def plot_process():
    p = Process(target=read_and_plot)
    p.start()


class PlotHandler(PatternMatchingEventHandler):
    def on_modified(self, event):
        plot_process()


def main():
    patterns = [solutions_file]
    event_handler = PlotHandler(patterns=patterns)
    observer = Observer()
    observer.schedule(event_handler, '..')
    observer.start()
    plot_process()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == '__main__':
    main()
