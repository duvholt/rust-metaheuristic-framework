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
import matplotlib as mpl

solutions_file = '../solutions.json'

# Modified version of matplotlib's default color scheme
optimal_color = '#ff7f0e'
new_colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
algorithm_colors = {
    'MOAMO': '#1f77b4',
    'AbYSS': '#78C831',
    'MOCell': '#d62728',
    'MOEADD': '#9467bd',
    'NSGAII': '#8c564b',
    'NSGAIII': '#F88297',
    'PAES': '#7f7f7f',
    'SMPSO': '#FFCC3D',
    'SPEA2': '#6CDBC2'
}
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=new_colors)

markers = ['o', ',', 'X', '^', 'D', 'd', 'P', '*']
big_markers = ['X', '*', 'P']
marker_count = 0

plot_objectives = [0, 1]

def plot_json_solutions(json_solutions):
    fig = plot.figure()
    fig.canvas.set_window_title(json_solutions['test_function'])
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
    elif test_function_name == 'high-elliptic':
        test_function = tf.high_elliptic
    elif test_function_name == 'bent-cigar':
        test_function = tf.bent_cigar
    elif test_function_name == 'griewank':
        test_function = tf.griewank
    elif test_function_name == 'schwefel':
        test_function = tf.schwefel
    elif test_function_name == 'weierstrass':
        test_function = tf.weierstrass
    elif test_function_name == 'happycat':
        test_function = tf.happycat
    elif test_function_name == 'hgbat':
        test_function = tf.hgbat
    elif test_function_name == 'levy05':
        test_function = tf.levy05
    elif test_function_name == 'easom':
        test_function = tf.easom
    elif test_function_name == 'discus':
        test_function = tf.discus
    elif test_function_name == 'griewank-rosenbrock':
        test_function = tf.griewank_rosenbrock
    elif test_function_name == 'expanded-schaffer6':
        test_function = tf.expanded_schaffer6

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
    cset = ax.contour(X, Y, Z, zdir='z', offset=Z.min(), cmap=cm.jet)

    x, y, z = np.transpose(solutions)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Fitness')
    ax.plot(
        x, y, z, 'o',
        mew=1, markersize=2, color='white',
        path_effects=[PathEffects.withStroke(
            linewidth=2, foreground='black')]
    )
    for iteration, solution in enumerate(solutions):
        plot_solution(*solution, iteration)

    plot.show()

def multi_plot_optimal(ax, function_name, objectives, **kwargs):
    plot_data = json.load(open(
        '../optimal_solutions/' + function_name + '-' + str(objectives) + 'd.json'))
    pf_true = np.transpose(np.unique(np.array(plot_data), axis=0))
    if len(plot_objectives) > 2:
        kwargs = {**kwargs, **{'depthshade': False}}
    ax.scatter(*[pf_true[i] for i in plot_objectives], marker='x', s=0.5, color=optimal_color, **kwargs)

def multi_plot(function_name, solutions, fig=None, algorithm=None, ax3d=None, optimal=True):
    global marker_count
    if not fig:
        fig = plot.figure(100)
    if algorithm:
        if algorithm == 'MOEADD':
            display_algorithm = 'MOEA/DD'
        elif algorithm == 'NSGAIII':
            display_algorithm = 'NSGA-III'
        elif algorithm == 'NSGAII':
            display_algorithm = 'NSGA-II'
        else:
            display_algorithm = algorithm
        fig.canvas.set_window_title('{} on {}'.format(
            display_algorithm, function_name.upper()))
    else:
        fig.canvas.set_window_title(function_name.upper())
        display_algorithm = None
    kwargs = {}
    if len(plot_objectives) == 2:
        ax = fig.add_subplot(111)
        ax.tick_params(axis='both', which='major', labelsize=15)
    elif len(plot_objectives) == 3:
        if ax3d:
            ax = ax3d
        else:
            ax = Axes3D(fig)
        ax.view_init(elev=45, azim=45)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.dist = 11
        kwargs['depthshade'] = False
        ax.set_zlabel('f3')
    else:
        print('WARNING! Too many objectives to plot!')
        return
    ax.set_xlabel('f1', fontsize=15)
    ax.set_ylabel('f3', fontsize=15)
    if optimal:
        multi_plot_optimal(ax, function_name, len(solutions), **kwargs)
    if algorithm in algorithm_colors:
        color = algorithm_colors[algorithm]
    else:
        color = None
    marker = markers[marker_count]
    marker_count = (marker_count + 1) % len(markers)
    if marker in big_markers:
        size = 140
    else:
        size = 100
    ax.scatter(*[solutions[i] for i in plot_objectives], marker=marker, s=size, label=display_algorithm, linewidths=0.4, alpha=0.8, edgecolors='black',
               color=color, **kwargs)


def read_jmetal_algorithm_and_plot(algorithm, function_name, fig=None, ax3d=None, same=False):
    suite = ''.join([i for i in function_name if not i.isdigit()])
    i = 0
    solutions = json.load(open(
        '../jmetal_data/{}/{}/{}/FUN{}.tsv.json'.format(suite, algorithm, function_name.upper(), i)))
    solutions = np.transpose(solutions)
    multi_plot(function_name, solutions, algorithm=algorithm, fig=fig, ax3d=ax3d, optimal=not same)

def read_and_plot():
    json_solutions = json.load(open(solutions_file))
    if len(json_solutions['solutions'][0]['x']) > 2:
        print('WARNING! Solutions with more than two dimensions is not supported!')
    if len(json_solutions['solutions'][0]['fitness']) > 1:
        solutions = np.array(list(
            map(lambda s: s['fitness'], json_solutions['solutions'])
        ))
        solutions = np.transpose(solutions)
        function_name = json_solutions['test_function']
        multi_plot(function_name, solutions)
        if json_solutions['plot_input']:
            for i in range(len(json_solutions['solutions'][0]['x']) - 25):
                input_variables = np.array(
                    list(map(lambda s: s['x'][i:i+3], json_solutions['solutions'])))
                input_variables = np.transpose(input_variables)
                input_data = json.load(
                    open('../optimal_input/' + function_name + '-optimal-input.json'))
                ps_true = np.transpose(np.unique(np.array(input_data), axis=0))
                fig = plot.figure(i)
                fig.canvas.set_window_title(function_name)
                ax = Axes3D(fig)
                ax.scatter(*input_variables, marker='o', s=5)
                ax.scatter(*ps_true[i:i+3], marker='x', s=0.5)
        plot.show()
    else:
        plot_json_solutions(json_solutions)

def several_multi_plot():
    files = [
        ('Archive', '../solutions.json'),
        ('Hybrid', '../solutions.json'),
        ('Non-dominated', '../solutions.json')
    ]
    fig = plot.figure()
    json_solutions = json.load(open(files[0][1]))
    if len(json_solutions['solutions'][0]['fitness']) > 2:
        ax3d = Axes3D(fig)
    else:
        ax3d = fig.add_subplot(111)
    json_solutions = json.load(open(files[0][1]))
    solutions = np.array(list(
        map(lambda s: s['fitness'], json_solutions['solutions'])
    ))
    multi_plot_optimal(ax3d, json_solutions['test_function'], len(solutions[0]))
    for name, file in files:
        json_solutions = json.load(open(file))
        solutions = np.array(list(
            map(lambda s: s['fitness'], json_solutions['solutions'])
        ))
        solutions = np.transpose(solutions)
        function_name = json_solutions['test_function']
        multi_plot(function_name, solutions, fig=fig, algorithm=name, ax3d=ax3d, optimal=False)
    plot.legend(loc=1)
    plot.show()

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

def plot_jmetal(same=False):
    json_solutions = json.load(open(solutions_file))
    if same:
        objectives = len(json_solutions['solutions'][0]['fitness'])
        fig = plot.figure()
        if len(plot_objectives) > 2:
            ax3d = Axes3D(fig)
        else:
            ax3d = fig.add_subplot(111)
        multi_plot_optimal(
            ax3d, json_solutions['test_function'], objectives)
    else:
        fig = None
        ax3d = None
    for alg in ['MOAMO', 'AbYSS', 'MOCell', 'MOEADD', 'NSGAII', 'NSGAIII', 'PAES', 'SMPSO', 'SPEA2']:
        if alg == 'MOAMO':
            json_solutions = json.load(open(solutions_file))
            solutions = np.array(list(
                map(lambda s: s['fitness'], json_solutions['solutions'])
            ))
            solutions = np.transpose(solutions)
            function_name = json_solutions['test_function']
            multi_plot(function_name, solutions, fig, alg, ax3d, optimal=not same)
        else:
            read_jmetal_algorithm_and_plot(alg, json_solutions['test_function'], fig=fig, ax3d=ax3d, same=same)
        if not same:
            plot.show()
    if same:
        plot.legend(loc=1, fontsize=14)
        plot.show()


if len(sys.argv) > 1:
    mode = sys.argv[1]
    if mode == 'jmetal':
        if len(sys.argv) > 2:
            same = sys.argv[2] == 'all'
        else:
            same = False
        plot_jmetal(same)
    elif mode == 'custom':
        several_multi_plot()
    elif mode == 'listen':
        main()
else:
    print('Please select between listen, jmetal, and custom')
