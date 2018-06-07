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
import random

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
    'SPEA2': '#6CDBC2',
    'Hybrid': '#1f77b4',
    'Archive': '#d62728',
    'Non-dominated': '#FFCC3D',
}
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=new_colors)

markers = ['o', ',', 'X', '^', 'D', 'd', 'P', '*']
markers = ['o'] * 10
big_markers = ['X', '*', 'P']
marker_count = 0

# font = {'family': 'serif', 'size': 16, 'serif': ['computer modern roman']}
# plot.rc('font', **font)
plot.rc('text', usetex=True)
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
label_size = 24
# 2 obj: 8
label_padding = 16
label_grid_size = 20
legend_size = 17
legend_padding = 0.2

plot_objectives = [0, 1, 2]

def plot_json_solutions(json_solutions):
    fig, ax = plot.subplots()
    # ax = plot
    fig.canvas.set_window_title(json_solutions['test_function'])
    # ax = Axes3D(fig)
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
    linspace_size = 500
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
    # ax.plot_surface(
    #     X, Y, Z, rcount=500, ccount=500,
    #     linewidth=0, edgecolors='#333333',
    #     cmap=cm.jet, norm=norm
    # )
    # plot.tight_layout()
    # Moved axis
    # cset = ax.contour(X, Y, Z, [1, 10, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 6000, 10000, 15000, 20000, 30000], offset=Z.min(), cmap=cm.jet, norm=norm)
    cset = ax.contour(X, Y, Z, [1, 10, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 6000, 10000, 15000, 20000, 30000], offset=Z.min(), cmap=cm.jet, norm=norm)

    x, y, z = np.transpose(solutions)
    ax.set_xlabel(r'$x$', fontsize=label_size, labelpad=label_padding - 15)
    ax.set_ylabel(r'$y$', fontsize=label_size, labelpad=label_padding - 15)
    # ax.set_zlabel('Fitness', fontsize=15, labelpad=8)
    ax.tick_params(axis='both', which='major', labelsize=label_grid_size)
    # ax.subplots_adjust(right=10)
    # plot.rc('text', usetex=True)
    # azim = ax.azim
    # ax.view_init(elev=30, azim=-65)
    # ax.rc('font', family='serif')
    # ax.rc('text', usetex=True)
    ax.plot(
        x, y, 'o',
        mew=1, markersize=1.5, color='white',
        path_effects=[PathEffects.withStroke(
            linewidth=2, foreground='#333333')]
    )
    # for iteration, solution in enumerate(solutions):
    #     plot_solution(*solution, iteration)

    fig.tight_layout()
    plot.show()

def multi_plot_optimal(ax, function_name, objectives, **kwargs):
    plot_data = json.load(open(
        '../optimal_solutions/' + function_name + '-' + str(objectives) + 'd.json'))
    pf_true = np.transpose(np.unique(np.array(plot_data), axis=0))
    if len(plot_objectives) > 2:
        kwargs = {**kwargs, **{'depthshade': False}}
    ax.scatter(*[pf_true[i] for i in plot_objectives],
               zorder=0, marker='o', s=0.5, color=optimal_color, **kwargs)

labelled = []

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
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(r'$f_3$', fontsize=label_size, labelpad=label_padding - 3, rotation=90)
    else:
        print('WARNING! Too many objectives to plot!')
        return
    ax.set_xlabel(r'$f_1$', fontsize=label_size, labelpad=label_padding)
    ax.set_ylabel(r'$f_2$', fontsize=label_size, labelpad=label_padding)
    if optimal:
        multi_plot_optimal(ax, function_name, len(solutions), **kwargs)
    if algorithm in algorithm_colors:
        color = algorithm_colors[algorithm]
    else:
        color = None
    marker = markers[marker_count]
    marker_count = (marker_count + 1) % len(markers)
    size = 60
    if marker in big_markers:
        size *= 1.25
    if marker == ',':
        size = 50
    # Adjust x axis
    # plot.ylim(ymax=1.35)
    # plot.xlim(xmax=1.35)
    plot.tight_layout()
    # plot.subplots_adjust(right=40)
    # plot.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # fig.subplots_adjust(left=0.5, right=1, bottom=0.5, top=1, hspace=50)
    ax.tick_params(axis='both', which='major', labelsize=label_grid_size)
    # plot.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    dumb_mode = len(plot_objectives) > 2
    if dumb_mode:
        if display_algorithm in labelled:
            d_label = None
        else:
            d_label = display_algorithm
            labelled.append(display_algorithm)
        ax.scatter(*[solutions[i] for i in plot_objectives], marker=marker, s=size, label=d_label, linewidths=0.4, alpha=1.0, edgecolors='black',
                   color=color, **kwargs)
    else:
        for solution in np.transpose([solutions[i] for i in plot_objectives]):
            if display_algorithm in labelled:
                d_label = None
            else:
                d_label = display_algorithm
                labelled.append(display_algorithm)
            # print(solution)
            ax.scatter(*solution, marker=marker, zorder=random.randint(0,100), s=size, label=d_label, linewidths=0.4, alpha=1.0, edgecolors='black',
                color=color, **kwargs)


def read_jmetal_algorithm_and_plot(algorithm, function_name, fig=None, ax3d=None, same=False):
    suite = ''.join([i for i in function_name if not i.isdigit()])
    if function_name.upper() in ['UF1', 'UF3', 'UF5', 'UF6']:
        i = 4
    elif function_name.upper() == 'UF4':
        i = 15
    elif function_name.upper() == 'UF7':
        i = 14
    elif function_name.upper() == 'UF8':
        i = 17
    else:
        i = 0
    print(function_name)
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

def several_multi_plot(tf):
    # tf = 'uf4'
    files = [
        ('Archive', '../extension_graphs/{}_archive.json'.format(tf)),
        ('Hybrid', '../extension_graphs/{}_hybrid.json'.format(tf)),
        ('Non-dominated', '../extension_graphs/{}_nsamo.json'.format(tf))
    ]
    fig = plot.figure()
    json_solutions = json.load(open(files[0][1]))
    objectives = len(json_solutions['solutions'][0]['fitness'])
    if objectives < len(plot_objectives):
        plot_objectives.remove(2)
    if objectives > 2:
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
    l = plot.legend(loc=1, fontsize=legend_size, borderpad=legend_padding)
    l.set_zorder(1000)
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


def plot_jmetal(function_name, same=False):
    solutions_file = '../moamo_solutions/{}/0.json'.format(function_name)
    # test_function = 'ZDT1'
    # solutions_file = '../uf_graph/{}_dumb.json'.format(test_function.lower())
    json_solutions = json.load(open(solutions_file))
    objectives = len(json_solutions['solutions'][0]['fitness'])
    if objectives < len(plot_objectives):
        plot_objectives.remove(2)
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

    test_function = json_solutions['test_function'].upper()
    algos = ['MOAMO', 'AbYSS', 'MOCell', 'MOEADD', 'NSGAII', 'NSGAIII', 'PAES', 'SMPSO', 'SPEA2']
    if test_function == 'ZDT1':
        algos = ['MOAMO', 'AbYSS', 'SMPSO']
    elif test_function == 'ZDT2':
        algos = ['MOAMO', 'MOCell', 'AbYSS']
    elif test_function == 'ZDT3':
        algos = ['MOAMO', 'MOCell', 'SMPSO']
    elif test_function == 'ZDT4':
        algos = ['MOAMO', 'SMPSO', 'MOCell']
    elif test_function == 'ZDT6':
        algos = ['MOAMO', 'SMPSO', 'ABySS']
    elif test_function == 'DTLZ1':
        algos = ['MOAMO', 'MOEADD', 'NSGAIII']
    elif test_function == 'DTLZ2':
        algos = ['MOAMO', 'NSGAIII', 'MOEADD']
    elif test_function == 'DTLZ3':
        algos = ['MOAMO', 'MOEADD', 'NSGAIII']
    elif test_function == 'DTLZ4':
        algos = ['MOAMO', 'AbYSS', 'MOEADD']
    elif test_function == 'DTLZ5':
        algos = ['MOAMO', 'AbYSS', 'SMPSO']
    elif test_function == 'DTLZ6':
        algos = ['MOAMO', 'SMPSO', 'PAES']
    elif test_function == 'DTLZ7':
        algos = ['MOAMO', 'NSGAII', 'NSGAIII']
    elif test_function == 'UF1':
        algos = ['MOAMO', 'AbYSS', 'MOEADD']
    elif test_function in ['UF2', 'UF3']:
        algos = ['MOAMO', 'AbYSS', 'SMPSO']
    elif test_function == 'UF4':
        algos = ['MOAMO', 'MOEADD', 'NSGAIII']
    elif test_function in ['UF5', 'UF6']:
        algos = ['MOAMO', 'SPEA2', 'NSGAII']
    elif test_function == 'UF7':
        algos = ['MOAMO', 'SMPSO', 'NSGAII']
    elif test_function == 'UF8':
        algos = ['MOAMO', 'MOEADD', 'NSGAIII']
    # UF10
    # for alg in ['MOAMO', 'NSGAIII', 'MOEADD']:
    # algos = ['MOAMO']
    # algos = ['MOAMO', 'AbYSS', 'MOCell', 'MOEADD', 'NSGAII', 'NSGAIII', 'PAES', 'SMPSO', 'SPEA2']
    for alg in algos:
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
        l = plot.legend(loc=1, fontsize=legend_size, borderpad=legend_padding)
        l.set_zorder(1000)
        plot.show()


if len(sys.argv) > 1:
    mode = sys.argv[1]
    if mode == 'jmetal':
        if len(sys.argv) > 3:
            same = sys.argv[3] == 'all'
        else:
            same = False
        plot_jmetal(sys.argv[2], same)
    elif mode == 'custom':
        several_multi_plot(sys.argv[2])
    elif mode == 'listen':
        main()
else:
    print('Please select between listen, jmetal, and custom')
