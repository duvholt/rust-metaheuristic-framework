from collections import OrderedDict
from subprocess import check_output
import json
import sys

default_parameters = {
    'r': 25,  # Runs
    'd': 30,  # Dimensions
    'u': 30,  # Upper bound
    'i': 10000,  # Iterations
    'p': 50,  # Population
}


def run_algorithm(parameters, results):
    command = 'cargo run --release -- -t cec2014 -r {r} -u {u} -d {d} -i {i} -j -p {p} {algorithm}'.format(
        **parameters)
    json_output = json.loads(check_output(
        command.split()).split()[-1].decode('utf-8'))
    for tf in json_output:
        if tf['test_function'] in results:
            results[tf['test_function']].append(tf)
        else:
            results[tf['test_function']] = [tf]


def tune_algorithm():
    algorithm = 'amo'
    configs = [
        {
            'p': 25,
            'i': 10000,
        },
        {
            'p': 35,
            'i': 10000,
        },
        {
            'p': 50,
            'i': 10000,
        },
        {
            'p': 75,
            'i': 10000,
        }
    ]

    results = OrderedDict()

    for config_parameters in configs:
        parameters = {**default_parameters, **config_parameters, 'algorithm': algorithm}
        run_algorithm(parameters, results)

    generate_table(results)


def algorithms():
    configs = [
        {'algorithm': 'amo'},
        {'algorithm': 'da', 'p': 1},
        {'algorithm': 'ewa'},
        {'algorithm': 'loa'},
    ]

    results = OrderedDict()

    for config_parameters in configs:
        parameters = {**default_parameters, **config_parameters}
        run_algorithm(parameters, results)

    generate_table(results)


def generate_table(results):
    color = False
    for test_function, alg_data in results.items():
        table_data = test_function
        table_data += ' & Mean'
        min_mean = min(map(lambda x: x['mean'], alg_data))
        for stat_data in alg_data:
            mean = stat_data["mean"]
            mean_format = '{:.2E}'.format(mean)
            if mean == min_mean:
                mean_format = '\\tbnum{{{}}}'.format(mean_format)
            table_data += ' & {} '.format(mean_format)
        table_data += '\\\\\n'
        table_data += ' & StdDev'
        for stat_data in alg_data:
            table_data += ' & {:.2E} '.format(stat_data["standard_deviation"])
        table_data += '\\\\'
        table_data += '\\showrowcolors' if color else '\\hiderowcolors'
        print(table_data)
        color = not color


if len(sys.argv) == 2 and sys.argv[1] == 'tune':
    tune_algorithm()
else:
    algorithms()
