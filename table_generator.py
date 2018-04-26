from collections import OrderedDict
from subprocess import check_output
import json
import sys

default_parameters = {
    '-r': 1,  # Runs
    '-d': 2,  # Dimensions
    '-u': 100,  # Upper bound
    '-i': 1000,  # Iterations
    '-p': 50,  # Population
}


def dict_to_params(d):
    params = ''
    for key, value in d.items():
        params += ' {key} {value}'.format(key=key, value=value)
    return params


def specific_parameters(parameters):
    if 'specific' in parameters:
        algorithm_parameters = dict_to_params(parameters['specific'])
        del parameters['specific']
    else:
        algorithm_parameters = ''
    return algorithm_parameters


def run_algorithm(algorithm, parameters, results):
    algorithm_parameters = specific_parameters(parameters)
    suite_parameters = dict_to_params(parameters)
    command = 'cargo run --release -- -t single -j {suite_parameters} {algorithm} {algorithm_parameters}'.format(
        algorithm_parameters=algorithm_parameters, suite_parameters=suite_parameters, algorithm=algorithm)
    json_output = json.loads(check_output(
        command.split()).split()[-1].decode('utf-8'))
    for tf in json_output:
        if tf['test_function'] in results:
            results[tf['test_function']].append(tf)
        else:
            results[tf['test_function']] = [tf]


def tune_algorithm():
    algorithm = 'ewa'
    configs = [
        {
            'specific': {
                '-c': 0.9
            }
        },
         {
            'specific': {
                '-c': 0.99
            }
        },
        {
            '-u': 32,
            '-l': -28,
        },
        {
            '-u': 32,
            '-l': -28,
             'specific': {
                 '-c': 0.99
             }
        },
    ]

    results = OrderedDict()

    for config_parameters in configs:
        parameters = {**default_parameters, **config_parameters}
        run_algorithm(algorithm, parameters, results)

    generate_table(results)


def algorithms():
    configs = [
        {'algorithm': 'amo'},
        {'algorithm': 'da', '-p': 1},
        {'algorithm': 'ewa'},
        {'algorithm': 'loa'},
    ]

    results = OrderedDict()

    for config_parameters in configs:
        algorithm = config_parameters['algorithm']
        del config_parameters['algorithm']
        parameters = {**default_parameters, **config_parameters}
        run_algorithm(algorithm, parameters, results)

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
