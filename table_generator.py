from subprocess import check_output
import json


algorithms = {
    'amo': {
        'p': 50
    },
    'da': {
        'p': 1
    },
    'ewa': {
        'p': 50
    },
    'loa': {
        'p': 50
    }
}

results = {}

for name, parameters in algorithms.items():
    command = 'cargo run --release -- -t cec2014 -r 25 -u 30 -d 30 -i 10000 -j -p {} {}'.format(parameters['p'], name)
    json_output = json.loads(check_output(command.split()).split()[-1].decode('utf-8'))
    for tf in json_output:
        if tf['test_function'] in results:
            results[tf['test_function']].append(tf)
        else:
            results[tf['test_function']] = [tf]

color = False

for test_function, alg_data in results.items():
    table_data = test_function
    table_data += ' & Mean'
    for stat_data in alg_data:
        table_data += ' & {:.2E} '.format(stat_data["mean"])
    table_data += '\\\\\n'
    table_data += ' & StdDev'
    for stat_data in alg_data:
        table_data += ' & {:.2E} '.format(stat_data["standard_deviation"])
    table_data += '\\\\'
    table_data += '\\showrowcolors' if color else '\\hiderowcolors'
    print(table_data)
    color = not color
