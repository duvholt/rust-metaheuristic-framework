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
    # 'loa': {
    #     'p': 150
    # }
}

results = {}

for name, parameters in algorithms.items():
    command = f'cargo run --release -- -t cec2014 -r 25 -u 30 -d 30 -i 10000 -j -p {parameters["p"]} {name}'
    json_output = json.loads(check_output(command.split()).split()[-1])
    for tf in json_output:
        if tf['test_function'] in results:
            results[tf['test_function']].append(tf)
        else:
            results[tf['test_function']] = [tf]

for test_function, alg_data in results.items():
    table_data = f'{test_function} '
    for stat_data in alg_data:
        table_data += f'& {stat_data["mean"]:.2E} & {stat_data["standard_deviation"]:.2E} '
    print(table_data + '\\\\')
