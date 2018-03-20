import matplotlib.pyplot as plt
import numpy as np
import json

json_solutions = json.load(open('../solutions.json'))

solutions = np.array(list(
    map(lambda s: s['fitness'], json_solutions['solutions'])
))

print(solutions)

x = solutions[:, 0]
y = solutions[:, 1]
plt.scatter(x, y, marker='o', s=5)

function_name = json_solutions['test_function']
plot_data = json.load(open(function_name + '.json'))
pf_true = np.array(plot_data)


x = pf_true[:, 0]
y = pf_true[:, 1]
plt.scatter(x, y, marker='x', s=0.5)


# Show the boundary between the regions:
# theta = np.arange(0, np.pi / 2, 0.01)
# r0 = 0.6
# plt.plot(r0 * np.cos(theta), r0 * np.sin(theta))

plt.show()
