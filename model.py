import numpy as np

filename = 'Flood_dataset.txt'
data = []
input = []
design_output = []
with open(filename) as f:
    for line in f.readlines()[2:]:
        data.append([float(element[:-1]) for element in line.split()])
        print(f"{data}")
# data = np.array(data)
# np.random.shuffle(data)
# min_vals = np.min(data, axis=0)
# max_vals = np.max(data, axis=0)
# epsilon = 1e-8  # A very small value to avoid division by zero
# data = (data - min_vals) / (max_vals - min_vals + epsilon)
# for i in data:
#     input.append(i[:-1])
#     design_output.append(np.array(i[-1]))