import numpy as np
import os

directory = '../dataset/metadata_dev'
col = ['frames', 'event', 'x', 'y', 'z']
i = 0
for filename in os.listdir(directory):
    print(i)
    i += 1
    cur_file = os.path.join(directory, filename)
    f = np.genfromtxt(cur_file, delimiter=',', dtype=float)
    f = np.delete(f, 1, 1)
    # print(f)
    np.savetxt(cur_file, f, delimiter = ",")  