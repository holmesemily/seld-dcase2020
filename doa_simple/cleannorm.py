import numpy as np
import os 

dir = '../dataset/myfeatures/foa_dev_norm'
i=0
for filename in os.listdir(dir):
    print(i)
    i += 1
    cur_file = os.path.join(dir, filename)
    f = np.genfromtxt(cur_file, delimiter=',', dtype=float)
    if f.shape != (3000, 512): 
        print("reshape")
        f = np.delete(f, 0, 1)
        f = np.delete(f, 0, 1)
        f = np.delete(f, 0, 0)
        np.savetxt(cur_file, f, delimiter = ",")  

