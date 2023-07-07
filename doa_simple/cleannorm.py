import numpy as np
import os 

dir = '../dataset/myfeatures/foa_dev'
i=0
max = -1
min = 0
for filename in os.listdir(dir):
    print(i)
    i += 1
    cur_file = os.path.join(dir, filename)
    f = np.genfromtxt(cur_file, delimiter=',', skip_header=0, dtype=float)
    # print(f.shape)
    if f.shape != (300, 256): 
        f = np.delete(f, 0, 1)
        f = np.delete(f, 0, 0)
    # f = np.delete(f, 0, 1)
    max = f.max() if f.max() > max else max
    min = f.min() if f.min() < min else min
    print("max=",max)
    print("min=",min)
    np.savetxt(cur_file, f, fmt='%i', delimiter = ",")  


print("Total max:", max)
print("Total min:", min)
i=0
#TODO need to save normalisation (min max) to scale future data the same
for filename in os.listdir(dir):
    print(i)
    i += 1
    cur_file = os.path.join(dir, filename)
    f = np.genfromtxt(cur_file, delimiter=',', skip_header=0, dtype=float)
    # f between min and max
    f = (f-min) # f between 0 and max-min
    f = f/(max-min) # f beween 0 and 1
    f = f*255 # scale for int8
    np.savetxt(cur_file, f.astype(np.uint8), fmt='%i', delimiter = ",")  
