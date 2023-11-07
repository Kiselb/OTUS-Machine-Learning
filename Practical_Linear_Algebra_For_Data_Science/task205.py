import numpy as np

rowVec = np.array([ [1.0, 2.0, 3.0] ])
colVec = np.zeros((rowVec.shape[1], 1))

for i in range(rowVec.shape[1]):
    colVec[i, 0] = rowVec[0, i]

print(colVec)
