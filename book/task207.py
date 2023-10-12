import numpy as np

vector_1 = [1.0, 2.0, 3.0]
vector_2 = [1.0, 2.0, 3.0]

dot_1 = 0
for i in range(len(vector_1)):
    dot_1 = dot_1 + vector_1[i]**2

dot_2 = 0
for i in range(len(vector_1)):
    dot_2 = dot_2 + vector_2[i]**2

print(f'Dot 1: {dot_1} Dot 2: {dot_2}')
