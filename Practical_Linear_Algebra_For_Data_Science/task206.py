import numpy as np
from task202 import vector_norma

vector = [1.0, 2.0, 3.0]
norma_square = vector_norma(vector) * vector_norma(vector)
dot = 0
for i in range(len(vector)):
    dot = dot + vector[i] * vector[i]

print(f'Norma square: {norma_square} Dot: {dot}')
