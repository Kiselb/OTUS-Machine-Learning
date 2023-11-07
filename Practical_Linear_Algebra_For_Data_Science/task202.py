from math import sqrt
from numpy import linalg

def vector_norma(vector):
    norma = 0.0
    for item in iter(vector):
        norma = norma + item * item
    norma = sqrt(norma)
    return norma

vector = [1, 2, 4, 8]
print(vector_norma(vector))
print(linalg.norm(vector))
