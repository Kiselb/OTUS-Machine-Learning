from task202 import vector_norma

def base_vector(vector):
    norma = vector_norma(vector)
    for i in range(len(vector)):
        vector[i] = vector[i] / norma
    return vector

vector = [1, 2, 4, 8]
print(base_vector(vector))
