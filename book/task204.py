def norm_vector(vector, norma):
    for i in range(len(vector)):
        vector[i] = vector[i] / norma
    return vector

vector = [1, 2, 4, 8]
print(norm_vector(vector, 100))
