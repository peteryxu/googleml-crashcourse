import numpy as np


# Matrix-vector multiplication
#A(m,n) * b(n,p) = c(m,p)

A = np.array([[1, 2], [3, 4]])  # 2x2 matrix
b = np.array([1, 2])            # 2-dimensional vector
result = A.dot(b)               # Result: [1*1 + 2*2, 3*1 + 4*2] = [5, 11]


xi = np.array([[1, 2], [3, 4]])  # 2x2 matrix
theta = np.array([0.5, 1.5])     # 2-dimensional vector
# Step-by-step calculation
dot_product = xi.dot(theta)       # Result: [1*0.5+2*1.5, 3*0.5+4*1.5] = [3.5, 7.5]

yi = np.array([1, 2])            # 2-dimensional vector
difference = dot_product - yi     # Result: [2.5, 5.5]


transpose = xi.T                  # Result: [[1, 3], [2, 4]]
gradients = 2 * transpose.dot(difference)  # Result: [2*20.5, 2*45.5] = [41, 91]