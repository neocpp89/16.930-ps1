import unittest
import numpy as np

# Returns a tuple with n points and weights (x, w).
remembered_points_and_weights = {}
def golub_welsch(n):
    if n in remembered_points_and_weights:
        return remembered_points_and_weights[n]

    r = np.arange(1, n)
    beta = 0.5 / np.sqrt(1 - (2.0 * r) ** (-2))
    T = np.diag(beta, 1) + np.diag(beta, -1)
    D, V = np.linalg.eig(T)
    x = D
    i = np.argsort(x)
    x = x[i]
    w = V[0, :]
    w = 2.0 * (w[i] ** 2)
    # print x, w
    remembered_points_and_weights[n] = (x, w)
    return (x, w)

# Integrates a function 'f' on the interval [a, b] with n gauss points.
def quadrature(f, a = -1, b = 1, n = 5):
    if (a == b):
        return 0
    if (b < a):
        sign = -1
        a, b = b, a
    else:
        sign = 1
    phi = lambda xi: (((xi + 1) * (b - a) / 2.0) + a)
    x, w = golub_welsch(n)
    s = 0
    for i in range(0, len(x)):
        s += w[i] * f(phi(x[i])) 
    return s*sign*(b-a)/2.0
