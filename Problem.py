#!/usr/bin/env python
import Element
import numpy as np
import pylab
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve

left = 0
right = 1
Ne = 4
h = (right - left) / Ne

nu = 1
b = 0
c = 0
g0 = 0
g1 = 1
f = lambda x: x ** 2

X = np.linspace(left, right, Ne+1)
H = np.diff(X)
Elements = [0]*(Ne)

order = 1
ConnectivityFunction = lambda element_idx, local_node_idx: (order)*element_idx + local_node_idx
ConnectivityMatrix = np.empty([Ne, 2])
BoundaryFluxDOFs = [Ne+1, Ne+2]
Ndof = Ne+3

I = np.zeros([(2+Ndof)*(order+1)**2])
J = np.zeros([(2+Ndof)*(order+1)**2])
K = np.zeros([(2+Ndof)*(order+1)**2])
F = np.zeros([Ndof, 1])
coo_idx = 0

# Create triangulation
for i in range(0, Ne):
    Elements[i] = Element.Linear_1D(X[i], X[i+1])
    print Elements[i]
    for j in range(0, order+1):
        ConnectivityMatrix[i, j] = ConnectivityFunction(i, j)
    print ConnectivityMatrix[i, :]

# Assemble stiffness matrix
for elidx, el in enumerate(Elements):
    for i in range(0, order+1):
        phi_i = el.phi(i)
        gradphi_i = el.gradphi(i)
        T = el.T
        load = el.integrate_f(lambda x: phi_i(T(x))*f(x), 2)
        print load
        glob_i = int(ConnectivityMatrix[elidx][i])
        F[glob_i] += load
        for j in range(0, order+1):
            phi_j = el.phi(j)
            gradphi_j = el.gradphi(j)
            k = el.integrate_f(lambda x: phi_i(T(x))*b*phi_j(T(x)))
            k += -el.integrate_f(lambda x: gradphi_i(T(x))*el.jacobian(x)*(c - nu*gradphi_j(T(x))*el.jacobian(x)))
            glob_j = int(ConnectivityMatrix[elidx][j])
            I[coo_idx] = glob_i 
            J[coo_idx] = glob_j
            K[coo_idx] = k
            coo_idx += 1

# Apply boundary conditions
lambda_left = lambda_right = 1
left_bc_idx = ConnectivityMatrix[0][0]
left_flux_idx = Ne+1
I[coo_idx] = left_bc_idx
J[coo_idx] = left_flux_idx
K[coo_idx] = -1
coo_idx += 1
I[coo_idx] = left_flux_idx
J[coo_idx] = left_bc_idx
K[coo_idx] = -lambda_left
coo_idx += 1
F[left_flux_idx] = lambda_left*g0

right_bc_idx = ConnectivityMatrix[Ne-1][1]
right_flux_idx = Ne+2
I[coo_idx] = right_bc_idx
J[coo_idx] = right_flux_idx
K[coo_idx] = 1
coo_idx += 1
I[coo_idx] = right_flux_idx
J[coo_idx] = right_bc_idx
K[coo_idx] = lambda_right
coo_idx += 1
F[right_flux_idx] = lambda_right*g1

Kmat = sps.coo_matrix((K, (I, J)), shape=(Ndof, Ndof)).tocsc()
print Kmat.todense()
print F
v = spsolve(Kmat, F)
print v
print Kmat * v

# pylab.plot(np.linspace(0,1,Ndof), F)
pylab.plot(X, v[0:Ne+1])
pylab.plot(np.linspace(0,1), map(lambda x: (-x ** 4 + 13*x) / 12.0 , np.linspace(0,1)))
pylab.savefig("foo.png")

