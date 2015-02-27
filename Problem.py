#!/usr/bin/env python
import Element
import ConvectionDiffusionReactionElement
import numpy as np
import pylab
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve

'''
from multiprocessing import Pool

def elk(el):
    return el.stiffness_matrix()

def elf(el):
    return el.load_vector(lambda x: 0)
'''

left = 0
right = 1
Ne = 40
h = (right - left) / Ne

nu = 1e-2
b = 0
c = 1.0
g0 = 0
g1 = 1
# f = lambda x: x ** 2
f = lambda x: 0

X = np.linspace(left, right, Ne+1)
H = np.diff(X)
Elements = [0]*(Ne)

order = 1
ConnectivityFunction = lambda element_idx, local_node_idx: int((order)*element_idx + local_node_idx)
ConnectivityMatrix = np.empty([Ne, 2], dtype=np.int_)
BoundaryFluxDOFs = [Ne+1, Ne+2]
Ndof = Ne+3

I = np.zeros([(2+Ndof)*(order+1)**2])
J = np.zeros([(2+Ndof)*(order+1)**2])
K = np.zeros([(2+Ndof)*(order+1)**2])
F = np.zeros([Ndof, 1])
coo_idx = 0

# Create triangulation
for i in range(0, Ne):
    Elements[i] = ConvectionDiffusionReactionElement.Linear_1D_VMS(X[i], X[i+1], nu, b ,c)
    for j in range(0, order+1):
        ConnectivityMatrix[i, j] = ConnectivityFunction(i, j)

# p = Pool(4)
# K_els = p.map(elk, Elements)
# F_els = p.map(elf, Elements)

# Assemble elemental stiffness matrices
print "Calculating Elemental Stiffness..."
for elidx, el in enumerate(Elements):
    K_el = el.stiffness_matrix()
    F_el = el.load_vector(f)
    # K_el = K_els[elidx]
    # F_el = F_els[elidx]
    for i in range(0, order+1):
        glob_i = int(ConnectivityMatrix[elidx][i])
        F[glob_i] += F_el[i]
        for j in range(0, order+1):
            glob_j = int(ConnectivityMatrix[elidx][j])
            I[coo_idx] = glob_i 
            J[coo_idx] = glob_j
            K[coo_idx] = K_el[i,j]
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
F[left_flux_idx] = -lambda_left*g0

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

# Assemble global stiffness matrix and solve system
print "Assembling Global Stiffness..."
Kmat = sps.coo_matrix((K, (I, J)), shape=(Ndof, Ndof)).tocsc()
print "Solving Matrix..."
v = spsolve(Kmat, F)

# pylab.plot(np.linspace(0,1,Ndof), F)
# pylab.plot(np.linspace(0,1), map(lambda x: (-x ** 4 + 13*x) / 12.0 , np.linspace(0,1)), linewidth=3.0, color='orange')

# char_root = np.sqrt(b / nu)
# pylab.plot(np.linspace(0,1,400), map(lambda x: np.sinh(char_root * x) / np.sinh(char_root) , np.linspace(0,1,400)))

# char_root = c / nu
# pylab.plot(np.linspace(0,1,400), map(lambda x: (1.0 - np.exp(char_root * x)) / (1.0 - np.exp(char_root)) , np.linspace(0,1,400)))
for elidx, el in enumerate(Elements):
    Xel,Yel = el.interpxy(v[ConnectivityMatrix[elidx,:]])
    pylab.plot(Xel, Yel)
# pylab.plot(X, v[0:Ne+1])
print "Writing plots..."
pylab.savefig("foo.png")
print "Done."

