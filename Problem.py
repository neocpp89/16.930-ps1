#!/usr/bin/env python
import Element
import ConvectionDiffusionReactionElement
import numpy as np
import pylab
import matplotlib as mpl
import scipy.sparse as sps
from scipy.sparse.linalg  import spsolve
mpl.rc_file(r'mpl.rc')

'''
from multiprocessing import Pool

def elk(el):
    return el.stiffness_matrix()

def elf(el):
    return el.load_vector(lambda x: 0)
'''

poisson_problem = {
    'left': 0,
    'right': 1,
    'nu': 1,
    'b': 0.0,
    'c': 0.0,
    'g0': 0,
    'g1': 1,
    'f': lambda x: x ** 2,
    'analytic_solution': lambda x: (-x ** 4 + 13.0*x) / 12.0,
    'grad_analytic_solution': lambda x: (-4.0*(x ** 3) + 13.0) / 12.0,
    'desc': 'Poisson Equation',
    'shortdesc': 'poisson'
}

reaction_diffusion_problem = {
    'left': 0,
    'right': 1,
    'nu': 1e-4,
    'b': 1.0,
    'c': 0.0,
    'g0': 0,
    'g1': 1,
    'f': lambda x: 0,
    'analytic_solution': lambda x: np.sinh(100.0 * x) / np.sinh(100.0),
    'grad_analytic_solution': lambda x: 100.0 *np.cosh(100.0 * x) / np.sinh(100.0),
    'desc': 'Reaction-Diffusion Equation',
    'shortdesc': 'reaction_diffusion'
}

convection_diffusion_problem = {
    'left': 0,
    'right': 1,
    'nu': 1e-2,
    'b': 0.0,
    'c': 1.0,
    'g0': 0,
    'g1': 1,
    'f': lambda x: 0,
    'analytic_solution': lambda x: (1.0 - np.exp(100* x)) / (1.0 - np.exp(100)),
    'grad_analytic_solution': lambda x: -100 * np.exp(100* x) / (1.0 - np.exp(100)),
    'desc': 'Convection-Diffusion Equation',
    'shortdesc': 'convection_diffusion'
}

problems = [
    poisson_problem,
    reaction_diffusion_problem,
    convection_diffusion_problem
]

for problem in problems:
    left = problem['left']
    right = problem['right']
    Ne = 20
    h = (right - left) / Ne

    nu = problem['nu']
    b = problem['b']
    c = problem['c']
    g0 = problem['g0']
    g1 = problem['g1']
    f = problem['f']

    ElementType = [
        ConvectionDiffusionReactionElement.Linear_1D,
        ConvectionDiffusionReactionElement.Linear_1D_VMS,
        ConvectionDiffusionReactionElement.Cubic_1D,
    ]

    X = np.linspace(left, right, Ne+1)
    Elements = [0]*(Ne)

    order = 3
    ConnectivityFunction = lambda element_idx, local_node_idx: int((order*element_idx) + local_node_idx)
    ConnectivityMatrix = np.empty([Ne, order+1], dtype=np.int_)
    BoundaryFluxDOFs = [(order*Ne)+1, (order*Ne)+2]
    Ndof = order*Ne+3

    I = np.zeros([Ndof*(order+1)**2])
    J = np.zeros([Ndof*(order+1)**2])
    K = np.zeros([Ndof*(order+1)**2])
    F = np.zeros([Ndof, 1])
    coo_idx = 0

    # Create triangulation
    for i in range(0, Ne):
        Elements[i] = ConvectionDiffusionReactionElement.Cubic_1D_VMS(X[i], X[i+1], nu, b ,c)
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
    left_flux_idx = BoundaryFluxDOFs[0]
    I[coo_idx] = left_bc_idx
    J[coo_idx] = left_flux_idx
    K[coo_idx] = -1
    coo_idx += 1
    I[coo_idx] = left_flux_idx
    J[coo_idx] = left_bc_idx
    K[coo_idx] = -lambda_left
    coo_idx += 1
    F[left_flux_idx] = -lambda_left*g0

    right_bc_idx = ConnectivityMatrix[Ne-1][order]
    right_flux_idx = BoundaryFluxDOFs[1]
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

    pylab.figure(figsize=(3,3))
    pylab.plot(np.linspace(0,1,400), map(problem['analytic_solution'], np.linspace(0,1,400)), linewidth=3.0, color='orange')
    for elidx, el in enumerate(Elements):
        Xel,Yel = el.interpxy(v[ConnectivityMatrix[elidx,:]])
        pylab.plot(Xel, Yel)
    print "Writing plots..."
    pylab.savefig(problem['shortdesc']+".png")
    L2e = map(lambda i: Elements[i].L2_error(v[ConnectivityMatrix[i, :]], problem['analytic_solution']), range(0, Ne))
    H1e = map(lambda i: Elements[i].H1_error(v[ConnectivityMatrix[i, :]], problem['analytic_solution'], problem['grad_analytic_solution']), range(0, Ne))
    print "L2 error", float(sum(L2e))
    print "H1 error", float(sum(H1e))
    print "Done."

