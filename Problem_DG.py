#!/usr/bin/env python
import Element
import ConvectionDiffusionReactionElement
import numpy as np
import pylab
import matplotlib as mpl
import scipy.sparse as sps
from scipy.sparse.linalg  import spsolve
mpl.rc_file(r'mpl.rc')

class Face:
    def __init__(self, left_element, right_element):
        self.left = left_element
        self.right = right_element
        self.elements = (left_element, right_element)
        self.c = self.left.c
        self.nu = self.left.nu
        return

    # def calc_F_I(self, v_left, v_right):
        # return 0.5*(self.c * (v_right + v_left) - abs(c) * (v_right - v_left))
    def F_I_coeff(self, i):
        if (i == 0):
            # left
            return 0.5*(self.c + abs(c))
        elif (i == 1):
            # right
            return 0.5*(self.c - abs(c))
        else:
            return None

    def jump_coeff(self, i):
        if (i == 0):
            # left
            return 1
        elif (i == 1):
            # right
            return -1
        else:
            return None

    def avg_coeff(self, i):
        if (i == 0):
            # left
            return 0.5
        elif (i == 1):
            # right
            return 0.5
        else:
            return None

    def w_jump_coeff(self, i, j):
        if (i == j):
            return 1
        else:
            return 0

    def grad_coeff(self, i):
        if (i == 0):
            # left
            slo = self.left.order
            return self.left.calc_gradphi(slo, 1)*self.left.jacobian(self.left.T(1))
        elif (i == 1):
            # right 
            return self.right.calc_gradphi(0, -1)*self.right.jacobian(self.right.T(-1))
        return None

    def F_I(self):
        K = np.zeros([2, 2])
        for i in range(0, 2):
            for j in range(0, 2):
                K[i,j] = self.F_I_coeff(i)*self.jump_coeff(j)
        return K

    def consistency(self):
        K = np.zeros([2, 2])
        for i in range(0, 2):
            for j in range(0, 2):
                K[i,j] = -self.nu*self.avg_coeff(i)*self.grad_coeff(i)*self.jump_coeff(j)
                K[i,j] += -self.nu*self.avg_coeff(j)*self.grad_coeff(j)*self.jump_coeff(i)
        return K

    def stability(self):
        K = np.zeros([2, 2])
        for i in range(0, 2):
            for j in range(0, 2):
                K[i,j] = -2*(1+(self.left.order-1)/2.0)*self.nu*self.jump_coeff(i)*self.jump_coeff(j)
        return K

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
    Ne = 100
    Nf = Ne - 1
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
    Faces = [0]*(Nf)

    order = 1
    ConnectivityFunction = lambda element_idx, local_node_idx: int((order+1)*element_idx + local_node_idx)
    ConnectivityMatrix = np.empty([Ne, order+1], dtype=np.int_)
    BoundaryFluxDOFs = [(order+1)*Ne, (order+1)*Ne+1]
    Ndof = (order+1)*Ne+2

    splen = 4*Nf + Ndof*(order+1)**2
    I = np.zeros([splen])
    J = np.zeros([splen])
    K = np.zeros([splen])
    F = np.zeros([Ndof, 1])
    coo_idx = 0

    # Create triangulation
    for i in range(0, Ne):
        Elements[i] = ConvectionDiffusionReactionElement.Linear_1D(X[i], X[i+1], nu, b ,c)
        for j in range(0, order+1):
            ConnectivityMatrix[i, j] = ConnectivityFunction(i, j)

    for i in range(0, Nf):
        Faces[i] = Face(Elements[i], Elements[i+1])

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

    # Apply face fluxes
    for faidx, fa in enumerate(Faces):
        K_fa = fa.F_I() + fa.consistency() + fa.stability()
        gl_idx = ConnectivityMatrix[faidx][order]
        gr_idx = ConnectivityMatrix[faidx+1][0]
        I[coo_idx] = gl_idx
        J[coo_idx] = gl_idx
        K[coo_idx] = K_fa[0, 0]
        coo_idx += 1
        I[coo_idx] = gl_idx
        J[coo_idx] = gr_idx
        K[coo_idx] = K_fa[0, 1]
        coo_idx += 1
        I[coo_idx] = gr_idx
        J[coo_idx] = gl_idx
        K[coo_idx] = K_fa[1, 0]
        coo_idx += 1
        I[coo_idx] = gr_idx
        J[coo_idx] = gr_idx
        K[coo_idx] = K_fa[1, 1]
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
    print Kmat.todense()    
    print "Solving Matrix..."
    v = spsolve(Kmat, F)

    pylab.figure(figsize=(3,3))
    pylab.plot(np.linspace(0,1,400), map(problem['analytic_solution'], np.linspace(0,1,400)), linewidth=3.0, color='orange')
    for elidx, el in enumerate(Elements):
        Xel,Yel = el.interpxy(v[ConnectivityMatrix[elidx,:]])
        pylab.plot(Xel, Yel)
    print "Writing plots..."
    pylab.savefig(problem['shortdesc']+"_dg.png")
    L2e = map(lambda i: Elements[i].L2_error(v[ConnectivityMatrix[i, :]], problem['analytic_solution']), range(0, Ne))
    H1e = map(lambda i: Elements[i].H1_error(v[ConnectivityMatrix[i, :]], problem['analytic_solution'], problem['grad_analytic_solution']), range(0, Ne))
    print "L2 error", float(sum(L2e))
    print "H1 error", float(sum(H1e))
    print "Done."

