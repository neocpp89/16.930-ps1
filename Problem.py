#!/usr/bin/env python
import Element
import ConvectionDiffusionReactionElement
import numpy as np
import pylab
import matplotlib as mpl
import scipy.sparse as sps
from scipy.sparse.linalg  import spsolve
import os
import errno
import pickle

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return

mpl.rc_file(r'mpl.rc')

class Problem:
    def __init__(self, problem_description, element, Ne, np=1):
        self.params = problem_description
        self.ElementType = element
        self.Ne = Ne
        if (np > 1):
            self.parallel = True
            self.nproc = np
        else:
            self.parallel = False
            self.nproc = 1
        return

    def CreateTriangulation(self, verbose=False):
        if verbose:
            print "Creating Triangulation...",
        X = np.linspace(self.params['left'], self.params['right'], self.Ne+1)
        self.Elements = [0]*(self.Ne)

        order = self.ElementType.order
        ConnectivityFunction = lambda element_idx, local_node_idx: int((order*element_idx) + local_node_idx)
        self.ConnectivityMatrix = np.empty([self.Ne, order+1], dtype=np.int_)
        self.BoundaryFluxDOFs = [(order*self.Ne)+1, (order*self.Ne)+2]
        self.Ndof = order*self.Ne+3

        splen = self.Ndof*(order+1)**2
        self.I = np.zeros([splen])
        self.J = np.zeros([splen])
        self.K = np.zeros([splen])
        self.F = np.zeros([self.Ndof])
        self.spi = 0

        # Acutally create triangulation
        for i in range(0, Ne):
            self.Elements[i] = self.ElementType(X[i],
                X[i+1], self.params['nu'], self.params['b'], self.params['c'])
            for j in range(0, order+1):
                self.ConnectivityMatrix[i, j] = ConnectivityFunction(i, j)
        if verbose:
            print "Done."
        return

    def CalculateElementStiffnesses(self, verbose=False):
        if verbose:
            print "Calculating Element Stiffness...",
        for elidx, el in enumerate(self.Elements):
            K_el = el.stiffness_matrix()
            F_el = el.load_vector(self.params['f'])
            for i in range(0, self.ElementType.order+1):
                glob_i = int(self.ConnectivityMatrix[elidx][i])
                self.F[glob_i] += F_el[i]
                for j in range(0, self.ElementType.order+1):
                    glob_j = int(self.ConnectivityMatrix[elidx][j])
                    self.I[self.spi] = glob_i 
                    self.J[self.spi] = glob_j
                    self.K[self.spi] = K_el[i,j]
                    self.spi += 1
        if verbose:
            print "Done..."
        return

    def ApplyBoundaryConditions(self):
        left_dirchlet_value = self.params['g0']
        right_dirchlet_value = self.params['g1']
        lambda_left = lambda_right = 1
        left_bc_idx = self.ConnectivityMatrix[0][0]
        left_flux_idx = self.BoundaryFluxDOFs[0]
        self.I[self.spi] = left_bc_idx
        self.J[self.spi] = left_flux_idx
        self.K[self.spi] = -1
        self.spi += 1
        self.I[self.spi] = left_flux_idx
        self.J[self.spi] = left_bc_idx
        self.K[self.spi] = -lambda_left
        self.spi += 1
        self.F[left_flux_idx] = -lambda_left*left_dirchlet_value

        right_bc_idx = self.ConnectivityMatrix[self.Ne-1][self.ElementType.order]
        right_flux_idx = self.BoundaryFluxDOFs[1]
        self.I[self.spi] = right_bc_idx
        self.J[self.spi] = right_flux_idx
        self.K[self.spi] = 1
        self.spi += 1
        self.I[self.spi] = right_flux_idx
        self.J[self.spi] = right_bc_idx
        self.K[self.spi] = lambda_right
        self.spi += 1
        self.F[right_flux_idx] = lambda_right*right_dirchlet_value
        return

    def Assemble(self, verbose=False):
        if verbose:
            print "Assembling Global Stiffness...",
        self.Kmat = sps.coo_matrix((self.K, (self.I, self.J)), shape=(self.Ndof, self.Ndof)).tocsc()
        if verbose:
            print "Done"
        return

    def Solve(self, verbose=False):
        if verbose:
            print "Solving system of equations...",
        self.V = spsolve(self.Kmat, self.F)
        if verbose:
            print "Done"
        return self.V

    def CalculateErrors(self):
        L2_error = sum(map(lambda i: self.Elements[i].L2_error(self.V[self.ConnectivityMatrix[i, :]], self.params['analytic_solution']), range(0, self.Ne)))
        H1_error = sum(map(lambda i: self.Elements[i].H1_error(self.V[self.ConnectivityMatrix[i, :]], self.params['analytic_solution'], self.params['grad_analytic_solution']), range(0, self.Ne)))
        return map(float, (L2_error, H1_error))

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

ElementTypes = [
    ConvectionDiffusionReactionElement.Linear_1D,
    ConvectionDiffusionReactionElement.Linear_1D_VMS,
    ConvectionDiffusionReactionElement.Cubic_1D,
    ConvectionDiffusionReactionElement.Cubic_1D_VMS,
]

NumElements = [10, 20, 40, 80, 100, 160, 200, 320, 400]

errors = {}

for problem in problems:
    errors[problem['desc']] = {}

    for ET in ElementTypes:
        errors[problem['desc']][ET.desc] = {}

        for Ne in NumElements:
            p = Problem(problem, ET, Ne)
            print "Problem type:", p.params['desc'], "Element type:", p.ElementType.desc, "Ne:", p.Ne
            p.CreateTriangulation()
            p.CalculateElementStiffnesses()
            p.ApplyBoundaryConditions()
            p.Assemble()
            v = p.Solve()
            L2e, H1e = p.CalculateErrors()
            errors[problem['desc']][ET.desc][Ne] = (L2e, H1e)

            f = pylab.figure(figsize=(3,3))
            pylab.plot(np.linspace(0,1,400), map(p.params['analytic_solution'], np.linspace(0,1,400)), linewidth=3.0, color='orange')
            for elidx, el in enumerate(p.Elements):
                Xel,Yel = el.interpxy(v[p.ConnectivityMatrix[elidx,:]])
                pylab.plot(Xel, Yel)
            print "Writing plots..."
            pylab.xlabel('Spatial Coordinate X')
            pylab.ylabel('Solution V')
            esplit = p.ElementType.desc.split(' ')
            etype = esplit[0]
            if (len(esplit) > 1):
                estab = " ".join(esplit[1:])
            else:
                estab = ""
            pylab.title(p.params['desc'] + "\nusing " + str(p.Ne) + " " + etype + " Elements " + estab)
            pylab.tight_layout(pad=0.1)
            make_sure_path_exists('report/figs')
            pylab.savefig('report/figs/'+p.params['shortdesc']+"_"+p.ElementType.shortdesc+"_"+str(p.Ne)+".png")
            print "L2 error", L2e
            print "H1 error", H1e
            print "Done."
            pylab.close(f)


# output errors in a latex-friendly table using siunitx's '\num'
for problem in problems:
    print r'\begin{table}[!h]'
    print r'\centering'
    print r'\caption{'+problem['desc']+r': $L_2$ Error (CG methods)}'
    print r'\begin{tabular}{r | ' + ' '.join(['r']*len(ElementTypes)) + r'}'
    print r'Number of Elements & ' + ' & '.join(map(lambda ET: ET.desc, ElementTypes)) + r' \\'
    print r'\midrule'
    for Ne in NumElements:
        print r'\num{{{:d}}} &'.format(Ne),
        print ' & '.join(map(lambda ET: r'\num{{{:.4e}}}'.format(errors[problem['desc']][ET.desc][Ne][0]), ElementTypes)) + r' \\'
    print r'\end{tabular}'
    print r'\end{table}'

# save errors if we want to do something later
pickle.dump(errors, open('error_pickle.p', 'wb'))
