#!/usr/bin/env python
import Element
import Face
import ConvectionDiffusionReactionElement
import numpy as np
import pylab
import matplotlib as mpl
import scipy.sparse as sps
from scipy.sparse.linalg  import spsolve
import os
import errno
import pickle
import Problem
mpl.rc_file(r'./mpl.rc')

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

class Problem_DG(Problem.Problem):
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
        self.Nf = Ne-1
        X = np.linspace(self.params['left'], self.params['right'], self.Ne+1)
        self.Elements = [0]*(self.Ne)
        self.Faces = [0]*(self.Nf)

        order = self.ElementType.order
        ConnectivityFunction = lambda element_idx, local_node_idx: int((order+1)*element_idx + local_node_idx)
        self.ConnectivityMatrix = np.empty([self.Ne, order+1], dtype=np.int_)
        self.BoundaryFluxDOFs = [(order+1)*self.Ne, (order+1)*self.Ne+1]
        self.Ndof = (order+1)*self.Ne+2

        splen = (4*self.Nf + self.Ndof)*(order+1)**2 
        self.I = np.zeros([splen])
        self.J = np.zeros([splen])
        self.K = np.zeros([splen])
        self.F = np.zeros([self.Ndof])
        self.spi = 0

        # Acutally create triangulation
        for i in range(0, self.Ne):
            self.Elements[i] = self.ElementType(X[i],
                X[i+1], self.params['nu'], self.params['b'], self.params['c'])
            for j in range(0, order+1):
                self.ConnectivityMatrix[i, j] = ConnectivityFunction(i, j)
        for i in range(0, self.Nf):
            self.Faces[i] = Face._1D(self.Elements[i], self.Elements[i+1])

        if verbose:
            print "Done."
        return

    def CalculateFaceStiffnesses(self, verbose=False):
        if verbose:
            print "Calculating Face Stiffness...",
        for faidx, fa in enumerate(self.Faces):
            K_fa = fa.K()
            GL = self.ConnectivityMatrix[faidx, :]
            GR = self.ConnectivityMatrix[faidx+1, :]
            SS = np.concatenate((GL, GR))
            GL, GR = (SS, SS)
            for i, gl_idx in enumerate(GL):
                for j, gr_idx in enumerate(GR):
                    self.I[self.spi] = gl_idx
                    self.J[self.spi] = gr_idx
                    self.K[self.spi] = K_fa[i, j]
                    self.spi += 1
        if verbose:
            print "Done..."
        return


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

params = [
    poisson_problem,
    reaction_diffusion_problem,
    convection_diffusion_problem
]

ElementTypes = [
    ConvectionDiffusionReactionElement.Linear_1D,
    ConvectionDiffusionReactionElement.Cubic_1D,
]

NumElements = [10, 20, 40, 80, 100, 160, 200, 320, 400]

errors = {}

for param in params:
    errors[param['desc']] = {}

    for ET in ElementTypes:
        errors[param['desc']][ET.desc] = {}

        for Ne in NumElements:
            p = Problem_DG(param, ET, Ne)
            print "Problem type:", p.params['desc'], "Element type:", p.ElementType.desc, "Ne:", p.Ne
            p.CreateTriangulation()
            p.CalculateElementStiffnesses()
            p.CalculateFaceStiffnesses()
            p.ApplyBoundaryConditions()
            p.Assemble()
            v = p.Solve()
            L2e, H1e = p.CalculateErrors()
            errors[param['desc']][ET.desc][Ne] = (L2e, H1e)

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
            pylab.savefig('report/figs/dg_'+p.params['shortdesc']+"_"+p.ElementType.shortdesc+"_"+str(p.Ne)+".png")
            print "L2 error", L2e
            print "H1 error", H1e
            print "Done."
            pylab.close(f)

# save errors if we want to do something later
pickle.dump(errors, open('dg_error_pickle.p', 'wb'))

