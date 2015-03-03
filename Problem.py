import Element
import ConvectionDiffusionReactionElement
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg  import spsolve

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
        for i in range(0, self.Ne):
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

