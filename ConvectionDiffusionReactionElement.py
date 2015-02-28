import numpy as np 
import gauss
import Element

class Linear_1D(Element.Linear_1D):
    def __init__(self, x0, x1, nu, b, c):
        Element.Linear_1D.__init__(self, x0, x1)
        self.nu = nu
        self.c = c
        self.b = b
        self.order = 1
        return

    # Returns the (dense) stiffness matrix
    def stiffness_matrix(self):
        K = np.zeros([2,2])
        for i in range(0, self.order+1):
            phi_i = self.phi(i)
            gradphi_i = self.gradphi(i)
            T = self.T
            for j in range(0, self.order+1):
                phi_j = self.phi(j)
                gradphi_j = self.gradphi(j)
                f = lambda x: (phi_i(T(x))*self.b*phi_j(T(x)) -
                    gradphi_i(T(x))*self.jacobian(x) *
                    (self.c*phi_j(T(x)) - self.nu*gradphi_j(T(x))*self.jacobian(x)))
                k = self.integrate(f)
                K[i, j] += k
        return K

    # returns the dense load vector given function 'f'
    def load_vector(self, f):
        F = np.zeros([2, 1])
        for i in range(0, self.order+1):
            phi_i = self.phi(i)
            gradphi_i = self.gradphi(i)
            T = self.T
            F[i, 0] = self.integrate(lambda x: phi_i(T(x))*f(x), 2)
        return F

class Linear_1D_VMS(Element.Linear_1D):
    def __init__(self, x0, x1, nu, b, c):
        Element.Linear_1D.__init__(self, x0, x1)
        self.nu = nu
        self.c = c
        self.b = b
        self.order = 1
        return

    # Returns the (dense) stiffness matrix
    def stiffness_matrix(self):
        c = abs(self.c)
        if (c != 0):
            pe = self.h * c / (2.0 * self.nu) 
            tau = (self.h / (2.0 * c)) * (np.cosh(pe)/np.sinh(pe) - 1.0/pe)
        else:
            tau = self.h * self.h / (12.0 * self.nu)
        K = np.zeros([2,2])
        for i in range(0, self.order+1):
            phi_i = self.phi(i)
            gradphi_i = self.gradphi(i)
            T = self.T
            for j in range(0, self.order+1):
                phi_j = self.phi(j)
                gradphi_j = self.gradphi(j)
                f = lambda x: (phi_i(T(x))*self.b*phi_j(T(x)) -
                    gradphi_i(T(x))*self.jacobian(x) *
                    (self.c*phi_j(T(x)) - self.nu*gradphi_j(T(x))*self.jacobian(x)))
                k = self.integrate(f)
                vms = lambda x: ((self.c*self.jacobian(x)*gradphi_i(T(x)) - self.b*phi_i(T(x))) *
                    tau*(self.c*self.jacobian(x)*gradphi_j(T(x)) + self.b*phi_j(T(x))))
                k_stab = self.integrate(vms)
                K[i, j] += (k+k_stab)
        return K

    # returns the dense load vector given function 'f'
    def load_vector(self, f):
        F = np.zeros([2, 1])
        for i in range(0, self.order+1):
            phi_i = self.phi(i)
            gradphi_i = self.gradphi(i)
            T = self.T
            F[i, 0] = self.integrate(lambda x: phi_i(T(x))*f(x), 2)
        return F

class Cubic_1D(Element.Cubic_1D):
    def __init__(self, x0, x1, nu, b, c):
        Element.Cubic_1D.__init__(self, x0, x1)
        self.nu = nu
        self.c = c
        self.b = b
        return

    # Returns the (dense) stiffness matrix
    def stiffness_matrix(self):
        K = np.zeros([self.order+1,self.order+1])
        for i in range(0, self.order+1):
            phi_i = self.phi(i)
            gradphi_i = self.gradphi(i)
            T = self.T
            for j in range(0, self.order+1):
                phi_j = self.phi(j)
                gradphi_j = self.gradphi(j)
                f = lambda x: (phi_i(T(x))*self.b*phi_j(T(x)) -
                    gradphi_i(T(x))*self.jacobian(x) *
                    (self.c*phi_j(T(x)) - self.nu*gradphi_j(T(x))*self.jacobian(x)))
                K[i, j] = self.integrate(f)
        return K

    # returns the dense load vector given function 'f'
    def load_vector(self, f):
        F = np.zeros([self.order+1, 1])
        for i in range(0, self.order+1):
            phi_i = self.phi(i)
            gradphi_i = self.gradphi(i)
            T = self.T
            F[i, 0] = self.integrate(lambda x: phi_i(T(x))*f(x))
        return F
