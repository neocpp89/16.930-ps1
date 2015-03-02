import numpy
import gauss

class _1D:
    def __init__(self, x0, x1, order): 
        self.x0 = float(x0)
        self.x1 = float(x1)
        self.h = float(x1 - x0)
        self.order = order
        self.w = numpy.zeros([order+1, 1])
        self.xipts = numpy.linspace(-1, 1, order+1)
        for i in range(0, order+1):
            prod = 1
            for j in range(0, order+1):
                if (i != j):
                    prod *= (self.xipts[i] - self.xipts[j])
            self.w[i] = 1.0 / prod
        return

    def T(self, x):
        return (2.0 * ((x - self.x0) / self.h) - 1.0)

    def T_inv(self, xi):
        return (self.x0 + (self.h * (xi + 1.0) / 2.0))

    def jacobian(self, x):
        return (2.0 / self.h)

    def calc_phi(self, local_node, xi):
        prod = 1.0
        for i in range(0, self.order+1):
            if (i != local_node):
                prod *= (xi - self.xipts[i])
        prod *= self.w[local_node]
        return prod

    def phi(self, local_node):
        return lambda xi: self.calc_phi(local_node, xi) 
        
    def calc_gradphi(self, local_node, xi):
        s = 0.0
        for i in range(0, self.order+1):
            if (i != local_node):
                prod = 1.0
                for j in range(0, self.order+1):
                    if (i != j and j != local_node):
                        prod *= (xi - self.xipts[j])
                s += prod
        s *= self.w[local_node]
        return s

    def gradphi(self, local_node):
        return lambda xi: self.calc_gradphi(local_node, xi)

    def calc_gradsqphi(self, local_node, xi):
        if (self.order < 2):
            return 0 
        s = 0.0
        for i in range(0, self.order+1):
            if (i != local_node):
                for j in range(0, self.order+1):
                    if (i != j and j != local_node):
                        prod = 1.0
                        for k in range(0, self.order+1):
                            if (k != i and k != j and k != local_node):
                                prod *= (xi - self.xipts[k])
                        s += prod
        s *= self.w[local_node]
        return s

    def gradsqphi(self, local_node):
        return lambda xi: self.calc_gradsqphi(local_node, xi)

    def calc_p(self, v, xi):
        if (len(v) != self.order+1):
            print "Can't interpolate data (wrong coefficient length)."
            return None
        num = 0.0
        den = 0.0
        for i in range(0, self.order+1):
            if (xi - self.xipts[i] == 0):
                return v[i]
            num += ((self.w[i] * v[i]) / (xi - self.xipts[i]))
            den += (self.w[i] / (xi - self.xipts[i]))
        return float(num/den)

    def L2_norm(self, f):
        return numpy.sqrt(self.integrate(lambda x: f(x)*f(x)))

    def H1_norm(self, f, df):
        return numpy.sqrt(self.integrate(lambda x: f(x)*f(x) + df(x)*df(x)))

    # Compare a function f to the interpolation on this element with given
    # nodal coefficients.
    def L2_error(self, node_coeffs, f):
        g = lambda x: self.calc_p(node_coeffs, self.T(x)) - f(x)
        return self.L2_norm(g)

    # Compare a function f to the interpolation on this element with given
    # nodal coefficients.
    def H1_error(self, node_coeffs, f, df):
        # this is slow, but easy for me to reason about (no d(calc_p)/dx)
        g = lambda x: self.calc_p(node_coeffs, self.T(x)) - f(x)
        dg = lambda x: sum(map(lambda i: node_coeffs[i]*self.calc_gradphi(i, self.T(x))*self.jacobian(x), range(0, self.order+1))) - df(x)
        return self.H1_norm(g, dg)

    def interpxy(self, node_coeffs, npts=20):
        if (len(node_coeffs) != self.order+1):
            print "Can't interpolate data (wrong coefficient length)."
            return None
        lower = self.x0
        upper = self.x1
        X = numpy.linspace(lower, upper, npts)
        Y = map(lambda x: self.calc_p(node_coeffs, self.T(x)), X)
        return (X, Y)

    # NOTE: f should be a function of x, not xi!
    # It is automatically transformed into a function of xi and integrated on
    # the master.
    def integrate(self, f, order_hint=10):
        return self.integrate_on_master(lambda xi: f(self.T_inv(xi))/self.jacobian(self.T_inv(xi)), order_hint)

    # NOTE: f is integrated on the master, so xi and x coordinates are the same
    def integrate_on_master(self, f, order_hint):
        return gauss.quadrature(f, -1, 1, order_hint)

    def __repr__(self):
        return "x_pts = (" + ", ".join(map(lambda xi: str(self.T_inv(xi)), self.xipts)) + ")"

class Linear_1D(_1D):
    order = 1
    def __init__(self, x0, x1):
        _1D.__init__(self, x0, x1, 1)
        return

class Cubic_1D(_1D):
    order = 3
    def __init__(self, x0, x1):
        _1D.__init__(self, x0, x1, 3)
        return

