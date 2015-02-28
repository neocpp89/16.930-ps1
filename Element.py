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
                prod = 1.0
                for j in range(0, self.order+1):
                    if (i != j and j != local_node):
                        prod *= (xi - self.xipts[j])
                s += prod
        s *= self.w[local_node]
        return s

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


    def interpxy(self, node_coeffs, npts=20):
        if (len(node_coeffs) != self.order+1):
            print "Can't interpolate data (wrong coefficient length)."
            return None
        lower = self.x0
        upper = self.x1
        X = numpy.linspace(lower, upper, npts)
        Y = map(lambda x: self.calc_p(node_coeffs, self.T(x)), X)
        return (X, Y)

class Linear_1D(_1D):
    order = 1
    def __init__(self, x0, x1):
        _1D.__init__(self, x0, x1, 1)
        return

    def integrate(self, f, order_hint=10):
        return self.integrate_on_master(lambda xi: f(self.T_inv(xi))/self.jacobian(self.T_inv(xi)), order_hint)

    def integrate_on_master(self, f, order_hint=10):
        return gauss.quadrature(lambda xi: f(xi), -1, 1, order_hint)

    '''
    def phi(self, local_node_idx):
        if (local_node_idx == 0):
            return lambda xi: (1.0 - xi) / 2.0
        elif (local_node_idx == 1):
            return lambda xi: (1.0 + xi) / 2.0
        return None;

    def gradphi(self, local_node_idx):
        if (local_node_idx == 0):
            return lambda xi: -1.0 / 2.0
        elif (local_node_idx == 1):
            return lambda xi: 1.0 / 2.0
        return None;
    '''

    def __repr__(self):
        return "x = (" + str(self.x0) + ',' + str(self.x1) + ")"

class Cubic_1D(_1D):
    order = 3
    def __init__(self, x0, x1):
        _1D.__init__(self, x0, x1, 3)
        return

    def integrate(self, f, order_hint=10):
        return self.integrate_on_master(lambda xi: f(self.T_inv(xi))/self.jacobian(self.T_inv(xi)), order_hint)

    def integrate_on_master(self, f, order_hint=10):
        return gauss.quadrature(lambda xi: f(xi), -1, 1, order_hint)

    def __repr__(self):
        return "x = (" + str(self.x0) + ',' + str(self.x1) + ")"
