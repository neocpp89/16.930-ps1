import numpy
import gauss

class _1D:
    def __init__(self, x0, x1): 
        self.x0 = x0
        self.x1 = x1
        self.h = (x1 - x0)
        return

    def T(self, x):
        return 2.0 * ((x - self.x0) / self.h) - 1.0

    def T_inv(self, xi):
        return self.x0 + (self.h * (xi + 1.0) / 2.0)

    def jacobian(self, x):
        return 2.0 / self.h

class Linear_1D(_1D):
    def integrate_f(self, f, order_hint=10):
        return self.integrate_on_master(lambda xi: f(self.T_inv(xi))/self.jacobian(self.T_inv(xi)), order_hint)
    def integrate(self, u, v, order_hint=10):
        #  return gauss.quadrature(lambda x: u(x)*v(x), self.x0, self.x1, order_hint)
        return self.integrate_on_master(lambda x: u(x)*v(x)/self.jacobian(x), order_hint=order_hint)

    def integrate_on_master(self, f, order_hint=10):
        return gauss.quadrature(lambda xi: f(xi), -1, 1, order_hint)

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

    def __repr__(self):
        return "x = (" + str(self.x0) + ',' + str(self.x1) + ")"
