import Element
import numpy as np

class _1D:
    def __init__(self, left_element, right_element):
        self.left = left_element
        self.right = right_element
        self.elements = (left_element, right_element)
        self.order = self.left.order
        self.c = self.left.c
        self.nu = self.left.nu
        self.dofs = 2*(self.order+1)
        return

    def calc_w_lr(self, i):
        if (i <= self.order):
            # inside left element, so w+ is  0
            w_plus = 0
            w_minus = self.left.calc_phi(i, 1)
        else:
            # inside right element, so w- is 0
            w_minus = 0
            w_plus = self.right.calc_phi(i - self.order-1, -1)
        return (w_minus, w_plus)

    def calc_gradw_lr(self, i):
        if (i <= self.order):
            # inside left element, so w+ is  0
            w_plus = 0
            w_minus = self.left.calc_gradphi(i, 1)*self.left.jacobian(self.left.T_inv(1))
        else:
            # inside right element, so w- is 0
            w_minus = 0
            w_plus = self.right.calc_gradphi(i - self.order-1, -1)*self.right.jacobian(self.right.T_inv(-1))
        return (w_minus, w_plus)

    def calc_rf(self, i):
        if (self.order == 3):
            return map(lambda x: -8, self.calc_w_lr(i))
        elif (self.order == 1):
            return map(lambda x: -2, self.calc_w_lr(i))
        return None

    def w_jump_coeffs(self):
        w_jump = np.zeros([self.dofs])
        for i in range(0, self.dofs):
            wm, wp = self.calc_w_lr(i)
            w_jump[i] = wm - wp
        return w_jump

    def w_avg_coeffs(self):
        w_avg = np.zeros([self.dofs])
        for i in range(0, self.dofs):
            wm, wp = self.calc_w_lr(i)
            w_avg[i] = 0.5 * (wm + wp)
        return w_avg

    def gradw_avg_coeffs(self):
        gradw_avg = np.zeros([self.dofs])
        for i in range(0, self.dofs):
            dphim, dphip = self.calc_gradw_lr(i)
            gradw_avg[i] = 0.5 * (dphim + dphip)
        return gradw_avg

    def rf_avg_coeffs(self):
        rf = np.zeros([self.dofs])
        for i in range(0, self.dofs):
            rfl, rfr = self.calc_rf(i)
            rf[i] = 0.5 * (rfl + rfr)
        return rf

    def F_I(self):
        phi_i = self.w_jump_coeffs()
        phi_j = (self.c*self.w_avg_coeffs() + 0.5*abs(self.c)*self.w_jump_coeffs())
        K = np.outer(phi_i, phi_j)
        return K

    def K_consistent(self):
        dphi_i = self.nu*self.gradw_avg_coeffs()
        phi_j = self.w_jump_coeffs()
        K = -(np.outer(dphi_i, phi_j) + np.outer(phi_j, dphi_i))
        return K

    def K_stable(self, eta=2):
        K = -eta*np.outer(self.w_jump_coeffs(), self.nu*self.rf_avg_coeffs()*self.w_jump_coeffs())
        return K

    def K(self):
        K_full = self.F_I()  + self.K_consistent() + self.K_stable()
        return K_full

