#!/usr/bin/env python
import unittest
import numpy
import Element
import ConvectionDiffusionReactionElement

class Test_Linear_1D(unittest.TestCase):
    def setUp(self):
       return

    def test_T_master(self):
        el = Element.Linear_1D(-1, 1)
        self.assertAlmostEqual(el.T(-1), -1)
        self.assertAlmostEqual(el.T(-0.5), -0.5)
        self.assertAlmostEqual(el.T(0), 0)
        self.assertAlmostEqual(el.T(0.5), 0.5)
        self.assertAlmostEqual(el.T(1), 1)
        self.assertAlmostEqual(el.w[0], -0.5)
        self.assertAlmostEqual(el.w[1], 0.5)
        self.assertAlmostEqual(el.xipts[0], -1)
        self.assertAlmostEqual(el.xipts[1], 1)
        return

    def test_T_arbitrary(self):
        el = Element.Linear_1D(0.5, 0.7)
        self.assertAlmostEqual(el.T(0.5), -1)
        self.assertAlmostEqual(el.T(0.6), 0)
        self.assertAlmostEqual(el.T(0.7), 1)
        self.assertAlmostEqual(el.w[0], -0.5)
        self.assertAlmostEqual(el.w[1], 0.5)
        self.assertAlmostEqual(el.xipts[0], -1)
        self.assertAlmostEqual(el.xipts[1], 1)
        return

    def test_jacobian_master(self):
        el = Element.Linear_1D(-1, 1)
        self.assertAlmostEqual(el.jacobian(0), 1.0)
        return

    def test_jacobian_arbitrary(self):
        el = Element.Linear_1D(-23, 51)
        self.assertAlmostEqual(el.jacobian(0), 2.0 / 74.0)
        return

    def test_integrate_master(self):
        el = Element.Linear_1D(-1, 1)
        I = el.integrate(lambda x: 1)
        self.assertAlmostEqual(I, 2.0)
        I = el.integrate(lambda x: x**2)
        self.assertAlmostEqual(I, 2.0 / 3.0)
        return

    def test_integrate_arbitrary(self):
        el = Element.Linear_1D(1.0, 1.5)
        I = el.integrate(lambda x: 2)
        self.assertAlmostEqual(I, 1.0)
        I = el.integrate(lambda x: x ** 3)
        self.assertAlmostEqual(I, ((1.5)**4 - 1) / 4.0)
        return

class Test_CDR_Linear_1D(unittest.TestCase):
    def setUp(self):
        return

    def test_stiffness_diffusion(self):
        el = ConvectionDiffusionReactionElement.Linear_1D(0, 0.25, 1, 0, 0);
        K = el.stiffness_matrix()
        self.assertAlmostEqual(K[0,0], 4.0)
        self.assertAlmostEqual(K[0,1], -4.0)
        self.assertAlmostEqual(K[1,0], -4.0)
        self.assertAlmostEqual(K[1,1], 4.0)
        return

    def test_stiffness_reaction(self):
        el = ConvectionDiffusionReactionElement.Linear_1D(0, 0.25, 0, 1, 0);
        K = el.stiffness_matrix()
        self.assertAlmostEqual(K[0,0], 1.0/12.0)
        self.assertAlmostEqual(K[0,1], 1.0/24.0)
        self.assertAlmostEqual(K[1,0], 1.0/24.0)
        self.assertAlmostEqual(K[1,1], 1.0/12.0)
        return

    def test_stiffness_convection(self):
        el = ConvectionDiffusionReactionElement.Linear_1D(0, 0.25, 0, 0, 1);
        K = el.stiffness_matrix()
        self.assertAlmostEqual(K[0,0], 0.5)
        self.assertAlmostEqual(K[0,1], 0.5)
        self.assertAlmostEqual(K[1,0], -0.5)
        self.assertAlmostEqual(K[1,1], -0.5)
        return

    def test_load(self):
        el = ConvectionDiffusionReactionElement.Linear_1D(0, 0.25, 1, 0, 0);
        F = el.load_vector(lambda x: x ** 2)
        self.assertAlmostEqual(F[0], (1.0 / (3.0 * 64.0) - 1.0 / 256.0))
        self.assertAlmostEqual(F[1], (1.0 / 256.0))
        return

class Test_CDR_Linear_1D_VMS(unittest.TestCase):
    def setUp(self):
        return

    def test_stiffness_diffusion(self):
        el = ConvectionDiffusionReactionElement.Linear_1D_VMS(0, 0.25, 1, 0, 0);
        K = el.stiffness_matrix()
        self.assertAlmostEqual(K[0,0], 4.0)
        self.assertAlmostEqual(K[0,1], -4.0)
        self.assertAlmostEqual(K[1,0], -4.0)
        self.assertAlmostEqual(K[1,1], 4.0)
        return

    def test_stiffness_reaction(self):
        el = ConvectionDiffusionReactionElement.Linear_1D_VMS(0, 0.25, 1, 1, 0);
        K = el.stiffness_matrix()
        self.assertAlmostEqual(K[0,0], 4.0 + 1.0/12.0)
        self.assertAlmostEqual(K[0,1], -4.0 + 1.0/24.0)
        self.assertAlmostEqual(K[1,0], -4.0 + 1.0/24.0)
        self.assertAlmostEqual(K[1,1], 4.0 + 1.0/12.0)
        return

    def test_stiffness_convection(self):
        el = ConvectionDiffusionReactionElement.Linear_1D_VMS(0, 0.25, 1, 0, 1);
        K = el.stiffness_matrix()
        tau = numpy.cosh(1.0 / 8.0)/(8.0 * numpy.sinh(1.0/8.0)) - 1.0
        self.assertAlmostEqual(K[0,0], 4.0 + 0.5 + 4*tau)
        self.assertAlmostEqual(K[0,1], -4.0 + 0.5 - 4*tau)
        self.assertAlmostEqual(K[1,0], -4.0 + -0.5 - 4*tau)
        self.assertAlmostEqual(K[1,1], 4.0 + -0.5 + 4*tau)
        return

    def test_load(self):
        el = ConvectionDiffusionReactionElement.Linear_1D_VMS(0, 0.25, 1, 0, 0);
        F = el.load_vector(lambda x: x ** 2)
        self.assertAlmostEqual(F[0], (1.0 / (3.0 * 64.0) - 1.0 / 256.0))
        self.assertAlmostEqual(F[1], (1.0 / 256.0))
        return

class Test_Cubic_1D(unittest.TestCase):
    def setUp(self):
       return

    def test_T_master(self):
        el = Element.Cubic_1D(-1, 1)
        self.assertAlmostEqual(el.T(-1), -1)
        self.assertAlmostEqual(el.T(-0.5), -0.5)
        self.assertAlmostEqual(el.T(0), 0)
        self.assertAlmostEqual(el.T(0.5), 0.5)
        self.assertAlmostEqual(el.T(1), 1)
        self.assertAlmostEqual(el.w[0], -9.0/16.0)
        self.assertAlmostEqual(el.w[1], 27.0/16.0)
        self.assertAlmostEqual(el.w[2], -27.0/16.0)
        self.assertAlmostEqual(el.w[3], 9.0/16.0)
        self.assertAlmostEqual(el.xipts[0], -1)
        self.assertAlmostEqual(el.xipts[1], -1.0/3.0)
        self.assertAlmostEqual(el.xipts[2], 1.0/3.0)
        self.assertAlmostEqual(el.xipts[3], 1)
        return

    def test_T_arbitrary(self):
        el = Element.Cubic_1D(0.5, 0.7)
        self.assertAlmostEqual(el.T(0.5), -1)
        self.assertAlmostEqual(el.T(0.6), 0)
        self.assertAlmostEqual(el.T(0.7), 1)
        self.assertAlmostEqual(el.w[0], -9.0/16.0)
        self.assertAlmostEqual(el.w[1], 27.0/16.0)
        self.assertAlmostEqual(el.w[2], -27.0/16.0)
        self.assertAlmostEqual(el.w[3], 9.0/16.0)
        self.assertAlmostEqual(el.xipts[0], -1)
        self.assertAlmostEqual(el.xipts[1], -1.0/3.0)
        self.assertAlmostEqual(el.xipts[2], 1.0/3.0)
        self.assertAlmostEqual(el.xipts[3], 1)
        return

    def test_jacobian_master(self):
        el = Element.Cubic_1D(-1, 1)
        self.assertAlmostEqual(el.jacobian(0), 1.0)
        return

    def test_jacobian_arbitrary(self):
        el = Element.Cubic_1D(-23, 51)
        self.assertAlmostEqual(el.jacobian(0), 2.0 / 74.0)
        return

    def test_integrate_master(self):
        el = Element.Cubic_1D(-1, 1)
        I = el.integrate(lambda x: 1)
        self.assertAlmostEqual(I, 2.0)
        I = el.integrate(lambda x: x**2)
        self.assertAlmostEqual(I, 2.0 / 3.0)
        return

    def test_integrate_arbitrary(self):
        el = Element.Cubic_1D(1.0, 1.5)
        I = el.integrate(lambda x: 2)
        self.assertAlmostEqual(I, 1.0)
        I = el.integrate(lambda x: x ** 3)
        self.assertAlmostEqual(I, ((1.5)**4 - 1) / 4.0)
        return

class Test_CDR_Cubic_1D(unittest.TestCase):
    def setUp(self):
        return

    def test_stiffness_diffusion(self):
        el = ConvectionDiffusionReactionElement.Cubic_1D(0, 0.25, 1, 0, 0);
        K = el.stiffness_matrix()
        self.assertAlmostEqual(K[0,0], 74.0/5.0)
        self.assertAlmostEqual(K[0,1], -18.9)
        self.assertAlmostEqual(K[0,2], 27.0/5.0)
        self.assertAlmostEqual(K[0,3], -1.3)
        self.assertAlmostEqual(K[1,0], -18.9)
        self.assertAlmostEqual(K[1,1], 216.0/5.0)
        self.assertAlmostEqual(K[1,2], -29.7)
        self.assertAlmostEqual(K[1,3], 27.0/5.0)
        self.assertAlmostEqual(K[2,0], 27.0/5.0)
        self.assertAlmostEqual(K[2,1], -29.7)
        self.assertAlmostEqual(K[2,2], 216.0/5.0)
        self.assertAlmostEqual(K[2,3], -18.9)
        self.assertAlmostEqual(K[3,0], -1.3)
        self.assertAlmostEqual(K[3,1], 27.0/5.0)
        self.assertAlmostEqual(K[3,2], -18.9)
        self.assertAlmostEqual(K[3,3], 74.0/5.0)
        return

    def test_stiffness_reaction(self):
        el = ConvectionDiffusionReactionElement.Cubic_1D(0, 0.25, 0, 1, 0);
        K = el.stiffness_matrix()
        self.assertAlmostEqual(K[0,0], 1.0/12.0)
        self.assertAlmostEqual(K[0,1], 1.0/24.0)
        self.assertAlmostEqual(K[1,0], 1.0/24.0)
        self.assertAlmostEqual(K[1,1], 1.0/12.0)
        return

    def test_stiffness_convection(self):
        el = ConvectionDiffusionReactionElement.Cubic_1D(0, 0.25, 0, 0, 1);
        K = el.stiffness_matrix()
        self.assertAlmostEqual(K[0,0], 0.5)
        self.assertAlmostEqual(K[0,1], 0.5)
        self.assertAlmostEqual(K[1,0], -0.5)
        self.assertAlmostEqual(K[1,1], -0.5)
        return

    def test_load(self):
        el = ConvectionDiffusionReactionElement.Cubic_1D(0, 0.25, 1, 0, 0);
        F = el.load_vector(lambda x: x ** 2)
        self.assertAlmostEqual(F[0], 1.0/7680.0)
        self.assertAlmostEqual(F[1], 0)
        self.assertAlmostEqual(F[2], 9.0/2560.0)
        self.assertAlmostEqual(F[3], 1.0/640.0)
        return

if __name__ == '__main__':
    unittest.main()
