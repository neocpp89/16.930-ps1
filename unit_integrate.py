#!/usr/bin/env python
import random
import unittest

# quadrature functions
import scipy.integrate as spint
import gauss

class TestGauss(unittest.TestCase):
    def setUp(self):
       return

    def test_p0(self):
        f = lambda x: 1
        quad = gauss.quadrature(f, 0, 1, 2)
        self.assertAlmostEqual(1, quad)
        return

    def test_p1(self):
        f = lambda x: 1 + x
        quad = gauss.quadrature(f, 0, 1)
        self.assertAlmostEqual(1.5, quad)
        return

    def test_p2(self):
        f = lambda x: 1 + x + x ** 2
        quad = gauss.quadrature(f, 0, 1)
        self.assertAlmostEqual((1+1.0/2.0 + 1.0/3.0), quad)
        return

    def test_p3(self):
        f = lambda x: 1 + x + x ** 2 + x ** 3
        quad = gauss.quadrature(f, 0, 1, 10)
        self.assertAlmostEqual((1+1.0/2.0 + 1.0/3.0 + 1.0/4.0), quad)
        return

    def test_p4(self):
        f = lambda x: 1 + x + x ** 2 + x ** 3 + x ** 4
        quad = gauss.quadrature(f, 0, 1)
        self.assertAlmostEqual((1+1.0/2.0 + 1.0/3.0 + 1.0/4.0 + 1.0/5.0), quad)
        return

    def test_p5(self):
        f = lambda x: 1 + x + x ** 2 + x ** 3 + x ** 4 + x ** 5
        quad = gauss.quadrature(f, 0, 12)
        self.assertAlmostEqual(2766372.0/5.0, quad)
        return

class TestSciPyQuadrature(unittest.TestCase):
    def setUp(self):
       return

    def test_p0(self):
        f = lambda x: 1
        quad = spint.fixed_quad(f, 0, 1)[0]
        self.assertAlmostEqual(1, quad)
        return

    def test_p1(self):
        f = lambda x: 1 + x
        quad = spint.fixed_quad(f, 0, 1)[0]
        self.assertAlmostEqual(1.5, quad)
        return

    def test_p2(self):
        f = lambda x: 1 + x + x ** 2
        quad = spint.fixed_quad(f, 0, 1)[0]
        self.assertAlmostEqual((1+1.0/2.0 + 1.0/3.0), quad)
        return

    def test_p3(self):
        f = lambda x: 1 + x + x ** 2 + x ** 3
        quad = spint.fixed_quad(f, 0, 1)[0]
        self.assertAlmostEqual((1+1.0/2.0 + 1.0/3.0 + 1.0/4.0), quad)
        return

    def test_p4(self):
        f = lambda x: 1 + x + x ** 2 + x ** 3 + x ** 4
        quad = spint.fixed_quad(f, 0, 1)[0]
        self.assertAlmostEqual((1+1.0/2.0 + 1.0/3.0 + 1.0/4.0 + 1.0/5.0), quad)
        return

    def test_p5(self):
        f = lambda x: 1 + x + x ** 2 + x ** 3 + x ** 4 + x ** 5
        quad = spint.fixed_quad(f, 0, 12)[0]
        self.assertAlmostEqual(2766372.0/5.0, quad)
        return

if __name__ == '__main__':
    unittest.main()
