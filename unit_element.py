#!/usr/bin/env python
import unittest
import Element

class Test_Linear_1D(unittest.TestCase):
    def setUp(self):
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
        I = el.integrate(lambda x: 1, lambda x: 1)
        self.assertAlmostEqual(I, 2.0)

    def test_integrate_arbitrary(self):
        el = Element.Linear_1D(1.0, 1.5)
        I = el.integrate(lambda x: 1, lambda x: 2)
        self.assertAlmostEqual(I, 1.0)
        return

if __name__ == '__main__':
    unittest.main()
