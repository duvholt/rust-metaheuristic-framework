import unittest
import test_functions as tf
import numpy as np


class Plot(unittest.TestCase):

    def test_himmelblau(self):
        self.assertEqual(0, tf.himmelblau(np.array([3]), np.array([2])))
        self.assertAlmostEqual(0, tf.himmelblau(np.array([-2.805118]), np.array([3.131312]))[0])
        self.assertAlmostEqual(0, tf.himmelblau(np.array([-3.779310]), np.array([-3.283186]))[0])
        self.assertAlmostEqual(0, tf.himmelblau(np.array([3.584428]), np.array([-1.848126]))[0])

    def test_rosenbrock(self):
        X = np.array([1])
        Y = np.array([1])
        self.assertEqual(0, tf.rosenbrock(X, Y)[0])

    def test_ackley(self):
        X = np.array([0])
        Y = np.array([0])
        self.assertAlmostEqual(0, tf.ackley(X, Y)[0])

    def test_zakharov(self):
        X = np.array([0])
        Y = np.array([0])
        self.assertEqual(0, tf.zakharov(X, Y)[0])
