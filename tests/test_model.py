import unittest
import numpy as np
from unittest.mock import Mock
from blobmodel import Model


class TestComputeStartStop(unittest.TestCase):
    def setUp(self):
        # Mock the geometry object
        self.geometry = Mock()
        self.geometry.t = np.linspace(0, 10, 101)  # t from 0 to 10 with 101 points
        self.geometry.dt = 0.1  # Sampling time
        self.geometry.Lx = 100.0  # Domain size

        # Mock the Blob object
        self.blob = Mock()
        self.blob.t_init = 2.0  # Default initial time
        self.blob.v_x = 10.0  # Default velocity
        self.blob.width_prop = 1.0  # Default width property
        self.blob.pos_x = 0.0  # Default initial position
        self.model = Model()
        self.model._geometry = self.geometry

        self.error = 1e-5

    def test_no_speed_up(self):
        start, stop = self.model._compute_start_stop(
            self.blob, speed_up=False, error=self.error
        )
        self.assertEqual(start, 0)
        self.assertEqual(stop, self.geometry.t.size)

    def test_zero_velocity(self):
        self.blob.v_x = 0.0
        start, stop = self.model._compute_start_stop(
            self.blob, speed_up=True, error=self.error
        )
        self.assertEqual(start, 0)
        self.assertEqual(stop, self.geometry.t.size)

    def test_positive_velocity(self):
        self.blob.v_x = 10.0
        start, stop = self.model._compute_start_stop(
            self.blob, speed_up=True, error=self.error
        )
        self.assertEqual(start, 9)
        self.assertEqual(stop, 101)

    def test_negative_velocity(self):
        self.blob.v_x = -10.0
        start, stop = self.model._compute_start_stop(
            self.blob, speed_up=True, error=self.error
        )
        self.assertEqual(start, 9)
        self.assertEqual(stop, 101)

    def test_boundary_start_clipped(self):
        self.blob.t_init = 0
        self.blob.v_x = 10.0
        start, stop = self.model._compute_start_stop(
            self.blob, speed_up=True, error=self.error
        )
        self.assertEqual(start, 0)
        self.assertEqual(stop, 101)

    def test_small_error(self):
        self.blob.v_x = 10.0
        small_error = 1e-10
        start, stop = self.model._compute_start_stop(
            self.blob, speed_up=True, error=small_error
        )
        self.assertEqual(start, 0)
        self.assertEqual(stop, 101)

    def test_large_velocity(self):
        self.blob.v_x = 1000.0
        start, stop = self.model._compute_start_stop(
            self.blob, speed_up=True, error=self.error
        )
        self.assertEqual(start, 19)
        self.assertEqual(stop, 21)
