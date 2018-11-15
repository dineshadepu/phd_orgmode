import numpy as np
import unittest

from dem import (get_particle_array_dem, UpdateTangentialContacts,
                 LinearSpringForceParticleParticle)
from pysph.base.kernels import CubicSpline
from pysph.sph.equation import Group
from pysph.tools.sph_evaluator import SPHEvaluator


class TestCase1TangentialContacts(unittest.TestCase):
    def setUp(self):
        x = np.asarray([0., 2.0, 4.0, 0.0, 1.5])
        y = np.asarray([0., 2.0, 4.0, 3.0, 4.5])
        R = 0.1 * np.ones_like(x)
        pa = get_particle_array_dem(x=x, y=y, R=R, total_dem_entities=1,
                                    dem_id=0, dim=2, name="sand")
        np.testing.assert_almost_equal(len(pa.tng_idx), 30)
        np.testing.assert_approx_equal(0, pa.dem_id[0])
        np.testing.assert_approx_equal(1, pa.total_dem_entities[0])
        self.pa = pa

    def test_dem_particle_array(self):
        np.testing.assert_almost_equal(len(self.pa.tng_idx), 30)
        np.testing.assert_approx_equal(0, self.pa.dem_id[0])
        np.testing.assert_approx_equal(1, self.pa.total_dem_entities[0])

    def test_tangential_contacts(self):
        # before updating the  contact information lets check
        # tangential properties

        # number of contacts of each individual particles
        for i in range(len(self.pa.x)):
            self.assertAlmostEqual(self.pa.total_tng_contacts[i], 0.)

        for i in range(self.pa.total_dem_entities[0]):
            self.assertAlmostEqual(self.pa.tng_idx[i], -1)
            self.assertAlmostEqual(self.pa.tng_x[i], 0.)
            self.assertAlmostEqual(self.pa.tng_y[i], 0.)
            self.assertAlmostEqual(self.pa.tng_z[i], 0.)

        # ------------------------------
        # execute the interparticle force equation
        # ------------------------------
        # Given
        eqs = [
            Group(equations=[
                LinearSpringForceParticleParticle(dest='sand',
                                                  sources=['sand'])
            ])
        ]

        kernel = CubicSpline(dim=2)
        sph_eval = SPHEvaluator(
            arrays=[self.pa], equations=eqs,
            dim=2, kernel=kernel
        )

        # When
        sph_eval.evaluate(0.0, 0.1)

        # check the tangential contacts indices and displacements
        # number of contacts of each individual particles
        for i in range(len(self.pa.x)):
            self.assertAlmostEqual(self.pa.total_tng_contacts[i], 0)

        for i in range(self.pa.total_dem_entities[0]):
            self.assertAlmostEqual(self.pa.tng_idx[i], -1)
            self.assertAlmostEqual(self.pa.tng_x[i], 0.)
            self.assertAlmostEqual(self.pa.tng_y[i], 0.)
            self.assertAlmostEqual(self.pa.tng_z[i], 0.)


class TestCase2TangentialContacts(unittest.TestCase):
    def setUp(self):
        x = np.asarray([0., 0.18, -0.18, 0.21, 0.])
        y = np.asarray([0., 0.21, 0.05, 0.03, 0.19])
        R = 0.1 * np.ones_like(x)
        # x = np.asarray([0., 0.1])
        # y = np.asarray([0., 0.1])
        # R = 0.1 * np.ones_like(x)
        pa = get_particle_array_dem(x=x, y=y, R=R, total_dem_entities=1,
                                    dem_id=0, dim=2, name="sand")
        np.testing.assert_almost_equal(len(pa.tng_idx), 30)
        np.testing.assert_approx_equal(0, pa.dem_id[0])
        np.testing.assert_approx_equal(1, pa.total_dem_entities[0])
        self.pa = pa

    def test_dem_particle_array(self):
        np.testing.assert_almost_equal(len(self.pa.tng_idx), 30)
        np.testing.assert_approx_equal(0, self.pa.dem_id[0])
        np.testing.assert_approx_equal(1, self.pa.total_dem_entities[0])

    def test_tangential_contacts(self):
        # before updating the  contact information lets check
        # tangential properties

        # number of contacts of each individual particles
        for i in range(len(self.pa.x)):
            self.assertAlmostEqual(self.pa.total_tng_contacts[i], 0)

        for i in range(self.pa.total_dem_entities[0]):
            self.assertAlmostEqual(self.pa.tng_idx[i], -1)
            self.assertAlmostEqual(self.pa.tng_x[i], 0.)
            self.assertAlmostEqual(self.pa.tng_y[i], 0.)
            self.assertAlmostEqual(self.pa.tng_z[i], 0.)

        # ------------------------------
        # execute the interparticle force equation
        # ------------------------------
        # Given
        eqs = [
            Group(equations=[
                LinearSpringForceParticleParticle(dest='sand',
                                                  sources=['sand'])
            ])
        ]

        kernel = CubicSpline(dim=2)
        sph_eval = SPHEvaluator(
            arrays=[self.pa], equations=eqs,
            dim=2, kernel=kernel
        )

        # When
        sph_eval.evaluate(0.0, 0.1)

        # Then
        # check the tangential contacts indices and displacements
        # number of contacts of each individual particles
        tot_ctcs = [2, 2, 1, 1, 2]
        print(self.pa.total_tng_contacts)
        print(self.pa.tng_idx)
        for i in range(len(self.pa.x)):
            self.assertAlmostEqual(self.pa.total_tng_contacts[i], tot_ctcs[i])
