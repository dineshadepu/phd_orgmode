"""100 spheres falling inside hopper
Check the complete molecular dynamics code
"""
from __future__ import print_function
import numpy as np
# import matplotlib.pyplot as plt

# PySPH base and carray imports
from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver
from pysph.sph.integrator import (EulerIntegrator, EPECIntegrator)

from pysph.sph.equation import Group
from dem import (LinearSpringForceParticleParticle, MakeForcesZero, BodyForce,
                 get_particle_array_dem, EulerDEMStep, RK2DEMStep,
                 UpdateTangentialContacts)
from pysph.solver.application import Application


def add_properties(pa, *props):
    for prop in props:
        pa.add_property(name=prop)


def create_hopper(r):
    d = 2 * r
    x_start = 0.2
    x_final = x_start
    y_start = 0
    y_final = y_start
    theta = 60 * np.pi / 180.
    x = []
    y = []
    while x_final < 2:
        x.append(x_final)
        y.append(y_final)
        x_final = x_final + d * np.cos(theta)
        y_final = y_final + d * np.sin(theta)
    x_l = np.asarray(x)
    y_l = np.asarray(y)

    x_r = -np.asarray(x)
    y_r = np.asarray(y)

    x, y = np.concatenate([x_l, x_r]), np.concatenate([y_l, y_r])
    return x, y


class HopperFlow(Application):
    def initialize(self):
        self._sph_eval = None
        self.dx = 1

    def create_particles(self):
        x = np.linspace(-0.5, 0.5, 10)
        y = np.linspace(0.77, 1.77, 10)
        r = (x[1] - x[0]) / 2.
        x, y = np.meshgrid(x, y)
        x, y = x.ravel(), y.ravel()
        R = np.ones_like(x) * r
        _m = np.pi * 2 * r * 2 * r
        m = np.ones_like(x) * _m
        m_inverse = np.ones_like(x) * 1. / _m
        _I = 2. / 5. * _m * r**2
        I_inverse = np.ones_like(x) * 1. / _I
        h = np.ones_like(x) * r
        sand = get_particle_array_dem(x=x, y=y, m=m, m_inverse=m_inverse, R=R,
                                      h=h, I_inverse=I_inverse, name="sand",
                                      dem_id=0, dim=2, total_dem_entities=2)

        x, y = create_hopper(r)
        m = np.ones_like(x) * _m
        m_inverse = np.ones_like(x) * 1. / _m
        R = np.ones_like(x) * r
        h = np.ones_like(x) * r
        _I = 2. / 5. * _m * r**2
        I_inverse = np.ones_like(x) * 1. / _I
        wall = get_particle_array_dem(x=x, y=y, m=m, m_inverse=m_inverse, R=R,
                                      h=h, I_inverse=I_inverse, name="wall",
                                      dem_id=1, dim=2, total_dem_entities=2)
        return [sand, wall]

    def create_solver(self):
        kernel = CubicSpline(dim=2)

        integrator = EulerIntegrator(sand=EulerDEMStep())

        dt = 5e-5
        print("DT: %s" % dt)
        tf = 2
        solver = Solver(kernel=kernel, dim=2, integrator=integrator, dt=dt,
                        tf=tf, adaptive_timestep=False)

        return solver

    def create_equations(self):
        equations = [
            Group(equations=[
                BodyForce(dest='sand', sources=None, gy=-9.81),
                LinearSpringForceParticleParticle(
                    dest='sand', sources=['sand', 'wall'], kn=1e3),
            ]),
        ]
        return equations

    def post_step(self, solver):
        from pysph.tools.sph_evaluator import SPHEvaluator
        if self._sph_eval is None:
            equations = [
                UpdateTangentialContacts(dest='sand', sources=['sand', 'wall'])
            ]

            self._sph_eval = SPHEvaluator(arrays=self.particles,
                                          equations=equations, dim=2,
                                          kernel=CubicSpline(dim=2))

        self._sph_eval.evaluate()


if __name__ == '__main__':
    app = HopperFlow()
    app.run()
