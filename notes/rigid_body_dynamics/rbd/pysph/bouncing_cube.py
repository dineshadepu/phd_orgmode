"""A cube bouncing inside a box. (5 seconds)

This is used to test the rigid body equations.
"""

import numpy as np

from pysph.base.kernels import CubicSpline
from pysph.sph.equation import Group

from pysph.sph.integrator import EPECIntegrator

from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.examples.rigid_body.ten_spheres_in_vessel_2d import (get_2d_block,
                                                                get_2d_dam)
from rigid_body_2d import (get_particle_array_rigid_body, RK2StepRigidBody,
                           SumUpExternalForces, BodyForce2d,
                           RigidBodyCollision2d)


class BouncingCube(Application):
    def initialize(self):
        self.rho0 = 2000
        self.dx = 0.1
        self.hdx = 1.
        self.block_length = 1.
        self.block_height = 1.
        self.tank_length = 3.
        self.tank_height = 3.

    def create_particles(self):
        x, y = get_2d_block(self.block_length, self.block_height, self.dx)
        x = x + 1.
        y = y + 1.
        m = np.ones_like(x) * self.dx * self.dx * self.rho0
        h = np.ones_like(x) * self.hdx * self.dx
        # radius of each sphere constituting in cube
        rad_s = np.ones_like(x) * self.dx / 2.
        body = get_particle_array_rigid_body(name='body', x=x, y=y, h=h, m=m,
                                             rad_s=rad_s)
        body.omega[0] = 2.
        body.omega0[0] = 2.

        # Create the tank.
        x, y = get_2d_dam(self.tank_length, self.tank_height, self.dx)
        m = np.ones_like(x) * self.dx * self.dx * self.rho0
        h = np.ones_like(x) * self.hdx * self.dx
        # radius of each sphere constituting in cube
        rad_s = np.ones_like(x) * self.dx / 2.
        tank = get_particle_array_rigid_body(name='tank', x=x, y=y, h=h, m=m,
                                             rad_s=rad_s)

        return [body, tank]

    def create_solver(self):
        tf = 2.
        dt = 1e-4

        kernel = CubicSpline(dim=2)

        integrator = EPECIntegrator(body=RK2StepRigidBody())

        solver = Solver(kernel=kernel, dim=2, integrator=integrator, dt=dt,
                        tf=tf, adaptive_timestep=False)
        solver.set_print_freq(100)
        return solver

    def create_equations(self):
        equations = [
            Group(equations=[
                BodyForce2d(dest='body', sources=None, gy=-9.81),
                RigidBodyCollision2d(dest='body', sources=['tank'], kn=1e7,
                                     en=0.2)
            ]),
            Group(equations=[SumUpExternalForces(dest='body', sources=None)]),
        ]
        return equations


if __name__ == '__main__':
    app = BouncingCube()
    app.run()
