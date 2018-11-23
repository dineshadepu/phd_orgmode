import numpy

from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.equation import Equation
from pysph.cpy.types import declare
from math import sin, cos, sqrt


def get_particle_array_rigid_body(constants=None, **props):
    extra_props = [
        'xb_dash', 'yb_dash', 'xs_dash', 'ys_dash', 'fx', 'fy', 'fz',
        'tang_disp_y', 'tang_velocity_z', 'tang_velocity_y', 'tang_velocity_x',
        'tang_disp_z', 'tang_disp_x'
    ]

    body_id = props.pop('body_id', None)
    nb = 1 if body_id is None else numpy.max(body_id) + 1

    consts = {
        'total_mass': numpy.zeros(nb, dtype=float),
        'num_body': numpy.asarray(nb, dtype=int),
        'pos_cm': numpy.zeros(2 * nb, dtype=float),
        'pos_cm0': numpy.zeros(2 * nb, dtype=float),
        'mi_b': numpy.zeros(nb, dtype=float),
        'mi_b_inv': numpy.zeros(nb, dtype=float),
        'force': numpy.zeros(2 * nb, dtype=float),
        'torque': numpy.zeros(nb, dtype=float),
        'L': numpy.zeros(nb, dtype=float),
        'L0': numpy.zeros(nb, dtype=float),
        # velocity CM.
        'vel_cm': numpy.zeros(2 * nb, dtype=float),
        'vel_cm0': numpy.zeros(2 * nb, dtype=float),
        # angular velocity, acceleration of body.
        'omega': numpy.zeros(nb, dtype=float),
        'theta': numpy.zeros(nb, dtype=float),
        'omega0': numpy.zeros(nb, dtype=float),
        'theta0': numpy.zeros(nb, dtype=float),
    }
    if constants:
        consts.update(constants)
    pa = get_particle_array(constants=consts, additional_props=extra_props,
                            **props)
    if body_id is None:
        body_id = numpy.zeros(len(pa.x), dtype=int)

    pa.add_property('body_id', type='int', data=body_id)
    pa.add_property('indices', type='int', data=numpy.arange(len(pa.x)))

    compute_properties_of_rigid_body(pa)
    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm', 'p', 'pid', 'au', 'av',
        'aw', 'tag', 'gid', 'fx', 'fy'
    ])
    return pa


def compute_properties_of_rigid_body(pa):
    """Computes the precomputed values of rigid body such as total mass,
    center of mass, body moment of inertia inverse, body frame position vector
    of particles."""
    for i in range(pa.num_body[0]):
        # indices of a given body id
        cond = pa.body_id == i
        indices = pa.indices[cond]
        min_idx = min(indices)
        max_idx = max(indices) + 1

        pa.total_mass[i] = numpy.sum(pa.m[min_idx:max_idx])
        if pa.total_mass[i] == 0.:
            print("Total mass of the rigid body is\
            zero, please check mass of particles in body")

        # Compute center of mass body i
        x_cm = 0.
        y_cm = 0.
        for j in range(min_idx, max_idx):
            mj = pa.m[j]
            x_cm += pa.x[j] * mj
            y_cm += pa.y[j] * mj

        pa.pos_cm[2 * i] = x_cm / pa.total_mass[i]
        pa.pos_cm[2 * i + 1] = y_cm / pa.total_mass[i]
        pa.pos_cm0[2 * i] = x_cm / pa.total_mass[i]
        pa.pos_cm0[2 * i + 1] = y_cm / pa.total_mass[i]

        # save the body frame position vectors
        for j in range(min_idx, max_idx):
            mj = pa.m[j]
            # the posjtjon vector js from center of mass to the partjcle's
            # posjtjon jn body frame
            pa.xb_dash[j] = pa.x[j] - pa.pos_cm[2 * i]
            pa.yb_dash[j] = pa.y[j] - pa.pos_cm[2 * i + 1]
            # initially both body and global frames are same, so
            # global position vector from center of mass will be
            pa.xs_dash[j] = pa.x[j] - pa.pos_cm[2 * i]
            pa.ys_dash[j] = pa.y[j] - pa.pos_cm[2 * i + 1]

        # moment of inertia calculation
        i_zz = 0.
        for j in range(min_idx, max_idx):
            mj = pa.m[j]
            rbx = pa.xs_dash[j]
            rby = pa.ys_dash[j]
            i_zz += mj * (rbx**2. + rby**2.)

        # set moment of inertia inverse in body frame
        pa.mi_b[i] = i_zz
        pa.mi_b_inv[i] = 1. / i_zz

        # set the orientation of each body
        pa.omega[i] = 0.
        pa.omega0[i] = 0.
        pa.theta[i] = 0.
        pa.theta0[i] = 0.


class SumUpExternalForces(Equation):
    def reduce(self, dst, t, dt):
        xs_dash = declare('object')
        ys_dash = declare('object')
        fx = declare('object')
        fy = declare('object')
        torque = declare('object')
        force = declare('object')
        body_id = declare('object')
        i = declare('int')
        bid = declare('int')
        torque = dst.torque
        force = dst.force
        xs_dash = dst.xs_dash
        ys_dash = dst.ys_dash
        fx = dst.fx
        fy = dst.fy
        body_id = dst.body_id

        for i in range(len(fx)):
            bid = body_id[i]
            bid2 = bid * 2
            force[bid2] += fx[i]
            force[bid2 + 1] += fy[i]

            torque[bid] += xs_dash[i] * fy[i] - ys_dash[i] * fx[i]


class BodyForce2d(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0):
        self.gx = gx
        self.gy = gy
        super(BodyForce2d, self).__init__(dest, sources)

    def initialize(self, d_idx, d_m, d_fx, d_fy):
        d_fx[d_idx] = d_m[d_idx] * self.gx
        d_fy[d_idx] = d_m[d_idx] * self.gy


class RigidBodyCollision2d(Equation):
    """Force between two spheres is implemented using DEM contact force law.

    Refer https://doi.org/10.1016/j.powtec.2011.09.019 for more
    information.

    Open-source MFIX-DEM software for gas–solids flows:
    Part I—Verification studies .

    """

    def __init__(self, dest, sources, kn=1e3, mu=0.5, en=0.8):
        """Initialise the required coefficients for force calculation.


        Keyword arguments:
        kn -- Normal spring stiffness (default 1e3)
        mu -- friction coefficient (default 0.5)
        en -- coefficient of restitution (0.8)

        Given these coefficients, tangential spring stiffness, normal and
        tangential damping coefficient are calculated by default.

        """
        self.kn = kn
        self.kt = 2. / 7. * kn
        m_eff = numpy.pi * 0.5**2 * 1e-6 * 2120
        self.gamma_n = -(2 * numpy.sqrt(kn * m_eff) * numpy.log(en)) / (
            numpy.sqrt(numpy.pi**2 + numpy.log(en)**2))
        self.gamma_t = 0.5 * self.gamma_n
        self.mu = mu
        super(RigidBodyCollision2d, self).__init__(dest, sources)

    def loop(self, d_idx, d_fx, d_fy, d_h, d_total_mass, d_rad_s,
             d_tang_disp_x, d_tang_disp_y, d_tang_velocity_x,
             d_tang_velocity_y, s_idx, s_rad_s, XIJ, RIJ, R2IJ, VIJ):
        overlap = 0
        if RIJ > 1e-9:
            overlap = d_rad_s[d_idx] + s_rad_s[s_idx] - RIJ

        if overlap > 0:
            # normal vector passing from particle i to j
            nij_x = -XIJ[0] / RIJ
            nij_y = -XIJ[1] / RIJ

            # overlap speed: a scalar
            vijdotnij = VIJ[0] * nij_x + VIJ[1] * nij_y

            # normal velocity
            vijn_x = vijdotnij * nij_x
            vijn_y = vijdotnij * nij_y

            # normal force with conservative and dissipation part
            fn_x = -self.kn * overlap * nij_x - self.gamma_n * vijn_x
            fn_y = -self.kn * overlap * nij_y - self.gamma_n * vijn_y

            d_fx[d_idx] += fn_x
            d_fy[d_idx] += fn_y


class RK2StepRigidBody(IntegratorStep):
    def initialize(self, d_idx, d_pos_cm, d_pos_cm0, d_vel_cm, d_vel_cm0,
                   d_theta0, d_theta, d_force, d_torque, d_omega, d_omega0,
                   d_num_body):
        _i = declare('int')
        _j = declare('int')
        if d_idx == 0:
            for _i in range(d_num_body[0]):
                d_theta0[_i] = d_theta[_i]
                d_omega0[_i] = d_omega[_i]
                d_torque[_i] = 0.
                for _j in range(2):
                    d_force[2 * _i + _j] = 0.
                    d_pos_cm0[2 * _i + _j] = d_pos_cm[2 * _i + _j]
                    d_vel_cm0[2 * _i + _j] = d_vel_cm[2 * _i + _j]

    def py_stage1(self, dst, t, dt):
        dtb2 = dt / 2.
        for i in range(dst.num_body[0]):
            i2 = 2 * i
            # update center of mass position and velocity
            dst.pos_cm[i2:i2 + 2] = (
                dst.pos_cm0[i2:i2 + 2] + dtb2 * dst.vel_cm0[i2:i2 + 2])
            dst.vel_cm[i2:i2 + 2] = (
                dst.vel_cm0[i2:i2 + 2] +
                dtb2 * dst.force[i2:i2 + 2] / dst.total_mass[i])

            # update orientation
            dst.theta[i] = dst.theta0[i] + dst.omega0[i] * dtb2

            # update angular velocity
            dst.omega[i] = dst.omega0[i] + dst.torque[i] * dtb2

    def stage1(self, d_idx, d_indices, d_xb_dash, d_yb_dash, d_xs_dash,
               d_ys_dash, d_x, d_y, d_u, d_v, d_vel_cm, d_pos_cm, d_omega,
               d_theta, d_body_id):
        i2 = declare('int')
        bid = declare('int')
        bid = d_body_id[d_idx]
        i2 = 2 * bid

        d_xs_dash[d_idx] = (d_xb_dash[d_idx] * cos(d_theta[bid]) -
                            d_yb_dash[d_idx] * sin(d_theta[bid]))
        d_ys_dash[d_idx] = (d_xb_dash[d_idx] * sin(d_theta[bid]) +
                            d_yb_dash[d_idx] * cos(d_theta[bid]))

        d_x[d_idx] = d_pos_cm[i2] + d_xs_dash[d_idx]
        d_y[d_idx] = d_pos_cm[i2 + 1] + d_ys_dash[d_idx]
        d_u[d_idx] = d_vel_cm[i2] - d_omega[bid] * d_ys_dash[d_idx]
        d_v[d_idx] = d_vel_cm[i2 + 1] + d_omega[bid] * d_xs_dash[d_idx]

    def py_stage2(self, dst, t, dt):
        for i in range(dst.num_body[0]):
            i2 = 2 * i
            # update center of mass position and velocity
            dst.pos_cm[i2:i2 + 2] = (
                dst.pos_cm0[i2:i2 + 2] + dt * dst.vel_cm0[i2:i2 + 2])
            dst.vel_cm[i2:i2 + 2] = (
                dst.vel_cm0[i2:i2 + 2] +
                dt * dst.force[i2:i2 + 2] / dst.total_mass[i])

            # update orientation
            dst.theta[i] = dst.theta0[i] + dst.omega0[i] * dt

            # update angular velocity
            dst.omega[i] = dst.omega0[i] + dst.torque[i] * dst.mi_b_inv[i] * dt

    def stage2(self, d_idx, d_indices, d_xb_dash, d_yb_dash, d_xs_dash,
               d_ys_dash, d_x, d_y, d_u, d_v, d_vel_cm, d_pos_cm, d_omega,
               d_theta, d_body_id):
        i2 = declare('int')
        bid = declare('int')
        bid = d_body_id[d_idx]
        i2 = 2 * bid

        d_xs_dash[d_idx] = (d_xb_dash[d_idx] * cos(d_theta[bid]) -
                            d_yb_dash[d_idx] * sin(d_theta[bid]))
        d_ys_dash[d_idx] = (d_xb_dash[d_idx] * sin(d_theta[bid]) +
                            d_yb_dash[d_idx] * cos(d_theta[bid]))

        d_x[d_idx] = d_pos_cm[i2] + d_xs_dash[d_idx]
        d_y[d_idx] = d_pos_cm[i2 + 1] + d_ys_dash[d_idx]
        d_u[d_idx] = d_vel_cm[i2] - d_omega[bid] * d_ys_dash[d_idx]
        d_v[d_idx] = d_vel_cm[i2 + 1] + d_omega[bid] * d_xs_dash[d_idx]
