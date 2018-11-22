// library imports
extern crate itertools_num;

pub mod dump_output;
pub mod geometry;


pub struct RigidBody2d {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    pub x_bs: Vec<f32>,
    pub y_bs: Vec<f32>,
    pub r: Vec<f32>,
    pub fx: Vec<f32>,
    pub fy: Vec<f32>,
    pub m: Vec<f32>,
    pub total_mass: f32,
    pub position_com: Vec<f32>,
    pub velocity_com: Vec<f32>,
    pub theta: f32,
    pub omega: f32,
    pub inertia: f32,
    pub inertia_inv: f32,
    pub force: Vec<f32>,
    pub torque: f32,
}

impl RigidBody2d {
    pub fn new(x: Vec<f32>, y: Vec<f32>, m: Vec<f32>, r: Vec<f32>) -> RigidBody2d {
        // compute center of mass from the given coordinates
        let (xcm, ycm, total_mass) = {
            let mut x_cm = 0.;
            let mut y_cm = 0.;
            let mut total_mass = 0.;
            for i in 0..x.len() {
                x_cm += m[i] * x[i];
                y_cm += m[i] * y[i];
                total_mass += m[i];
            }
            (x_cm / total_mass, y_cm / total_mass, total_mass)
        };

        let mut x_bs = vec![];
        let mut y_bs = vec![];
        let mut inertia = 0.;
        for i in 0..x.len() {
            x_bs.push(x[i] - xcm);
            y_bs.push(y[i] - ycm);
            inertia += m[i] * (x_bs[i].powf(2.) + y_bs[i].powf(2.));
        }
        let inertia_inv = 1. / inertia;


        RigidBody2d {
            fx: vec![0.; x.len()],
            fy: vec![0.; x.len()],
            u: vec![0.; x.len()],
            v: vec![0.; x.len()],
            x: x,
            y: y,
            inertia: inertia,
            inertia_inv: inertia_inv,
            x_bs: x_bs,
            y_bs: y_bs,
            r: r,
            m: m,
            position_com: vec![xcm, ycm],
            total_mass: total_mass,
            velocity_com: vec![0., 0.],
            theta: 0.,
            omega: 0.,
            force: vec![0., 0.],
            torque: 0.0,
        }
    }

    pub fn reset_forces(&mut self) {
        self.torque = 0.0;
        self.force[0] = 0.0;
        self.force[1] = 0.0;

        for i in 0..self.x.len() {
            self.fx[i] = 0.0;
            self.fy[i] = 0.0;
        }
    }

    pub fn add_gravity(&mut self) {
        self.force[1] += - self.total_mass * 9.812;
    }

    pub fn aggregate_force_moments(&mut self) {
        for i in 0..self.x.len() {
            self.force[0] += self.fx[i];
            self.force[1] += self.fy[i];

            // rotate the particle in body space to world space
            // and compute torque
            let x_bs_to_ws = self.x_bs[i] * self.theta.cos() - self.y_bs[i] * self.theta.sin();
            let y_bs_to_ws = self.x_bs[i] * self.theta.sin() + self.y_bs[i] * self.theta.cos();
            self.torque += x_bs_to_ws * self.fy[i] - y_bs_to_ws * self.fx[i];
        }
    }

    pub fn update_body(&mut self, dt: f32) {
        self.velocity_com[0] = self.velocity_com[0] + self.force[0] / self.total_mass * dt;
        self.velocity_com[1] = self.velocity_com[1] + self.force[1] / self.total_mass * dt;

        self.position_com[0] = self.position_com[0] + self.velocity_com[0] * dt;
        self.position_com[1] = self.position_com[1] + self.velocity_com[1] * dt;

        self.omega += self.inertia_inv * self.torque * dt;
        self.theta += self.omega * dt;

        // update each particles velocity and position
        let x = &mut self.x;
        let y = &mut self.y;
        let u = &mut self.u;
        let v = &mut self.v;

        for i in 0..x.len(){
            let x_bs_to_ws = self.x_bs[i] * self.theta.cos() - self.y_bs[i] * self.theta.sin();
            let y_bs_to_ws = self.x_bs[i] * self.theta.sin() + self.y_bs[i] * self.theta.cos();
            x[i] = self.position_com[0] + x_bs_to_ws;
            y[i] = self.position_com[1] + y_bs_to_ws;

            u[i] = self.velocity_com[0] - self.omega * y_bs_to_ws;
            v[i] = self.velocity_com[1] + self.omega * x_bs_to_ws;
        }
    }
}

pub fn contact_force(dest: &mut RigidBody2d, source: &RigidBody2d, kn: f32) {
    let xd = &dest.x;
    let yd = &dest.y;
    let rd = &dest.r;
    let fxd = &mut dest.fx;
    let fyd = &mut dest.fy;
    let xs = &source.x;
    let ys = &source.y;
    let rs = &source.r;

    for i in 0..dest.x.len() {
        for j in 0..source.x.len() {
            let xij_vec = vec![xd[i] - xs[i], yd[i] - ys[i]];
            let rij = distance(&xij_vec);
            let overlap = rd[i] + rs[i] - rij;
            if overlap > 0. {
                fxd[i] += kn * overlap * &xij_vec[0] / rij;
                fyd[i] += kn * overlap * &xij_vec[1] / rij;
            }
        }
    }
}


fn distance(xij_vec: &Vec<f32>) -> f32 {
    (xij_vec[0].powf(2.0) + xij_vec[1].powf(2.0)).powf(0.5)
}
