extern crate itertools_num;
extern crate rbd;

use std::fs;
use rbd::{geometry, dump_output, contact_force, RigidBody2d};

fn main() {
    let spacing = 0.1;
    let (xb, yb) = geometry::grid_arange(1., 2., spacing, 2., 2.15, spacing);
    let rb = vec![spacing / 2.; xb.len()];
    let rho = 2500.;
    let m = vec![rho * spacing.powf(2.); xb.len()];
    let mut body = RigidBody2d::new(xb, yb, m, rb);

    let (xt, yt) = geometry::grid_arange(1.5, 2.5, spacing, 0., 0.15, spacing);
    let rt = vec![spacing / 2.; xt.len()];
    let m = vec![rho * spacing.powf(2.); xt.len()];
    let mut tank = RigidBody2d::new(xt, yt, m, rt);

    let mut tf = 1.0;
    // let dt = 2.;
    let dt = 1e-4;
    let mut step_no = 0;
    let pfreq = 100;

    let version = env!("CARGO_MANIFEST_DIR");
    let dir_name = version.to_owned() + "/bouncing_cube_output";
    fs::create_dir(&dir_name);

    // simulation
    // body.omega = 2.;
    while tf > 0.0 {
        // get the neighbours
        // log all the indices into verlet list
        // let neighbours = stash_particles();

        body.reset_forces();
        body.add_gravity();

        // contact force between body and tank
        contact_force(&mut body, &tank, 1e8);

        body.aggregate_force_moments();
        body.update_body(dt);

        if step_no % pfreq == 0 {
            // println!("{}", step_no);
            let filename = format!("{}/tank_{}.vtk", &dir_name, step_no);
            dump_output::write_vtk(&tank, filename);

            let filename = format!("{}/body_{}.vtk", &dir_name, step_no);
            dump_output::write_vtk(&body, filename);
            println!("{} ", step_no);
        }
        step_no += 1;
        tf -= dt;
    }
}
