use std::env::current_exe;
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use super::RigidBody2d;

pub fn write_vtk(body: &RigidBody2d, output: String) {
    // This is taken from
    // https://lorensen.github.io/VTKExamples/site/VTKFileFormats/#legacy-file-examples
    // let mut filename: String = current_exe().unwrap().to_str().unwrap().to_string();
    // filename.push_str(".vtk");
    let x = &body.x;
    let y = &body.y;
    let r = &body.r;
    let fx = &body.fx;
    let fy = &body.fy;
    let filename = output;

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(filename)
        .unwrap();

    writeln!(file, "# vtk DataFile Version 3.0").unwrap();
    writeln!(file, "Time some").unwrap();
    writeln!(file, "ASCII\nDATASET UNSTRUCTURED_GRID").unwrap();

    writeln!(file, "POINTS {} float", x.len()).unwrap();
    for i in 0..x.len() {
        writeln!(file, "{:.4} {:.4} 0.0", x[i], y[i]).unwrap();
    }

    writeln!(file, "POINT_DATA {}", x.len()).unwrap();
    writeln!(file, "SCALARS Diameter float 1").unwrap();
    writeln!(file, "LOOKUP_TABLE default").unwrap();
    for i in 0..x.len() {
        writeln!(file, "{:.4}", r[i]).unwrap();
    }

    writeln!(file, "VECTORS Force float").unwrap();
    for i in 0..x.len() {
        writeln!(file, "{:.4} {:.4} 0.0000", fx[i], fy[i]).unwrap();
    }
}
