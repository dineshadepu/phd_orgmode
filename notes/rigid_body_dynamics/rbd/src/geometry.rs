extern crate itertools_num;
use itertools_num::linspace;

pub fn grid_linspace(xl: f32, xr: f32, xnum: usize, yl: f32, yr: f32, ynum: usize) -> (Vec<f32>, Vec<f32>) {
    // create x range
    let x = linspace::<f32>(xl, xr, xnum).collect::<Vec<_>>();

    // create y range
    let y = linspace::<f32>(yl, yr, ynum).collect::<Vec<_>>();

    let mut x_grid = vec![];
    let mut y_grid = vec![];

    for i in 0..y.len() {
        for j in 0..x.len() {
            x_grid.push(x[j]);
            y_grid.push(y[i]);
        }
    }
    (x_grid, y_grid)
}

pub fn arange(left: f32, right: f32, step: f32) -> Vec<f32>{
    let mut x = vec![];
    let mut tmp = left;

    while tmp < right{
        x.push(tmp);
        tmp += step;
    }
    x

}

#[test]
fn test_arange() {
    assert_eq!(vec![0., 1., 2., 3., 4.], arange(0., 5., 1.));
}


pub fn grid_arange(xl: f32, xr: f32, x_spacing: f32, yl: f32, yr: f32, y_spacing: f32) -> (Vec<f32>, Vec<f32>) {
    let x_arange = arange(xl, xr, x_spacing);
    let y_arange = arange(yl, yr, y_spacing);

    let mut x_grid = vec![];
    let mut y_grid = vec![];

    for i in 0..y_arange.len() {
        for j in 0..x_arange.len() {
            x_grid.push(x_arange[j]);
            y_grid.push(y_arange[i]);
        }
    }
    (x_grid, y_grid)
}

pub fn tank(
    xl: f32,
    xr: f32,
    x_spacing: f32,
    yl: f32,
    yr: f32,
    y_spacing: f32,
    layers: usize,
) -> (Vec<f32>, Vec<f32>) {
    let x_arange = arange(xl, xr, x_spacing);
    let y_arange = arange(yl, yr, y_spacing);

    let (xg, yg) = grid_arange(xl, xr, x_spacing, yl, yr, y_spacing);

    // now filter the particles which only belong to tank
    let (mut x, mut y) = (vec![], vec![]);

    let x_left_cutoff = xl + (layers - 1) as f32 * x_spacing + x_spacing / 2.;
    let x_right_cutoff =
        x_arange[x_arange.len() - 1] - (layers - 1) as f32 * x_spacing - x_spacing / 2.;
    let y_bottom_cutoff = yl + (layers - 1) as f32 * y_spacing + y_spacing / 2.;

    for i in 0..xg.len() {
        if xg[i] < x_left_cutoff || xg[i] > x_right_cutoff {
            x.push(xg[i]);
            y.push(yg[i]);
        } else if yg[i] < y_bottom_cutoff {
            x.push(xg[i]);
            y.push(yg[i]);
        }
    }
    (x, y)
}
