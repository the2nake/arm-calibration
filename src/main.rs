extern crate approx;
extern crate nalgebra as na;

use rand::prelude::*;
use std::{f64::consts::PI, time::Instant};

type ParamVec = na::SVector<f64, 11>;
type SenseVec = na::SVector<f64, 3>;
type SenseMat = na::Matrix<f64, na::U3, na::Dyn, na::VecStorage<f64, na::U3, na::Dyn>>;
type PosVec = na::Vector3<f64>;
type PosMat = na::Matrix3xX<f64>;

const N: usize = 200;
const MAX_ITERS: u32 = 1000;
const RP: i32 = 1;
const KB: usize = 2;
const G0: f64 = 0.1;
const BETA: f64 = 20.0;
const EPSILON: f64 = 1e-30;

struct GravitationalSearch<F: Fn(&ParamVec) -> f64> {
    rng: ThreadRng,
    iter: u32,
    best: ParamVec,
    best_score: f64,
    x: Vec<ParamVec>,
    v: Vec<ParamVec>,
    eval: F,
}

impl<F: Fn(&ParamVec) -> f64> GravitationalSearch<F> {
    pub fn from_dist(prior: &ParamVec, range: &ParamVec, eval: F) -> Self {
        let mut obj = GravitationalSearch {
            rng: rand::rng(),
            iter: 0,
            best: prior.clone(),
            best_score: eval(prior),
            x: vec![prior.clone()],
            v: vec![ParamVec::from_element(0.0)],
            eval,
        };

        obj.x.reserve(N);
        obj.v.reserve(N);
        for _ in 1..N {
            let mut guess = prior.clone();
            for i in 0..guess.nrows() {
                guess[i] += range[i] * rand::random_range(-0.5..=0.5);
            }
            obj.x.push(guess);
            obj.v.push(ParamVec::from_element(0.0));
        }

        obj
    }

    pub fn print(&self) {
        print!("best configuration: {}", self.best().0);
        println!("  with score: {}", self.best().1);
    }

    pub fn iter(&self) -> u32 {
        self.iter
    }

    pub fn best(&self) -> (&ParamVec, f64) {
        (&self.best, self.best_score)
    }

    pub fn step(&mut self) -> bool {
        // evaluate current points, attach their indices like so: (idx, fitness)
        let mut errs: Vec<(usize, f64)> = self
            .x
            .iter()
            .enumerate()
            .map(|(i, x)| (i, (self.eval)(x)))
            .collect();
        let best = errs.iter().min_by(|x, y| x.1.total_cmp(&y.1)).unwrap();
        if best.1 < self.best_score {
            self.best = self.x[best.0];
            self.best_score = best.1;
        }
        let best = best.1;
        let worst = errs.iter().max_by(|x, y| x.1.total_cmp(&y.1)).unwrap().1;

        // compute non-normalised mass values
        let masses: Vec<f64> = errs
            .iter()
            .map(|x| (x.1 - worst) / (best - worst))
            .collect();

        // normalise mass values (seemingly worse)
        // let total_mass: f64 = masses.iter().sum();
        // masses.iter_mut().for_each(|x| *x /= total_mass);

        // attract towards the KB best points
        let top_fitnesses = errs
            .select_nth_unstable_by(KB, |x, y| x.1.total_cmp(&y.1))
            .0;

        let g_iter = G0 * f64::exp(-BETA * (self.iter as f64 / MAX_ITERS as f64));
        let mut accelerations: Vec<ParamVec> = vec![ParamVec::repeat(0.0); N];
        for (i, point) in self.x.iter().enumerate() {
            for (j, _) in top_fitnesses.iter() {
                let dx = self.x[*j] - *point;
                accelerations[i] += self.rng.random_range(0.0..=1.0) * g_iter * masses[*j] * dx
                    / (dx.norm().powi(RP) + EPSILON);
            }
        }

        // normal euler integration
        for i in 0..N {
            self.v[i] += accelerations[i];
            self.x[i] += self.v[i];
        }

        self.iter += 1;
        self.iter < MAX_ITERS
    }
}

struct CalibrationEvaluator<F: Fn(&ParamVec, &SenseVec) -> PosVec> {
    targets: PosMat,
    senses: SenseMat,
    generator: F,
}

impl<F: Fn(&ParamVec, &SenseVec) -> PosVec> CalibrationEvaluator<F> {
    pub fn new(targets: PosMat, senses: SenseMat, generator: F) -> Self {
        if targets.ncols() != senses.ncols() {
            panic!(
                "mismatched targets and sensor values (CalibrationEvaluator): {} != {}",
                targets.ncols(),
                senses.ncols()
            );
        }
        CalibrationEvaluator {
            targets,
            senses,
            generator,
        }
    }

    pub fn effector_positions(&self, params: &ParamVec) -> PosMat {
        let mut positions = PosMat::from_element(self.targets.ncols(), 0.0);
        for i in 0..self.targets.ncols() {
            positions.set_column(
                i,
                &(self.generator)(params, &self.senses.column(i).into_owned()),
            );
        }
        positions
    }

    pub fn eval(&self, params: &ParamVec) -> f64 {
        (&self.targets - self.effector_positions(params))
            .abs()
            .column_sum()
            .norm()
            / (self.targets.ncols() as f64)
    }
}

#[allow(dead_code)]
fn dh_matrix(a: f64, d: f64, alpha: f64, theta: f64) -> na::Matrix4<f64> {
    let ct = theta.cos();
    let st = theta.sin();
    let ca = alpha.cos();
    let sa = alpha.sin();

    #[rustfmt::skip]
    let h = [
        ct, -st * ca, st * sa,  a * ct,
        st, ct * ca,  -ct * sa, a * st,
        0., sa,       ca,       d     ,
        0., 0.,       0.,       1.    ,
    ];
    na::Matrix4::<f64>::from_row_slice(&h)
}

fn generator(params: &ParamVec, senses: &SenseVec) -> PosVec {
    #[rustfmt::skip]
    let j = na::SMatrix::<f64, 3, 11>::from_row_slice(&[
        0., 0., 0., 0., 0., senses[0], 0., 0., 1., 0., 0.,
        0., 0., 0., 0., 0., 0., senses[1], 0., 0., 1., 0.,
        0., 0., 0., 0., 0., 0., 0., senses[2], 0., 0., 1.,
    ]) * params;

    /* matrix fk method, use as reference
    #[rustfmt::skip]
    let mut transform = na::Matrix4::<f64>::from_row_slice(&[
        1., 0., 0., params[0],
        0., 1., 0., params[1],
        0., 0., 1., 0.       ,
        0., 0., 0., 1.       ,
    ]);
    transform *= dh_matrix(0., params[2], PI / 2., PI / 2. + j[0])
        * dh_matrix(params[3], 0., 0., j[1])
        * dh_matrix(params[4], 0., 0., j[2]);

    transform.fixed_view::<3, 1>(0, 3).into()
    */

    let r = params[3] * j[1].cos() + params[4] * (j[1] + j[2]).cos();

    PosVec::new(
        params[0] - j[0].sin() * r,
        params[1] + j[0].cos() * r,
        params[2] + params[3] * j[1].sin() + params[4] * (j[1] + j[2]).sin(),
    )
}

fn main() {
    let train = CalibrationEvaluator::new(
        PosMat::from_columns(&[
            PosVec::new(0.0, 0.0, 0.0),
            PosVec::new(100.0, 0.0, 0.0),
            PosVec::new(100.0, 100.0, 0.0),
            PosVec::new(0.0, 100.0, 0.0),
            PosVec::new(50.0, 50.0, 0.0),
        ]),
        SenseMat::from_columns(&[
            SenseVec::new(5047., 5688., 4450.),
            SenseVec::new(6162., 5686., 4374.),
            SenseVec::new(5830., 5792., 3847.),
            SenseVec::new(5137., 5775., 3905.),
            SenseVec::new(5544., 5710., 4184.),
        ]),
        generator,
    );

    let m1 = (-PI / 2.) / (6497. - 3704.);
    let m2 = (-PI / 4.) / (6123. - 4777.);
    let m3 = (-PI / 4.) / (3760. - 2444.);
    let b1 = 0. - m1 * 3704.;
    let b2 = PI / 2. - m2 * 4777.;
    let b3 = -PI / 2. - m3 * 2444.;
    #[rustfmt::skip]
    let prior_array: [f64; 11] = [30., -130., 62., 314., 333., m1, m2, m3, b1 + f64::to_radians(50.), b2, b3];
    let prior = ParamVec::from_row_slice(&prior_array);

    let err_l = 5.;
    let err_m = 1e-4;
    let err_b = 0.1;
    #[rustfmt::skip]
    let range_array: [f64; 11] = [15., 15., err_l, err_l, err_l, err_m, err_m, err_m, err_b, err_b, err_b];
    let range = ParamVec::from_row_slice(&range_array);

    let mut grav = GravitationalSearch::from_dist(&prior, &range, |x| train.eval(x));

    println!();
    let start = Instant::now();
    while grav.step() {
        print!("[{}] iter complete\r", grav.iter());
    }
    grav.print();
    println!("optimisation took: {} ms", start.elapsed().as_millis());
}
