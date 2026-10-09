//! Fourier pseudospectral operators on a periodic square.

use rustfft::{Fft, FftPlanner, num_complex::Complex64};
use std::{f64::consts::PI, sync::Arc};

pub struct SpectralFlow {
    pub n: usize,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
}

impl SpectralFlow {
    pub fn new(n: usize) -> Self {
        assert!(n >= 4 && n % 2 == 0);
        let mut planner = FftPlanner::new();
        Self { n, forward: planner.plan_fft_forward(n), inverse: planner.plan_fft_inverse(n) }
    }

    fn wave(&self, index: usize) -> i32 {
        if index < self.n / 2 { index as i32 } else { index as i32 - self.n as i32 }
    }

    fn transform(&self, data: &mut [Complex64], inverse: bool) {
        let fft = if inverse { &self.inverse } else { &self.forward };
        for row in data.chunks_exact_mut(self.n) { fft.process(row); }
        let mut column = vec![Complex64::default(); self.n];
        for x in 0..self.n {
            for y in 0..self.n { column[y] = data[y*self.n+x]; }
            fft.process(&mut column);
            for y in 0..self.n { data[y*self.n+x] = column[y]; }
        }
        if inverse {
            let scale = (self.n*self.n) as f64;
            for value in data { *value /= scale; }
        }
    }

    pub fn fft(&self, field: &[f64]) -> Vec<Complex64> {
        assert_eq!(field.len(), self.n*self.n);
        let mut result = field.iter().map(|&x| Complex64::new(x, 0.0)).collect::<Vec<_>>();
        self.transform(&mut result, false);
        result
    }

    pub fn ifft(&self, modes: &[Complex64]) -> Vec<f64> {
        let mut result = modes.to_vec();
        self.transform(&mut result, true);
        result.iter().map(|z| z.re).collect()
    }

    pub fn truncate(&self, modes: &mut [Complex64]) {
        let cutoff = self.n as i32 / 3;
        for y in 0..self.n { for x in 0..self.n {
            if self.wave(x).abs() > cutoff || self.wave(y).abs() > cutoff {
                modes[y*self.n+x] = Complex64::default();
            }
        }}
    }

    pub fn project(&self, field: &[f64]) -> Vec<f64> {
        let mut modes = self.fft(field);
        self.truncate(&mut modes);
        self.ifft(&modes)
    }

    pub fn derivative(&self, field: &[f64], x_order: u32, y_order: u32) -> Vec<f64> {
        let mut modes = self.fft(field);
        for y in 0..self.n { for x in 0..self.n {
            let kx = self.wave(x) as f64;
            let ky = self.wave(y) as f64;
            modes[y*self.n+x] *= Complex64::new(0.0, kx).powu(x_order)
                * Complex64::new(0.0, ky).powu(y_order);
        }}
        self.ifft(&modes)
    }

    pub fn laplacian(&self, field: &[f64]) -> Vec<f64> {
        let mut modes = self.fft(field);
        for y in 0..self.n { for x in 0..self.n {
            let k2 = (self.wave(x).pow(2) + self.wave(y).pow(2)) as f64;
            modes[y*self.n+x] *= -k2;
        }}
        self.ifft(&modes)
    }

    pub fn vorticity(&self, u: &[f64], v: &[f64]) -> Vec<f64> {
        let vx = self.derivative(v, 1, 0);
        let uy = self.derivative(u, 0, 1);
        let omega = vx.iter().zip(uy).map(|(a,b)| a-b).collect::<Vec<_>>();
        self.project(&omega)
    }

    fn velocity_and_gradients(&self, omega: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut modes = self.fft(omega);
        self.truncate(&mut modes);
        let mut u = modes.clone();
        let mut v = modes.clone();
        let mut wx = modes.clone();
        let mut wy = modes.clone();
        for y in 0..self.n { for x in 0..self.n {
            let kx = self.wave(x) as f64;
            let ky = self.wave(y) as f64;
            let k2 = kx*kx + ky*ky;
            let p = y*self.n+x;
            if k2 == 0.0 { u[p] = Complex64::default(); v[p] = Complex64::default(); }
            else {
                u[p] *= Complex64::new(0.0, ky/k2);
                v[p] *= Complex64::new(0.0, -kx/k2);
            }
            wx[p] *= Complex64::new(0.0, kx);
            wy[p] *= Complex64::new(0.0, ky);
        }}
        (self.ifft(&u), self.ifft(&v), self.ifft(&wx), self.ifft(&wy))
    }

    pub fn velocity(&self, omega: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (u,v,_,_) = self.velocity_and_gradients(omega);
        (u,v)
    }

    pub fn rate(&self, omega: &[f64], nu: f64) -> Vec<f64> {
        let (u,v,wx,wy) = self.velocity_and_gradients(omega);
        let nonlinear = (0..omega.len()).map(|p| u[p]*wx[p] + v[p]*wy[p]).collect::<Vec<_>>();
        let mut product_modes = self.fft(&nonlinear);
        self.truncate(&mut product_modes);
        let mut omega_modes = self.fft(omega);
        self.truncate(&mut omega_modes);
        for y in 0..self.n { for x in 0..self.n {
            let p = y*self.n+x;
            let k2 = (self.wave(x).pow(2) + self.wave(y).pow(2)) as f64;
            product_modes[p] = -product_modes[p] - nu*k2*omega_modes[p];
        }}
        self.ifft(&product_modes)
    }

    pub fn energy(&self, u: &[f64], v: &[f64]) -> f64 {
        u.iter().zip(v).map(|(a,b)| a*a+b*b).sum::<f64>() / (2*self.n*self.n) as f64
    }

    pub fn enstrophy(&self, omega: &[f64]) -> f64 {
        omega.iter().map(|w| w*w).sum::<f64>() / (2*self.n*self.n) as f64
    }

    pub fn taylor_green(&self, nu: f64, t: f64) -> (Vec<f64>, Vec<f64>) {
        let amplitude = (-2.0*nu*t).exp();
        let mut u = Vec::with_capacity(self.n*self.n);
        let mut v = Vec::with_capacity(self.n*self.n);
        for y in 0..self.n { for x in 0..self.n {
            let xx = 2.0*PI*x as f64/self.n as f64;
            let yy = 2.0*PI*y as f64/self.n as f64;
            u.push(xx.cos()*yy.sin()*amplitude);
            v.push(-xx.sin()*yy.cos()*amplitude);
        }}
        (u,v)
    }
}
