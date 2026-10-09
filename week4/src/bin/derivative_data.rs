use std::f64::consts::PI;
use week4::flow::SpectralFlow;

fn wave(n: usize) -> Vec<f64> {
    (0..n).flat_map(|j| (0..n).map(move |i| {
        (3.0*2.0*PI*i as f64/n as f64).sin()*(2.0*2.0*PI*j as f64/n as f64).cos()
    })).collect()
}

fn errors(n: usize) -> ([f64;4],[f64;4]) {
    let flow = SpectralFlow::new(n);
    let g = wave(n);
    let spectral = [flow.derivative(&g,1,0),flow.derivative(&g,2,0),
        flow.derivative(&g,1,1),flow.laplacian(&g)];
    let mut spec_error = [0.0_f64;4];
    let mut fd_error = [0.0_f64;4];
    let h = 2.0*PI/n as f64;
    for j in 0..n { for i in 0..n {
        let p = j*n+i;
        let l = j*n+(i+n-1)%n;
        let r = j*n+(i+1)%n;
        let d = ((j+n-1)%n)*n+i;
        let u = ((j+1)%n)*n+i;
        let ld = ((j+n-1)%n)*n+(i+n-1)%n;
        let lu = ((j+1)%n)*n+(i+n-1)%n;
        let rd = ((j+n-1)%n)*n+(i+1)%n;
        let ru = ((j+1)%n)*n+(i+1)%n;
        let x = i as f64*h;
        let y = j as f64*h;
        let expected = [3.0*(3.0*x).cos()*(2.0*y).cos(),-9.0*g[p],
            -6.0*(3.0*x).cos()*(2.0*y).sin(),-13.0*g[p]];
        let fd = [(g[r]-g[l])/(2.0*h),(g[r]-2.0*g[p]+g[l])/(h*h),
            (g[ru]-g[rd]-g[lu]+g[ld])/(4.0*h*h),
            (g[r]+g[l]+g[u]+g[d]-4.0*g[p])/(h*h)];
        for k in 0..4 {
            spec_error[k] = spec_error[k].max((spectral[k][p]-expected[k]).abs());
            fd_error[k] = fd_error[k].max((fd[k]-expected[k]).abs());
        }
    }}
    (fd_error,spec_error)
}

fn main() {
    let (fd32,spec32) = errors(32);
    let (fd64,_) = errors(64);
    println!("derivative\tFD n=32\tFD n=64\tratio\tFourier n=32");
    for k in 0..4 {
        println!("{}\t{:.8}\t{:.8}\t{:.3}\t{:.3e}",
            ["dx","dxx","dxdy","laplacian"][k],fd32[k],fd64[k],fd32[k]/fd64[k],spec32[k]);
    }
}
