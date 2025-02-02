use std::fs::File;
use std::io::{BufWriter, Write};

fn capa(x: f64, k1: f64, k2: f64) -> f64 {
    (k1 - x) / k2
}

fn main() -> std::io::Result<()> {
    let dt = 0.1;
    let num_steps = 10000;

    let mut xs = Vec::with_capacity(num_steps + 1);
    xs.push(0.1);

    for i in 0..num_steps {
        let next_x = xs[i] + capa(xs[i], 51.0, 357.0) * dt;
        xs.push(next_x);
    }

    let max_val = xs.iter().cloned().fold(f64::MIN, |acc, v| acc.max(v.abs()));

    let xs_normalized: Vec<f64> = xs.iter().map(|&v| v / max_val).collect();

    let xs_rounded: Vec<f64> = xs_normalized.iter().map(|&v| (v * 1e9).round() / 1e9).collect();

    let file = File::create("capa_intermittency.dat")?;
    let mut writer = BufWriter::new(file);

    for v in xs_rounded {
        writeln!(writer, "{:.9}", v)?;
    }

    Ok(())
}
