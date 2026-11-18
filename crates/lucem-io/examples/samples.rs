//! Writes the app's samples as OME-Zarr (`lucem_io::samples`): what the page's build puts next to
//! the page (`cargo run --release -p lucem-io --example samples -- <dir>`), without building the
//! whole command line.

fn main() {
    let Some(dir) = std::env::args().nth(1) else {
        eprintln!("usage: samples <dir>");
        std::process::exit(2);
    };
    match lucem_io::samples::write_all(std::path::Path::new(&dir)) {
        Ok(written) => println!("wrote {} sample(s) into {dir}", written.len()),
        Err(e) => {
            eprintln!("samples: {e}");
            std::process::exit(1);
        }
    }
}
