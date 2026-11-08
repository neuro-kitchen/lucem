//! Command-line interface of lumen: run pipelines over images, check devices, serve images.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use lumen_core::compute::ComputeTask;
use lumen_core::cubecl::prelude::Client;
use lumen_core::memory::DeviceArray;
use lumen_core::{ComputeTarget, Shape};
use lumen_io::Pyramid;
use lumen_pipeline::{run_spec, PipelineSpec, RunOptions};
use lumen_stream::{client_config_with, server_config_with_clients, Authority, ClientIdentity, Permission, ServeOptions, Server, ServerIdentity, ServerTrust, Session};

const USAGE: &str = "lumen — GPU image analysis

USAGE
  lumen info <image>                     Describe an image
  lumen doctor                           Which compute runtimes are usable
  lumen run <spec.json> <image> <output> [--block N] [--runtime NAME]
                                         Run a pipeline over a whole image
  lumen sample <output.ome.zarr> [--size N] [--depth N]
                                         Write a synthetic image (a volume with --depth)
  lumen serve [--data DIR] [--bind ADDR] [--name HOST]... [--max-clients N] [--allow-anonymous]
              [--certificate FILE --private-key FILE]
                                         Serve the images under DIR to enrolled clients
  lumen clients enroll <name> [--control] --out FILE|-   (--auth DIR)
  lumen clients list | revoke <name>                     (--auth DIR)
                                         Who may connect to `serve`
  lumen fetch <host:port> --identity FILE --server-name NAME [--open PATH]
                                         List a server's images, or read a tile of one
";

fn main() -> ExitCode {
    let _ = tracing_subscriber::fmt().with_env_filter("warn").with_writer(std::io::stderr).try_init();
    let _ = rustls::crypto::ring::default_provider().install_default();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Arguments: flags with values, on/off flags, and what remains in order.
struct Args {
    rest: Vec<String>,
}

impl Args {
    fn take_value(&mut self, name: &str) -> Result<Option<String>, String> {
        let Some(i) = self.rest.iter().position(|a| a == name) else { return Ok(None) };
        if i + 1 >= self.rest.len() {
            return Err(format!("{name} needs a value"));
        }
        self.rest.remove(i);
        Ok(Some(self.rest.remove(i)))
    }

    fn take_values(&mut self, name: &str) -> Result<Vec<String>, String> {
        let mut out = Vec::new();
        while let Some(v) = self.take_value(name)? {
            out.push(v);
        }
        Ok(out)
    }

    fn take_flag(&mut self, name: &str) -> bool {
        match self.rest.iter().position(|a| a == name) {
            Some(i) => {
                self.rest.remove(i);
                true
            }
            None => false,
        }
    }

    fn number(&mut self, name: &str) -> Result<Option<usize>, String> {
        self.take_value(name)?.map(|v| v.parse().map_err(|_| format!("{name}: {v:?} is not a number"))).transpose()
    }

    /// Nothing but `n` positional arguments may remain.
    fn positional(self, n: usize) -> Result<Vec<String>, String> {
        if let Some(flag) = self.rest.iter().find(|a| a.starts_with("--")) {
            return Err(format!("unknown option {flag}"));
        }
        if self.rest.len() != n {
            return Err(format!("expected {n} argument(s), got {}\n\n{USAGE}", self.rest.len()));
        }
        Ok(self.rest)
    }
}

fn run(mut args: Vec<String>) -> Result<(), String> {
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" || args[0] == "help" {
        print!("{USAGE}");
        return Ok(());
    }
    if args[0] == "--version" {
        println!("lumen {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    let command = args.remove(0);
    let args = Args { rest: args };
    match command.as_str() {
        "info" => info(args),
        "doctor" => doctor(args),
        "run" => run_pipeline(args),
        "sample" => sample(args),
        "serve" => serve(args),
        "clients" => clients(args),
        "fetch" => fetch(args),
        other => Err(format!("unknown command {other:?}\n\n{USAGE}")),
    }
}

fn info(args: Args) -> Result<(), String> {
    let a = args.positional(1)?;
    let p = Pyramid::open(Path::new(&a[0])).map_err(|e| e.to_string())?;
    println!("{}", p.name);
    println!("  {} × {} × {} (z, y, x), {}", p.depth(), p.height(), p.width(), p.dtype);
    if let Some(u) = p.pixel_size_um {
        println!("  pixel {u} µm");
    }
    if let Some(z) = p.z_spacing_um {
        println!("  z spacing {z} µm");
    }
    for c in &p.channels {
        println!("  channel {}", c.label);
    }
    for (i, l) in p.levels.iter().enumerate() {
        println!("  level {i}: {} × {} × {}, tiles {} × {} × {}", l.depth, l.height, l.width, l.tile_d, l.tile_h, l.tile_w);
    }
    Ok(())
}

/// A small computation that proves a runtime works.
struct Probe;

impl ComputeTask for Probe {
    type Output = bool;
    fn run(self, client: Client) -> bool {
        let Ok(a) = DeviceArray::<f32>::from_host(&client, Shape::new([4, 4]), vec![1.0; 16]) else { return false };
        lumen_ops::filter::gaussian::gaussian_blur(&client, &a, &[1.0, 1.0], &Default::default()).is_ok_and(|b| b.to_host(&client).iter().all(|v| (v - 1.0).abs() < 1e-3))
    }
}

fn doctor(args: Args) -> Result<(), String> {
    args.positional(0)?;
    let mut selected_works = false;
    let selected = ComputeTarget::from_env().map_err(|e| e.to_string())?;
    for target in ComputeTarget::ALL {
        if !target.is_available() {
            println!("{target}: not compiled in");
            continue;
        }
        // A runtime without its driver panics when opened
        let result = std::panic::catch_unwind(|| target.run(Probe));
        let ok = matches!(result, Ok(Ok(true)));
        println!("{target}: {}", if ok { "usable" } else { "not usable (no device or driver)" });
        if target == selected {
            selected_works = ok;
        }
    }
    println!("Selected: {selected}");
    if !selected_works {
        return Err(format!("the selected runtime {selected} has no device: set LUMEN_RUNTIME to a usable one"));
    }
    Ok(())
}

struct RunTask<'a> {
    spec: &'a PipelineSpec,
    image: Arc<Pyramid>,
    output: PathBuf,
    block: Option<usize>,
}

impl ComputeTask for RunTask<'_> {
    type Output = Result<(), String>;
    fn run(self, client: Client) -> Self::Output {
        let progress = Arc::new(|done: usize, total: usize| {
            if done == total || done % 16 == 0 {
                eprintln!("{done} / {total} blocks");
            }
        });
        let options = RunOptions { block: self.block, progress: Some(progress), ..RunOptions::default() };
        let done = run_spec(&client, &self.image, self.spec, &self.output, options).map_err(|e| e.to_string())?;
        for w in &done.warnings {
            eprintln!("warning: {w}");
        }
        println!("wrote {} ({} blocks)", done.output.display(), done.blocks);
        if let Some(n) = done.objects {
            println!("{n} objects");
        }
        println!("spec: {}", done.spec.display());
        Ok(())
    }
}

fn run_pipeline(mut args: Args) -> Result<(), String> {
    let block = args.number("--block")?;
    let runtime = args.take_value("--runtime")?;
    let a = args.positional(3)?;
    let text = std::fs::read_to_string(&a[0]).map_err(|e| format!("{}: {e}", a[0]))?;
    let spec: PipelineSpec = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", a[0]))?;
    let image = Arc::new(Pyramid::open(Path::new(&a[1])).map_err(|e| e.to_string())?);
    let target = match runtime {
        Some(r) => ComputeTarget::parse(&r).and_then(ComputeTarget::checked).map_err(|e| e.to_string())?,
        None => ComputeTarget::from_env().map_err(|e| e.to_string())?,
    };
    target.run(RunTask { spec: &spec, image, output: PathBuf::from(&a[2]), block }).map_err(|e| e.to_string())?
}

fn sample(mut args: Args) -> Result<(), String> {
    let size = args.number("--size")?.unwrap_or(256);
    let depth = args.number("--depth")?;
    let a = args.positional(1)?;
    let py = lumen_io::sample_volume(size, depth.unwrap_or(8)).map_err(|e| e.to_string())?;
    let d = if depth.is_some() { py.depth() } else { 1 };
    let (h, w) = (py.height(), py.width());
    let options = lumen_io::WriterOptions { name: py.name.clone(), channels: py.channels.iter().map(|c| c.label.clone()).collect(), shape: [d, h, w], chunks: [d.min(32), 128.min(h), 128.min(w)], pixel_size_um: py.pixel_size_um, z_spacing_um: py.z_spacing_um };
    let chunks = options.chunks;
    let writer = lumen_io::ZarrWriter::create::<f32>(&a[0], options).map_err(|e| e.to_string())?;
    for b in lumen_core::tiling::schedule([d, h, w], chunks, [0; 3]) {
        for c in 0..py.channels.len() {
            // A 2D sample is the volume's middle plane
            let z = if d == 1 { py.depth() / 2 } else { b.origin[0] };
            let values = py.read_block(0, c, [z, b.origin[1], b.origin[2]], b.size).map_err(|e| e.to_string())?;
            writer.write(c, b.origin, b.size, &values).map_err(|e| e.to_string())?;
        }
    }
    println!("wrote {}", writer.finish::<f32>().map_err(|e| e.to_string())?.display());
    Ok(())
}

fn auth_dir(args: &mut Args) -> Result<PathBuf, String> {
    Ok(args.take_value("--auth")?.map(PathBuf::from).or_else(|| std::env::var_os("LUMEN_SERVE_AUTH_DIR").map(PathBuf::from)).unwrap_or_else(|| PathBuf::from(".lumen-auth")))
}

fn runtime() -> Result<tokio::runtime::Runtime, String> {
    tokio::runtime::Runtime::new().map_err(|e| e.to_string())
}

fn serve(mut args: Args) -> Result<(), String> {
    let dir = auth_dir(&mut args)?;
    let data = args.take_value("--data")?.or_else(|| std::env::var("LUMEN_SERVE_DATA").ok()).unwrap_or_else(|| ".".into());
    let bind = args.take_value("--bind")?.unwrap_or_else(|| "127.0.0.1:50051".into());
    let mut names = args.take_values("--name")?;
    names.insert(0, "localhost".into());
    let max = args.number("--max-clients")?.unwrap_or(lumen_stream::DEFAULT_MAX_CONNECTIONS);
    let anonymous = args.take_flag("--allow-anonymous");
    let certificate = args.take_value("--certificate")?;
    let key = args.take_value("--private-key")?;
    args.positional(0)?;
    let identity = match (certificate, key) {
        (Some(c), Some(k)) => ServerIdentity::Files { certificate_chain: c.into(), private_key: k.into() },
        (None, None) => ServerIdentity::SelfSigned { names },
        _ => return Err("--certificate and --private-key go together".into()),
    };
    let addr = bind.parse().map_err(|e| format!("--bind {bind}: {e}"))?;
    let authority = Arc::new(Authority::open_or_create(&dir).map_err(|e| e.to_string())?);
    let tls = server_config_with_clients(&authority, &identity, anonymous).map_err(|e| e.to_string())?;
    runtime()?.block_on(async move {
        let server = Server::bind(addr, tls.config).map_err(|e| e.to_string())?.with_authority(authority).with_max_connections(max);
        eprintln!("serving {data} on {addr} to enrolled clients");
        tokio::select! {
            r = server.run(ServeOptions { data: PathBuf::from(&data) }) => r.map_err(|e| e.to_string()),
            _ = shutdown() => { Ok(()) }
        }
    })
}

/// SIGINT or SIGTERM (podman stop, systemd).
async fn shutdown() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        if let Ok(mut term) = signal(SignalKind::terminate()) {
            tokio::select! {
                _ = term.recv() => {}
                _ = tokio::signal::ctrl_c() => {}
            }
            return;
        }
    }
    let _ = tokio::signal::ctrl_c().await;
}

fn clients(mut args: Args) -> Result<(), String> {
    let dir = auth_dir(&mut args)?;
    let authority = Authority::open_or_create(&dir).map_err(|e| e.to_string())?;
    if args.rest.is_empty() {
        return Err(USAGE.into());
    }
    let what = args.rest.remove(0);
    match what.as_str() {
        "enroll" => {
            let permission = if args.take_flag("--control") { Permission::Control } else { Permission::Read };
            let out = args.take_value("--out")?.ok_or("--out FILE (or - for standard output) is required")?;
            let a = args.positional(1)?;
            let e = authority.enroll(&a[0], permission).map_err(|e| e.to_string())?;
            if out == "-" {
                use std::io::Write as _;
                std::io::stdout().write_all(e.pem.as_bytes()).map_err(|e| e.to_string())?;
            } else {
                e.write(Path::new(&out)).map_err(|e| e.to_string())?;
                eprintln!("enrolled {} ({out}: hand it over privately, it holds the client's key)", a[0]);
            }
            Ok(())
        }
        "list" => {
            args.positional(0)?;
            for c in authority.clients().map_err(|e| e.to_string())? {
                println!("{} {:?}{}", c.name, c.permission, if c.active() { "" } else { " (revoked)" });
            }
            Ok(())
        }
        "revoke" => {
            let a = args.positional(1)?;
            authority.revoke(&a[0]).map_err(|e| e.to_string())?;
            eprintln!("revoked {}", a[0]);
            Ok(())
        }
        other => Err(format!("unknown clients command {other:?}")),
    }
}

fn fetch(mut args: Args) -> Result<(), String> {
    let identity = args.take_value("--identity")?.ok_or("--identity FILE is required")?;
    let name = args.take_value("--server-name")?.unwrap_or_else(|| "localhost".into());
    let open = args.take_value("--open")?;
    let a = args.positional(1)?;
    let addr = a[0].parse().map_err(|e| format!("{}: {e}", a[0]))?;
    let identity = ClientIdentity::from_file(Path::new(&identity)).map_err(|e| e.to_string())?;
    let config = client_config_with(&ServerTrust::from_identity(&identity), Some(&identity)).map_err(|e| e.to_string())?;
    runtime()?.block_on(async move {
        let session = Session::connect(addr, &name, config).await.map_err(|e| e.to_string())?;
        match open {
            None => {
                for p in session.list().await.map_err(|e| e.to_string())? {
                    println!("{p}");
                }
            }
            Some(path) => {
                let info = session.info(&path).await.map_err(|e| e.to_string())?;
                println!("{} channels {:?} levels {:?}", info.name, info.channels, info.levels);
                let t = session.tile(&path, 0, 0, 0, 0, 0).await.map_err(|e| e.to_string())?;
                let mean = t.data.iter().map(|v| *v as f64).sum::<f64>() / t.data.len().max(1) as f64;
                println!("tile 0,0: {} × {}, mean {mean}", t.width, t.height);
            }
        }
        session.close();
        Ok(())
    })
}
