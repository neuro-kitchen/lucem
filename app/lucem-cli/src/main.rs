//! Command-line interface of lucem: run pipelines over images, check devices, serve images.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use lucem_core::compute::ComputeTask;
use lucem_core::cubecl::prelude::Client;
use lucem_core::memory::DeviceArray;
use lucem_core::{ComputeTarget, Shape};
use lucem_io::Pyramid;
use lucem_pipeline::PipelineSpec;
use lucem_stream::{client_config_with, server_config_with_clients, Authority, ClientIdentity, Permission, ServeOptions, Server, ServerIdentity, ServerTrust, Session};

const USAGE: &str = "lucem — GPU image analysis

USAGE
  lucem info <image>                     Describe an image
  lucem doctor                           Which compute runtimes are usable
  lucem run <spec.json> <image> <output> [--block N] [--runtime NAME]
                                         Run a pipeline over a whole image
  lucem sample <output.ome.zarr> [--size N] [--depth N]
                                         Write a synthetic image (a volume with --depth)
  lucem serve [--data DIR] [--bind ADDR] [--name HOST]... [--max-clients N] [--allow-anonymous]
              [--pair] [--certificate FILE --private-key FILE]
                                         Serve the images under DIR to enrolled clients
                                         (--pair: show a pairing code on the terminal at start)
  lucem clients pair [--control] [--out FILE]            (--auth DIR)
                                         A one-time code to pair a computer (10 minutes), shown
                                         on the terminal (--out: a file only you can read)
  lucem clients codes [--cancel ID|all]                  (--auth DIR)
                                         The live pairing codes (no secrets); cancel them
  lucem clients enroll <name> [--control] --out FILE|-   (--auth DIR)
  lucem clients list | revoke <name>                     (--auth DIR)
                                         Who may connect to `serve`
  lucem pair <host:port> <code> [--name NAME] [--out FILE]
                                         Pair this computer with a server (its key is made here)
  lucem fetch <host:port> --identity FILE --server-name NAME [--open PATH]
                                         List a server's images, or read a tile of one
  lucem hist stains | tissue | cells <image> …
                                         Histology: stain vectors, tissue, cells (lucem hist)
  lucem ops [--json]                     Every pipeline operation: family, types, parameters
  lucem mcp [--runtime NAME]             lucem's tools for an agent (MCP over stdio), e.g.
                                         claude mcp add lucem -- lucem mcp
  lucem session run <session.json> <image>... [--out DIR] [--runtime NAME]
                                         Run an analysis saved by the app on many images
";

mod hist;
mod ops;
mod session;

fn main() -> ExitCode {
    lucem_ml::legacy::adopt();
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
pub(crate) struct Args {
    pub(crate) rest: Vec<String>,
}

impl Args {
    pub(crate) fn take_value(&mut self, name: &str) -> Result<Option<String>, String> {
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

    pub(crate) fn take_flag(&mut self, name: &str) -> bool {
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
    pub(crate) fn positional(self, n: usize) -> Result<Vec<String>, String> {
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
        println!("lucem {}", env!("CARGO_PKG_VERSION"));
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
        "pair" => pair(args),
        "hist" => hist::hist(args),
        "session" => session::session(args),
        "ops" => ops::ops(args),
        "mcp" => mcp(args),
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
        lucem_ops::filter::gaussian::gaussian_blur(&client, &a, &[1.0, 1.0], &Default::default()).is_ok_and(|b| b.to_host(&client).iter().all(|v| (v - 1.0).abs() < 1e-3))
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
        if let Some(why) = target.unusable_reason() {
            println!("{target}: not usable ({why})");
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
        return Err(format!("the selected runtime {selected} has no device: set LUCEM_RUNTIME to a usable one"));
    }
    Ok(())
}

fn run_pipeline(mut args: Args) -> Result<(), String> {
    let block = args.number("--block")?;
    let runtime = args.take_value("--runtime")?;
    let a = args.positional(3)?;
    let text = std::fs::read_to_string(&a[0]).map_err(|e| format!("{}: {e}", a[0]))?;
    let spec: PipelineSpec = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", a[0]))?;
    let target = match runtime {
        Some(r) => ComputeTarget::parse(&r).and_then(ComputeTarget::checked).map_err(|e| e.to_string())?,
        None => ComputeTarget::from_env().map_err(|e| e.to_string())?,
    };
    let progress: std::sync::Arc<dyn Fn(usize, usize) + Send + Sync> = Arc::new(|done: usize, total: usize| {
        if done == total || done.is_multiple_of(16) {
            eprintln!("{done} / {total} blocks");
        }
    });
    let done = lucem_mcp::work::run_pipeline(&target, &spec, Path::new(&a[1]), Path::new(&a[2]), block, Some(progress))?;
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

fn sample(mut args: Args) -> Result<(), String> {
    let size = args.number("--size")?.unwrap_or(256);
    let depth = args.number("--depth")?;
    let a = args.positional(1)?;
    let py = lucem_io::sample_volume(size, depth.unwrap_or(8)).map_err(|e| e.to_string())?;
    let d = if depth.is_some() { py.depth() } else { 1 };
    let (h, w) = (py.height(), py.width());
    let options = lucem_io::WriterOptions { name: py.name.clone(), channels: py.channels.iter().map(|c| c.label.clone()).collect(), shape: [d, h, w], chunks: [d.min(32), 128.min(h), 128.min(w)], pixel_size_um: py.pixel_size_um, z_spacing_um: py.z_spacing_um };
    let chunks = options.chunks;
    let writer = lucem_io::ZarrWriter::create::<f32>(&a[0], options).map_err(|e| e.to_string())?;
    for b in lucem_core::tiling::schedule([d, h, w], chunks, [0; 3]) {
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
    Ok(args.take_value("--auth")?.map(PathBuf::from).or_else(|| std::env::var_os("LUCEM_SERVE_AUTH_DIR").map(PathBuf::from)).unwrap_or_else(|| PathBuf::from(".lucem-auth")))
}

/// Serves lucem's tools to an agent over stdin/stdout (MCP). Nothing else may print to stdout.
fn mcp(mut args: Args) -> Result<(), String> {
    let name = args.take_value("--runtime")?;
    args.positional(0)?;
    runtime()?.block_on(lucem_mcp::serve_stdio(name))
}

fn runtime() -> Result<tokio::runtime::Runtime, String> {
    tokio::runtime::Runtime::new().map_err(|e| e.to_string())
}

fn serve(mut args: Args) -> Result<(), String> {
    let dir = auth_dir(&mut args)?;
    let data = args.take_value("--data")?.or_else(|| std::env::var("LUCEM_SERVE_DATA").ok()).unwrap_or_else(|| ".".into());
    let bind = args.take_value("--bind")?.unwrap_or_else(|| "127.0.0.1:50051".into());
    let mut names = args.take_values("--name")?;
    names.insert(0, "localhost".into());
    let max = args.number("--max-clients")?.unwrap_or(lucem_stream::DEFAULT_MAX_CONNECTIONS);
    let anonymous = args.take_flag("--allow-anonymous");
    let pair_now = args.take_flag("--pair");
    let certificate = args.take_value("--certificate")?;
    let key = args.take_value("--private-key")?;
    args.positional(0)?;
    let pair_names = names.clone();
    let identity = match (certificate, key) {
        (Some(c), Some(k)) => ServerIdentity::Files { certificate_chain: c.into(), private_key: k.into() },
        (None, None) => ServerIdentity::SelfSigned { names },
        _ => return Err("--certificate and --private-key go together".into()),
    };
    let addr = bind.parse().map_err(|e| format!("--bind {bind}: {e}"))?;
    let authority = Arc::new(Authority::open_or_create(&dir).map_err(|e| e.to_string())?);
    let tls = server_config_with_clients(&authority, &identity, anonymous).map_err(|e| e.to_string())?;
    // No pairing code unless asked for: a code is a secret, and one made at every start would be
    // a live secret nobody asked for. `--pair` shows one on the terminal (never in a log).
    let terminal = if pair_now { Some(Terminal::open().map_err(|e| format!("--pair: {e}"))?) } else { None };
    runtime()?.block_on(async move {
        let server = Server::bind(addr, tls.config).map_err(|e| e.to_string())?.with_authority(authority.clone()).with_anonymous(anonymous).with_names(pair_names).with_max_connections(max);
        eprintln!("serving {data} on {addr} to paired computers");
        match terminal {
            Some(mut t) => {
                let code = authority.new_pairing_code(Permission::Read).map_err(|e| e.to_string())?;
                t.show(&format!("pair a computer (read only, {} minutes, once): {addr}  code {}\n  in lucem: the server button › Pair with a lucem server…   or: lucem pair {addr} {}", lucem_stream::pairing::CODE_MINUTES, code.display(), code.display()))?;
            }
            None => eprintln!("pair a computer: `lucem clients pair` (in a container: podman exec -it <container> lucem clients pair)"),
        }
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
        "pair" => {
            let permission = if args.take_flag("--control") { Permission::Control } else { Permission::Read };
            let out = args.take_value("--out")?;
            args.positional(0)?;
            // Shown on the terminal itself, never on standard output: a redirect, a pipe or a log
            // never holds it. A script takes it from a file only its owner can read (`--out`).
            let mut terminal = if out.is_some() { None } else { Some(Terminal::open().map_err(|e| format!("{e}; or pass --out FILE for a script (written readable by you only)"))?) };
            let code = authority.new_pairing_code(permission).map_err(|e| e.to_string())?;
            match (&mut terminal, &out) {
                (Some(t), _) => t.show(&code.display())?,
                (None, Some(path)) => {
                    lucem_stream::pairing::save_code(Path::new(path), &code).map_err(|e| format!("{path}: {e}"))?;
                    eprintln!("wrote the code to {path} (readable by you only; remove it once used)");
                }
                (None, None) => unreachable!("no terminal is only taken with --out"),
            }
            eprintln!("works once, for {} minutes ({permission}); a running `lucem serve` accepts it at once; `lucem clients codes --cancel {}` cancels it", lucem_stream::pairing::CODE_MINUTES, code.id);
            Ok(())
        }
        "codes" => {
            let cancel = args.take_value("--cancel")?;
            args.positional(0)?;
            match cancel.as_deref() {
                Some(which) => {
                    let n = authority.cancel_pairing_codes(if which == "all" { None } else { Some(which) }).map_err(|e| e.to_string())?;
                    if n == 0 && which != "all" {
                        return Err(format!("no live pairing code {which:?} (see lucem clients codes)"));
                    }
                    eprintln!("cancelled {n} pairing code(s)");
                }
                None => {
                    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
                    let codes = authority.pairing_codes().map_err(|e| e.to_string())?;
                    if codes.is_empty() {
                        eprintln!("no live pairing codes");
                    }
                    for c in codes {
                        println!("{}  {}  {} min left  {} tries left", c.id, c.permission, c.expires_unix.saturating_sub(now).div_ceil(60), c.tries_left);
                    }
                }
            }
            Ok(())
        }
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

/// The terminal itself (`/dev/tty`; `CONOUT$` on Windows), where pairing codes are shown: what
/// is written there is never in a redirect, a pipe or a service's log.
struct Terminal(std::fs::File);

impl Terminal {
    fn open() -> Result<Self, String> {
        let path = if cfg!(windows) { "CONOUT$" } else { "/dev/tty" };
        std::fs::OpenOptions::new().write(true).open(path).map(Terminal).map_err(|_| "a pairing code is shown only on a terminal, and there is none here (in a container: podman exec -it)".to_string())
    }

    fn show(&mut self, text: &str) -> Result<(), String> {
        use std::io::Write as _;
        writeln!(self.0, "{text}").and_then(|_| self.0.flush()).map_err(|e| format!("the terminal: {e}"))
    }
}

/// Pairs this computer with a server: its key is made here and only the public key is sent; the
/// identity (certificate, key, the server's CA) is written readable by this user only.
fn pair(mut args: Args) -> Result<(), String> {
    let name = args.take_value("--name")?.unwrap_or_else(computer_name);
    let out = args.take_value("--out")?;
    let a = args.positional(2)?;
    let addr: std::net::SocketAddr = a[0].parse().map_err(|e| format!("{}: {e} (host:port, e.g. 10.0.0.5:50051)", a[0]))?;
    let paired = runtime()?.block_on(lucem_stream::pair(addr, &a[1], &name)).map_err(|e| e.to_string())?;
    let out = out.map(PathBuf::from).unwrap_or_else(|| servers_dir().join(format!("{}.pem", a[0].replace([':', '[', ']'], "_"))));
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    lucem_stream::pairing::save_identity(&out, &paired.identity_pem).map_err(|e| e.to_string())?;
    eprintln!("paired as {name} ({}); identity in {}", paired.permission, out.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned()));
    eprintln!("connect with: lucem fetch {} --identity <that file> --server-name {}", a[0], paired.names.first().map_or("localhost", String::as_str));
    Ok(())
}

/// This computer's name (for the server's list of paired computers).
fn computer_name() -> String {
    let raw = std::fs::read_to_string("/etc/hostname").ok().or_else(|| std::env::var("COMPUTERNAME").ok()).or_else(|| std::env::var("HOSTNAME").ok()).unwrap_or_else(|| "computer".into());
    let clean: String = raw.trim().chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') { c } else { '-' }).collect();
    if clean.is_empty() { "computer".into() } else { clean }
}

/// Where paired servers' identities live: `~/.config/lucem/servers` (each file mode 0600).
fn servers_dir() -> PathBuf {
    std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from).or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config"))).unwrap_or_else(std::env::temp_dir).join("lucem").join("servers")
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
