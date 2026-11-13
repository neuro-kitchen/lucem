# Changelog

All notable changes, newest first. Versions follow [Semantic Versioning](https://semver.org); before
1.0 any release may change the API.

## Unreleased

## [0.0.2] - 2026-11-13

The Python packages (`lucem`, with histology, the desktop app and the model plugins as extras;
built from the repository for now, on PyPI from a later release); InstanSeg's
nucleus models on Burn, and both model plugins 3–10× faster on a GPU; `lucem autotune` for the
whole machine; pairing codes kept out of logs; releases from a tag of `main` by trusted
publishing to crates.io and PyPI; brush size from the keyboard.

### Release process

- Releases come from a `v*` tag of `main` (`.github/workflows/release.yml`, as kitchen and
  dsp-kitchen): the tag is checked against the workspace version and CHANGELOG.md, the crates are
  packaged and checked by cargo-deny, the library crates go to crates.io by trusted publishing
  (environment `crates-io`, approved by a maintainer; crates already at that version are skipped),
  then the GitHub release takes this file's section as its notes. The app, the CLI and the MCP
  server stay off crates.io until kitchen-agent is published.
- The Python packages go to PyPI from the same tag once the repository variable `PUBLISH_PYPI`
  is `true` (until then a tag publishes the crates and the GitHub release only), by trusted
  publishing (approved by a maintainer): `lucem` and `lucem-hist` (abi3 wheels for CPython 3.12+ on Linux x86_64, glibc
  2.34+, Windows x64 and macOS arm64, and `lucem`'s sdist) in the environment `pypi`; the app and
  the model plugins (`lucem-app`, `lucem-ml-stardist`, `lucem-ml-sam`) each in its own
  (`pypi-app`, `pypi-ml-stardist`, `pypi-ml-sam`), so a failure there holds up nothing else.
  The extras' pins are checked against the workspace version. The "Private :: Do Not Upload"
  classifier that blocked uploads is gone.

### Added

- **Python packages, as dsp-kitchen's** (built from the repository, `pip install ./bindings/python`,
  until they are on PyPI): `lucem` gives `lucem.runtime`, `lucem.io`
  (images of any size read by block, plane or region; OME-Zarr written block by block),
  `lucem.ops` (every operator of the catalog on NumPy arrays, and pipelines that stay on the
  device), `lucem.custom` (your own functions over an image of any size, with halos and objects
  joined across blocks) and `lucem.ml` (the model plugins: found, asked for their models, run on
  an array). Extras: `lucem[hist]` adds `lucem.hist` (stains and deconvolution, tissue, cells
  found or grown from a model's nuclei, the app's steps built in Python or read from a
  `session.json` and run through the app's engine, tables and GeoJSON); `lucem[app]` the desktop
  app (`lucem app`); `lucem[stardist]`, `lucem[sam]`, `lucem[instanseg]` and `lucem[ml]` the model plugins
  as programs, kept apart from lucem as before (their weights download on first use); `lucem[all]`
  everything. The `lucem-hist` wheel is the whole extension built with histology, and `lucem`
  loads it in place of its own when it is there: one device per runtime. The binding crates
  (`lucem-py-core`, `lucem-py-hist`) register into the one extension, `lucem-py`.
- **InstanSeg on Burn:** the `lucem-ml-instanseg` plugin runs InstanSeg's published brightfield
  and single-channel nucleus models (Apache-2.0) on CUDA, wgpu or the CPU, as StarDist does
  (`models/instanseg`). Each bioimage.io package is fetched and checked (SHA-256). Its
  TorchScript archive is read in Rust, with no PyTorch: a small pickle reader and the zip, then
  kept as safetensors with the model's thresholds. The postprocessing is ported from the
  archive's own code: seeds, the pixel classifier, fragments and holes, overlaps joined. The
  published test images give the published nuclei. `pip install "lucem[instanseg]"` (also in
  `lucem[ml]`). The channel-invariant "nuclei and cells" model is not ported yet.
- **`lucem autotune`, one tuning for the whole machine** (as dsp-kitchen's): lucem's operators
  over a grid of image sizes, then every model plugin found (`<plugin> tune`: each installed model
  on every tile shape), on every usable runtime; the choices and compiled kernels are kept in the
  user cache folder. Also **File › Tune this device** in the app and `lucem.runtime.autotune()`.
  The model plugins have a `bench` command (timings per phase, cold and warm).
- **Brush size from the keyboard:** <kbd>Ctrl</kbd>+<kbd>+</kbd> and <kbd>Ctrl</kbd>+<kbd>−</kbd>
  make the brush bigger or smaller while the Brush, Eraser or Wand is in hand (they zoom
  otherwise, as <kbd>+</kbd> and <kbd>−</kbd>), next to <kbd>[</kbd> and <kbd>]</kbd>. The
  toolbar's stepper and both keys share one step.

### Security

- Pairing codes exist only when asked for and are never written where a log could keep them:
  `lucem serve` no longer makes one at every start (`--pair` asks for one, shown on the
  terminal); `lucem clients pair` shows its code on the terminal itself (`/dev/tty`, `CONOUT$`),
  not on standard output, so a redirect, a pipe or a service's log never holds it (a script
  takes it from `--out FILE`, readable by its owner only). `lucem clients codes` lists the live codes without their secrets and
  `--cancel ID|all` cancels them. Code scanning had flagged codes in cleartext logs.

### Changed

- **The models are Burn modules**: StarDist and InstanSeg are `#[derive(Module)]` structs of
  `burn::nn` layers (InstanSeg's mirroring its PyTorch classes), loaded by name with Burn's
  `SafetensorsStore` (InstanSeg's through its PyTorch adapter), instead of hand-written tensor
  functions and readers. The published outputs are still met: InstanSeg exactly, StarDist to 1e-6
  in single precision.
- **The models are 3–10× faster on a GPU** (RTX 2070, CUDA, warm):

  | Image | InstanSeg before → now | StarDist before → now |
  |---|---|---|
  | 512² | 1,179 → 111 ms | 225 → 39 ms |
  | 1024² | 4,136 → 503 ms | 541 → 200 ms |
  | 2048² | 21,666 → 2,790 ms | 2,226 → 696 ms |
  | 4096² | — → 11,106 ms | 10,914 → 3,336 ms |

  Half precision on the GPU, for the weights, the activations and the data to and from the
  device (half the bytes moved); BatchNorm folded into InstanSeg's convolutions at load; the 3 × 3
  layers of 64 channels and more as tensor-core matrix products, because Burn's own tensor-core
  convolution asks more shared memory than a Turing GPU has for them and falls back to its direct
  kernel (about 15× slower); when a matrix product cannot launch on a device, the direct
  convolutions take over for the rest of the run. InstanSeg's postprocessing works in logits a
  window row at a time (no exponential per pixel), compares only windows that touch, and fills
  holes near each nucleus only (512²: 499 → 14 ms). InstanSeg in half precision still gives the
  published nuclei exactly; StarDist's maps move by at most 1.8e-3 (probability) and 0.1 px
  (distances), same nuclei. `LUCEM_INSTANSEG_PRECISION` / `LUCEM_STARDIST_PRECISION=f32` and
  `…_CONV=direct` choose the single-precision, direct path (bit-faithful, slower).
- Channels open as the OME-Zarr file marks them (`omero` `active`; shown when it does not say):
  a 59-marker PhenoCycler image opens with its chosen few, the rest in the list.
- The workspace's data list (`.data/`) holds only images and OME-Zarr containers: licences,
  notes, archives and raw formats lucem does not open (`.qptiff`, `.zip`) are left out.

### Fixed

- CI runs green again: the app's device uses the CPU there (`LUCEM_RUNTIME=cpu`; it opened
  wgpu, whose device thread panics on a runner without a GPU, failing every app test after it),
  and cargo-deny allows kitchen's repository as a git source (kitchen-agent is not on crates.io
  yet). A test of the activity view found another test's Rulers entry in the shared log; the
  Python test compares the version to the installed package's instead of a fixed `0.1.0`.

### Documentation

- The README says what lucem is now (it described an empty skeleton), with the app's screenshot
  (dark or light as the reader's theme), installing the library crates from crates.io, and every
  crate; the docs site shows the screenshot on its front page and the App page. The sample image is
  credited (© PerkinElmer / Akoya Biosciences, CC BY 4.0).

## [0.0.1] - 2026-11-12

The first release: lucem's library crates on crates.io (`lucem-core`, `lucem-io`, `lucem-ops`,
`lucem-pipeline`, `lucem-hist`, `lucem-ml`, `lucem-hub`, `lucem-view` and the `lucem` umbrella).
The app, the command line and the MCP server follow once kitchen-agent is on crates.io. The
work of the `version/v0.0.1` branch, newest first:

### lumen is now lucem (2026-11-12)

- **The name.** lumen was taken on the registries; it is now **lucem** (Latin for light): the
  crates (`lucem-core`, `lucem-io`, `lucem-ops`, `lucem-pipeline`, `lucem-hist`, `lucem-ml`,
  `lucem-hub`, `lucem-view`, `lucem-stream`, the `lucem` umbrella), the programs (`lucem`,
  `lucem-app`, the `lucem-ml-*` plugins), the Python package, the app's title, the
  `LUCEM_*` variables, `project.lucem.json`, `<image>.lucem.zarr`, the MCP server (`lucem_app`).
- **What was kept under the old name keeps working:** `LUMEN_*` variables are read when the
  `LUCEM_*` one is not set; the `lumen` folders under the config, data and cache folders move to
  `lucem` once, on first start; `project.lumen.json` opens (and is saved as `project.lucem.json`);
  sessions and projects with `lumen_session` / `lumen_project` read; `# lumen:` script headers
  read; plugin programs installed as `lumen-ml-*` are found. The serving protocol is
  `lucem-stream/1` (a server and its clients update together; nothing was released before).
- kitchen comes from its released v0.0.4 tag (it was the `version/v0.0.4` branch).
- Entries below this one name lumen as it was then.

### Tools an agent can use (2026-11-12)

- **Steps answer with what they found.** A histology step run by an agent (tissue, cells,
  superpixels, density, classifiers) returns its job, and the agent gets the step's outcome when
  the job ends (the summary, each step's numbers, the ids of the shapes it made) instead of `null`
  at once. A step that cannot start says why (no image, the GPU still starting, no region
  selected). This was why "Find tissue" seemed to find nothing: the device was still starting.
- **`view.snapshot`:** the shown image as a picture (its channels as displayed, the shapes outlined
  in their class colours, at most 1024 px from a level of the pyramid that fits: one read, nothing
  at full resolution), with every shape's id, class and box in image pixels and the scale.
- **Shapes by id:** `shapes.list`, `shapes.draw` (rectangle, ellipse, polygon, point in image
  pixels), `shapes.set_class` ("call that region tissue"), `shapes.delete`, `classes.add`; each
  returns the ids it touched.
- A new script node (Automate › New Python / Rust script node) opens in the Pipeline's Inspector
  editor, selected, instead of the system's text editor (still a button there).

### Agents, the code editor and environments (2026-11-12)

- **Agents workspace:** the chat. It runs the agent you already have (Antigravity `agy acp`, or
  Claude Code through its ACP adapter) with your own login: no API key in lumen. The agent works
  through lumen's own commands (MCP, through the app's program as a bridge to its socket):
  anything that changes the project asks first (Allow / Deny), and every call shows in the
  conversation and on the Activity timeline. Agent menu: start, stop, a new session, which agent.
  Settings › Agents: what an agent may run without asking. Sessions are saved.
  The agent reaches lumen as `lumen_app` (not `lumen`, the headless server it may already have),
  is told before each session's first prompt that it works inside the open app through those
  tools and what is open, and sees each command under a tool name it accepts (`hist_find_cells`).
- **The code editor in the app:** a script node's code opens in the Pipeline's Inspector
  (highlighted, line numbers); Save and run writes the file and runs the node again; the errors of
  its last run are marked on their lines (Python tracebacks, Rust compiler messages); Revert puts
  back the file. Analyze's code on the table uses the same editor. The external editor stays as a
  button.
- **Environments** (Settings › Environment, was Tools): scripts run in the project's Python
  environment (`scripts/.venv`), else the one active, else lumen's default; uv makes and syncs it
  from `scripts/pyproject.toml` and `uv.lock` (NumPy in a new one). Rust scripts build with the
  compiler the project pins (`scripts/rust-toolchain.toml`, "Pin the compiler found"). A Python set
  under Programs is still used first.

### The shared frame (2026-11-12)

- **Activity** (Analyze › View › Activity timeline, or the search): an episodic memory of the
  work (kitchen-ui's timeline over kitchen-shell's activity log): what you did and what an agent
  did, the time in each workspace and every job, summarised into a few episodes (or every
  action); hover a bar for its card, click it for each action, where it came from (workspace,
  menu, key, search, agent) and what it produced.
- **kitchen-shell's frame.** The title bar (the Lumen menu, Help, the workspaces, the project,
  where lumen runs, the panel toggles, the theme), the workspace's menu row, the status bar
  (status line, every job with its cancel button, lumen's readings, CPU and memory), Settings and
  the search are kitchen-shell's, shared with dsp-kitchen. The window keeps its rounded frame and
  resizes from its edge (now the frame's).
- **Every action is a command** of one registry (`commands.rs`, and the Pipeline's and Analyze's
  own): id, title, icon, what it does, group, key, when it can run and why not, and the JSON
  Schema of its arguments. The menus, the keys, the search (Ctrl+K; F3 for the operations) and,
  next, an agent's tools come from that one list, so none drifts from the others. A menu entry
  that cannot run is greyed with the reason.
- Menus that follow the project (channels, images, recent files, the nucleus models that fit,
  the Pipeline's sources and model steps, Analyze's tables) are made when they open.
- An operation is one command (`process.op.*`): a step on the shown channel in Explore, a node on
  the canvas in the Pipeline.
- Settings: the frame's Appearance (theme, accent, text size) first, then lumen's Display (a
  theme that follows the system, the pasteboard), Views, Compute, Models, Tools, Performance and
  About. Keyboard & mouse (F1) is a sheet over the frame.
- Jobs are kitchen-shell's registry (lumen's own, moved there).
- `lumen mcp`: `run_pipeline` takes the pipeline as its real type, so an agent sees its schema,
  and its description carries a working example (an agent needed eleven tries before, one
  now); `run_session` says where results go (inside an OME-Zarr image, else `out_dir`).
- Tests: every command runs (or says why not), every menu entry is a registered command,
  Settings has lumen's sections, the frame and the store share the workspace.

### Agents, servers and the command line (2026-11-11)

- **`lumen mcp`: lumen for agents.** A Model Context Protocol server over stdio (`app/lumen-mcp`,
  the official Rust SDK): an agent (Claude Code, `agy`, any MCP client) lists the operations,
  describes images, runs histology steps on an image (each recorded in a session), saves and
  loads `session.json`, runs a session over many images, and runs pipelines. Register it with
  `claude mcp add lumen -- lumen mcp`. `lumen ops`, `lumen run` and `lumen session run` share
  its code.
- **`lumen ops [--json]`** lists every pipeline operation: family, ports, parameters and
  defaults, from the operation catalog and its schema.
- **Pairing with a code.** `lumen serve` prints a one-time code (`42-8317-2956`: 10 minutes,
  3 tries); `lumen clients pair` makes another; `lumen pair <host:port> <code>` pairs this
  computer. Both sides prove they know the code with SPAKE2, so it never crosses the network;
  the computer makes its own key and sends only the public key; the server records it like any
  enrolled client (revocable). A computer that has not paired can do nothing but pair.
- **Provenance names images privately.** Exported results record the image relative to the
  results (`../slides/case.ome.zarr`) when it is close by, else its file name and a SHA-256
  fingerprint: never the full path.

### Shared UI (2026-11-11)

- **kitchen-ui 0.0.2** (from crates.io, released 2026-11-12): CodeView, DiffView, TextView and the
  chat pane are available to the app; nothing in lumen changed for it.
- **kitchen-ui.** The app's tokens, widgets and parameter editor now come from
  [`kitchen-ui`](https://crates.io/crates/kitchen-ui) 0.0.1, shared with dsp-kitchen: plain-grey
  neutrals, the 8 px spacing grid, radii by tier, and text, outlines and slider fills in a
  contrast-checked accent. lumen keeps its own bar sizes, pasteboard and number formats.
- **Accent: blue, or any colour.** lumen's accent is blue. Settings has kitchen-ui's accent
  picker (suggested colours or a custom one, kept readable in both themes); the preference is
  saved as `"#rrggbb"`, and an accent saved by an earlier version still reads.
- **Panes as cards.** Explore, Analyze and Pipeline draw their regions (tools, views, side
  panels, the rail) as rounded cards a few pixels apart on the chrome, as kitchen-ui does. The
  window has no outline and no transparent shadow margin (what was behind the window showed
  through it); it resizes from its own edge.
- **Status bar.** Where lumen runs, at its left (as VS Code's remote indicator: this computer
  for now; connecting to a lumen server comes with pairing), and this computer's CPU and memory
  use at its right, every 2 s.
- **Paths stay private.** Messages, file details, the start screen and Settings show only the
  last two parts of a path (`…/slides/case-12.ome.zarr`), never the home folder.
- **Settings › Tools.** The programs scripts use (Python, uv, Ruff, Cargo, rustc): where each is
  found (a path set here, its `LUMEN_*` variable, or the `PATH`) and its version. With uv set,
  Python scripts run through it (NumPy provided). lumen installs none of them.
- **Parameter editor:** switches and two-number lists of an operation are now editable too.

### Documentation (2026-11-11)

- **The documentation site** (`docs/`, Starlight, built with Bun): Learn, App, Histology,
  Reference, Models, Extend, Deploy and the changelog, with lumen's own look (hematoxylin ink,
  eosin accent, a chef's hat logo). The Operations page is generated from `lumen ops --json`; the
  Rust API pages from rustdoc's JSON (`docs/scripts/rust_api.py`): every public function and type
  of lumen-core, io, ops, pipeline, hist, ml and hub, with its signature and documentation.
- **CI:** `.github/workflows/docs.yml` builds the site with Bun on pull requests, and deploys it
  to GitHub Pages from `main` once the repository is public.

### Supply chain (2026-11-11)

- CI's Rust job frees the runner's disk (unused preinstalled toolchains) and builds without debug
  info, so the workspace's test build no longer runs out of space.
- RUSTSEC-2026-0194 and -0195 (quick-xml, reached only through polars' `object_store`) are
  recorded as accepted advisories in `deny.toml` until polars moves to `object_store` 0.14.

### Architecture (2026-11-11)

- **One engine for histology steps.** `lumen-hist::engine::apply(ctx, analysis, action)` is the
  only way a step runs. The app's buttons, `lumen session run` and (later) an agent call it with a
  serializable `Action` and get a structured `Outcome` back. Pixel classes and density maps now
  run headless.
- **Jobs.** One registry (`data/jobs.rs`) for everything running in the background (histology,
  export, the Model node): progress, cancel, one GPU job at a time. The status bar shows every
  job.
- **Store split** into modules: the document and its undo history (`data/document.rs`), the
  selection (`data/selection.rs`), the saved views (`data/layout.rs`).
- **Project format 2.** The processing graph is written to `pipeline.json` beside
  `project.lumen.json`; format-1 projects open and are migrated on save. Each workspace's views
  are saved in `layout.json` and restored when the project opens.
- **Headless UI tests** of the real app with a built-in sample image: every workspace, every
  search command, Settings from the search, views added, arranged and closed, a project saved
  and reopened.
- **Dependencies:** gpui-kit 0.9.1, cubecl 0.11.4, rustls-pki-types 1.16.0; analysis outputs and
  model weights are ignored by git.

### App

- One view container shared by Explore and Analyze: the dock, "Add view or container", the
  arrangements (tabs, side by side, stacked, grid) and the tab buttons.
- Analyze: the table and plots (histogram, scatter, box plot, counts) are views in that
  container; a plot's tab switches its kind; the hierarchy and side pane can be hidden; the
  table scrolls both ways and stays fast (cached row order, CSV made on export, sampled plots).
- Title bar laid out as Affinity's: the Lumen (file) menu and Help, the workspaces, and a menu
  row under it that changes with the workspace. The Pipeline's operations sit under one Process
  menu, a submenu per family.
- Search (Ctrl+K): icon rows with hints and group headings; a command runs after the search
  closes, so Settings and dialogs open from it.
- Channel editor in tabs (Colour, Levels); a channel's processing is always shown.
- Region select (M) beside Select (V); Add, Subtract, Intersect and Xor for selected shapes.
- Histology saved with the project; every step recorded in `session.json` and listed in
  History; Export writes the output store.
- Pipeline Model node (a model plugin as a step, run from the Inspector).

### Histology and models

- StarDist (H&E and fluorescence) on Burn as a plugin; matches the published outputs.
- Output store: cells and tiles as Parquet, shapes as GeoParquet, OME label images and the
  provenance, inside the image's OME-Zarr or beside it.
- `lumen session run <session.json> <image>…` replays an app session on many images.
- Nucleus models, split touching nuclei at dark gaps, pixel classifier, superpixels, density
  maps.

### Earlier on this branch

- Histology workflows (stains, tissue, cells, classifiers, densities, export), the SAM plugin,
  the studio redesign.
- Plane, 2D and 3D views with split panes; a Layers panel; the plane position on the canvas.
- Serving image tiles to enrolled clients over QUIC and TLS; the container image and Quadlet
  service.
- The `lumen` command (info, doctor, run, sample, serve, clients, fetch).
- 2D and 3D image reading, block loading and an OME-Zarr writer; a volume renderer on the
  device; user functions over whole images and headless pipeline specs.
