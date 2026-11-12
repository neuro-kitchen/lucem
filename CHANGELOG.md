# Changelog

All notable changes, newest first. Versions follow [Semantic Versioning](https://semver.org); before
1.0 any release may change the API.

## Unreleased

No release has been made yet. Work on the `version/v0.0.1` branch, newest first.

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
