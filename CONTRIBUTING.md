# Contributing to lumen

Thanks for helping. Issues are welcome for bugs, results that differ from their reference, and
features (the forms ask for what we need to reproduce). For a security problem, see
[SECURITY.md](SECURITY.md) instead.

The code of conduct and support pages are shared across
[neuro-kitchen](https://github.com/neuro-kitchen/.github).

## Setup

```bash
git clone https://github.com/neuro-kitchen/lumen && cd lumen
uv sync                                                # Python development environment
cargo run -p lumen-app                                 # the desktop application
cargo build --workspace --profile validate             # every crate, for tests and benchmarks
```

On Linux the desktop application needs the development packages of fontconfig, freetype, Wayland,
xkbcommon and X11 XCB. CUDA is loaded at run time, so no toolkit is needed to build. The CPU
runtime downloads a prebuilt LLVM on its first build.

## One build

The workspace is heavy (CubeCL, wgpu, GPUI, Polars), and other feature sets or profiles recompile
it from scratch, so:

- write all the changes first and build once;
- use `cargo run -p lumen-app` for the application, and `--profile validate` for tests and
  benchmarks (dependencies optimized, our crates with debug assertions);
- build and test as separate steps: build with `--no-run` first, then run the test binaries;
- prefer `--workspace` to a list of packages: a package subset resolves features differently and
  can rebuild everything;
- pick runtimes at run time, not at build time: `LUMEN_TEST_RUNTIMES=cuda` (or `wgpu`, `cpu`).

## Tests

- Test image operations on every runtime compiled in, against a host reference (scipy.ndimage,
  scikit-image or a direct host computation). CI has no GPU and runs the CPU runtime only, so run
  the GPU runtimes locally when you change a kernel.
- Use small synthetic images. Never commit research data, credentials or API keys.

## Code

- Match the surrounding code: naming, comment density, module layout (one concept per module).
- Raw image data is never modified: results are new layers, masks, tables or annotations.
- Every tunable value is a setting with a visible default, not a constant buried in a function.
- Line width is 120 (`rustfmt.toml`, `pyproject.toml`).
- Commit messages: `type(scope): summary` (`feat`, `fix`, `perf`, `docs`, `test`, `ci`, `build`,
  `refactor`, `chore`), with what changed and how it was checked in the body.

## Licences

lumen is MIT OR Apache-2.0. Code written from another project's source keeps its licence and
notice: say where it comes from (file, version) in the module docs. Code written from a paper says
so and cites it.

By contributing you agree that your contribution is licensed as the code it changes.
