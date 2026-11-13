---
title: Install
description: Build the app, the command line and the model plugins from source.
sidebar:
  order: 1
---

lucem is not published yet: you build it from its repository. You need Rust 1.95 or newer
([rustup](https://rustup.rs)) and, on Linux, the libraries the app's UI builds against.

```sh
git clone https://github.com/neuro-kitchen/lucem
cd lucem
cargo run -p lucem-app            # the app
cargo build -p lucem-cli          # the `lucem` command: target/debug/lucem
```

## The app on Linux

```sh
sudo apt-get install libfontconfig-dev libfreetype-dev libwayland-dev \
  libxkbcommon-dev libxkbcommon-x11-dev libx11-xcb-dev
```

## Your GPU

lucem runs on CUDA (NVIDIA), on Vulkan or Metal through wgpu (any recent GPU), or on the CPU when
there is no GPU. It picks the first that works. Ask which runtimes this machine can use:

```sh
lucem doctor
```

To choose one, set `LUCEM_RUNTIME` (`cuda`, `wgpu`, `cpu`) or pick it in **Settings → Compute**.
A runtime compiled in but without its driver (for example CUDA with no NVIDIA driver loaded) is
reported as unusable, and lucem falls back to the next.

## Model plugins

Models run as separate programs, so their runtimes and licences stay out of lucem. Build the ones
you want next to the app, where it finds them:

```sh
cargo build -p lucem-ml-stardist   # nuclei: StarDist H&E and fluorescence
cargo build -p lucem-ml-sam        # Segment Anything 2
```

Then install their weights from **Settings → Models**. See [Models](/lucem/models/).
