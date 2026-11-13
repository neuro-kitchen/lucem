---
title: "lucem_hub"
description: "Verified downloads of published artifacts (model weights)."
sidebar:
  label: "lucem-hub"
  order: 7
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

`lucem-hub`: verified downloads of published artifacts (model weights) and their local cache,
as dsp-kitchen's `dspk-hub`. All network access of lucem's models is here.

A file is fetched to a temporary name, hashed while it streams, and committed to the cache only
when its SHA-256 and size match; a file without a hash is refused. The cache is
`$LUCEM_MODELS`, else `$XDG_CACHE_HOME/lucem/models`, else `~/.cache/lucem/models`, one folder
per model family.

## Functions

### `cache_dir`

```rust
pub fn cache_dir(family: &str) -> PathBuf
```

The cache folder of a model family.

### `sha256_file`

```rust
pub fn sha256_file(path: &Path) -> Result<String>
```

The SHA-256 of a file (lowercase hex).

### `is_valid`

```rust
pub fn is_valid(path: &Path, sha256: &str, bytes: u64) -> bool
```

Whether `path` holds the expected file (size first, then hash).

### `read_verified`

```rust
pub fn read_verified(path: &Path, sha256: &str) -> Result<Vec<u8>, String>
```

Reads a file only if its hash matches (what a model loads).

### `fetch`

```rust
pub fn fetch(
    url: &str,
    dest: &Path,
    sha256: &str,
    bytes: u64,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<PathBuf, String>
```

Downloads `url` to `dest` unless a valid copy is there; `progress(done, total)` follows the
bytes. The file is committed only when its SHA-256 (and size, when given) match.

### `fetch_archive`

```rust
pub fn fetch_archive(
    url: &str,
    archive: &Path,
    sha256: &str,
    bytes: u64,
    dest_dir: &Path,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<PathBuf, String>
```

Downloads a `.tgz` (verified like `fetch`) and unpacks it into `dest_dir`, unless
`dest_dir` already holds what it unpacked (a `.complete` mark is written last).
