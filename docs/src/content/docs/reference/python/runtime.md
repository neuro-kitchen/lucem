---
title: lucem.runtime
description: "Pick where CubeCL kernels run (cuda, wgpu, cpu) and autotune this device."
sidebar:
  order: 1
---

```python
import lucem

lucem.runtime.available()   # e.g. ['cuda', 'wgpu', 'cpu']
lucem.runtime.current()     # active default runtime
lucem.runtime.set("cuda")   # override default for subsequent calls
```

| Function | Signature | Description |
|---|---|---|
| `available` | `available() -> list[str]` | Runtimes compiled into this wheel, GPUs first. |
| `current` | `current() -> str` | Default runtime used when `runtime=` is omitted (respects `LUCEM_RUNTIME`). |
| `set` | `set(name: str) -> None` | Set the default runtime (`"cuda"`, `"wgpu"`, or `"cpu"`). |
| `autotune` | `autotune(runtime=None, *, sides=None, operators=True, models=True) -> dict` | Tune CubeCL kernels and installed ML plugins once and persist to the disk cache. |
