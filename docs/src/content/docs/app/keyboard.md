---
title: Keyboard and mouse
description: Every shortcut.
sidebar:
  order: 4
---

| Keys | Does |
|---|---|
| <kbd>Ctrl</kbd>+<kbd>K</kbd> | Search every command |
| <kbd>Ctrl</kbd>+<kbd>O</kbd> | Open an image or OME-Zarr |
| <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>O</kbd> | Open a project |
| <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>S</kbd> | Save the project in a folder |
| <kbd>Ctrl</kbd>+<kbd>N</kbd> | New project |
| <kbd>Ctrl</kbd>+<kbd>Z</kbd>, <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Z</kbd> | Undo, redo |
| <kbd>Ctrl</kbd>+<kbd>1</kbd> / <kbd>2</kbd> / <kbd>3</kbd> | Explore, Pipeline, Analyze |
| <kbd>Ctrl</kbd>+<kbd>[</kbd>, <kbd>Ctrl</kbd>+<kbd>]</kbd> | Left panel, right panel |
| <kbd>Ctrl</kbd>+<kbd>R</kbd> | Rulers |
| <kbd>Ctrl</kbd>+<kbd>,</kbd> | Settings |
| <kbd>Del</kbd> | Delete the selected shapes |
| <kbd>Space</kbd> (held) | Move the image with any tool |
| <kbd>F</kbd> | Fit the image (or the graph, in Pipeline) |
| <kbd>[</kbd>, <kbd>]</kbd> | Brush size |
| <kbd>Ctrl</kbd>+<kbd>=</kbd>, <kbd>Ctrl</kbd>+<kbd>−</kbd> | Brush size (Brush, Eraser, Wand) |
| <kbd>+</kbd>, <kbd>−</kbd> | Zoom in, out |
| <kbd>Ctrl</kbd>+<kbd>Q</kbd> | Quit |

Tool keys are listed in [Explore](/lucem/app/explore/#tools). On macOS, <kbd>Cmd</kbd> stands for
<kbd>Ctrl</kbd>.

## Your own keys and settings

Two files in the configuration folder (`~/.config/lucem/` on Linux, `%APPDATA%\lucem\` on
Windows) hold what you can change by hand. Both are written at the first start with every
default, so you see what there is; delete a line to go back to its default.

**`keybindings.toml`**: a key, or a list of keys, per command; `""` removes a binding. Read when the
app starts.

```toml
[commands]
"file.open_image" = "secondary-o"   # secondary: Ctrl, or Cmd on macOS
"view.left" = ["secondary-[", "f2"]

[canvas]
toggle_sidebar = "tab"   # hide or show both side panels, on the image
```

Keys are written as `secondary-shift-o`, `ctrl-shift-o`, `alt-g`, `space`, `left`, `f3`, `+`. A key that cannot be read,
or one key on two commands, is named in the status bar and the default is kept.

**`settings.toml`**: the look, the compute runtime and the Segment tool's model. The Settings
screen writes the same file and keeps your comments; an edit made outside applies while the app
runs (the runtime at the next start).

```toml
[appearance]
theme = "dark"          # "system" | "dark" | "light"
accent = "app"          # "app" or a colour as "#rrggbb"
font_size = 15.0
pasteboard = "grey"     # around the image: "grey" | "dark" | "light"

[compute]
runtime = "automatic"   # "automatic" | "wgpu" | "cuda" | "cpu"
tile_threads = 0        # 0 is the machine's cores, at most 8

[models]
segment = ""            # a plugin's model id; "" is the plugin's default
```
