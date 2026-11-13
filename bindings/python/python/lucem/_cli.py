"""The ``lucem`` command of the Python package: the desktop app, tuning this device, the model plugins.

The full command line (operators, pipelines, histology batches, the server, MCP) is the Rust
program ``lucem``: ``cargo install --git https://github.com/neuro-kitchen/lucem lucem-cli``.
"""

from __future__ import annotations

import argparse
import sys

import lucem

_COMMANDS = ("app", "autotune", "plugins", "version")


def _plugins() -> None:
    found = lucem.ml.plugins()
    if not found:
        print('no model plugins found: pip install "lucem[ml]" (or "lucem[stardist]", "lucem[sam]")')
    for p in found:
        print(f"{p['name']} {p['version']} ({p['licence']}): {p['program']}")
        for m in p["models"]:
            state = "installed" if m.get("installed") else "downloads on first use"
            print(f"  {m['id']}: {m['name']} [{state}]")
    for problem in lucem.ml.problems():
        print(f"  ! {problem}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    words = sys.argv[1:] if argv is None else argv
    if words and not words[0].startswith("-") and words[0] not in _COMMANDS:
        sys.exit(f"`lucem {words[0]}` is in the full command line: cargo install --git https://github.com/neuro-kitchen/lucem lucem-cli")
    parser = argparse.ArgumentParser(prog="lucem", description="lucem from Python: the desktop app and the model plugins")
    commands = parser.add_subparsers(dest="command", required=True)
    app = commands.add_parser("app", help="open the desktop app")
    app.add_argument("path", nargs="?", help="an image or project to open")
    tune = commands.add_parser("autotune", help="tune this device once: lucem's operators and every model plugin")
    tune.add_argument("--runtime", help="wgpu, cuda or cpu (default: every usable runtime)")
    commands.add_parser("plugins", help="list the model plugins found")
    commands.add_parser("version", help="print lucem's version")
    args = parser.parse_args(words)
    if args.command == "app":
        try:
            lucem.app.open(args.path)
        except FileNotFoundError as e:
            sys.exit(str(e))
    elif args.command == "autotune":
        for runtime, r in lucem.runtime.autotune(args.runtime).items():
            if "error" in r:
                print(f"skip {runtime}: {r['error']}")
                continue
            print(f"ok {runtime}: {r.get('operator_runs', 0)} operator runs in {r.get('seconds', 0):.0f} s")
            for f in r.get("operator_failures", []):
                print(f"  failed {f['operator']} at {f['side']}: {f['why']}")
            for name, outcome in r.get("plugins", {}).items():
                print(f"  {name}: {outcome}")
    elif args.command == "plugins":
        _plugins()
    else:
        print(lucem.__version__)


if __name__ == "__main__":
    main()
