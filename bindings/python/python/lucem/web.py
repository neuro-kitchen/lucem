"""The browser web app, served from Python (``pip install "lucem[web]"`` / ``uv add "lucem[web]"``)."""

from __future__ import annotations

import functools
import http.server
import importlib.util
import os
import shutil
import subprocess
import webbrowser
from pathlib import Path

__all__ = ["dist_dir", "serve"]


def dist_dir() -> Path | None:
    """Path to the built WASM web bundle, or ``None`` when ``lucem[web]`` is not installed."""
    if importlib.util.find_spec("lucem_web") is not None:
        import lucem_web  # type: ignore[import-not-found]

        d = lucem_web.dist_dir()
        if d is not None and (d / "index.html").is_file():
            return d
    # Development checkout fallback
    repo_dist = Path(__file__).resolve().parents[4] / "app" / "lucem-app-web" / "dist"
    if (repo_dist / "index.html").is_file():
        return repo_dist
    return None


class _IsolatedHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the standalone WASM app with COOP/COEP headers required for SharedArrayBuffer."""

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


def serve(
    data: str | os.PathLike | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    pair: bool = True,
    memory: str = "2GiB",
    open_browser: bool = False,
) -> None:
    """Serve the Lucem web app locally.

    When the native ``lucem`` CLI is available and ``data`` is given, delegates to
    ``lucem serve --data <data> --web <dist> --web-bind <host>:<port>`` so tiles and pipelines run
    on the host GPU. Otherwise serves the standalone browser viewer with COOP/COEP headers.
    """
    web_root = dist_dir()
    if web_root is None:
        raise FileNotFoundError('the lucem web bundle is not installed: pip install "lucem[web]"')

    cli = shutil.which("lucem-cli") or shutil.which("lucem")
    # Avoid recursing into the Python `lucem` console-script entrypoint itself
    if cli is not None:
        try:
            head = Path(cli).read_bytes()[:4]
            if head.startswith(b"#!"):
                cli = None
        except OSError:
            cli = None

    if cli is not None and data is not None:
        cmd = [
            cli,
            "serve",
            "--data",
            os.fspath(data),
            "--web",
            str(web_root),
            "--web-bind",
            f"{host}:{port}",
            "--memory",
            memory,
        ]
        if pair:
            cmd.append("--pair")
        subprocess.run(cmd, check=True)
        return

    url = f"http://{host}:{port}/"
    print(f"Serving standalone Lucem web viewer from {web_root} at {url}")
    if open_browser:
        webbrowser.open(url)
    handler = functools.partial(_IsolatedHandler, directory=str(web_root))
    with http.server.ThreadingHTTPServer((host, port), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
