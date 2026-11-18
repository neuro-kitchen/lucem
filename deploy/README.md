# Deploy: lucem as a server and batch runner (Podman)

`lucem serve` gives enrolled clients tiles of the images under a data folder, over QUIC with TLS 1.3
on UDP port 50051. The same image runs `lucem run` (a pipeline over a whole image) and `lucem doctor`.

| File | What it is |
|---|---|
| `Containerfile` | The image: the `lucem` command line on Debian slim, run as a non-root user |
| `.containerignore` | Keeps build output, data and notes out of the build context |
| `lucem-serve.container` | A Podman Quadlet: the server as a systemd service |
| `smoke-test.sh` | Checks an image end to end |

## Build

One `Containerfile`, three targets. They hold the same `lucem` (CUDA, wgpu and CPU compiled in) and
differ only in the system libraries each GPU needs:

| Target | For | Run with |
|---|---|---|
| `cpu` (default) | no GPU | — |
| `nvidia` | NVIDIA: CUDA and Vulkan; the driver comes from the host | `--device nvidia.com/gpu=all` |
| `mesa` | AMD, Intel: Vulkan (wgpu) | `--device /dev/dri` |

From the repository root:

```bash
podman build -f deploy/Containerfile --ignorefile deploy/.containerignore --target nvidia -t lucem-serve:nvidia .
```

## Smoke test

```bash
deploy/smoke-test.sh                    # target from the host: nvidia, mesa or cpu
deploy/smoke-test.sh --target cpu --no-build
```

## Run

```bash
mkdir -p ~/lucem-data
podman run --rm -p 50051:50051/udp --device nvidia.com/gpu=all --security-opt=label=disable \
  --userns=keep-id:uid=10001,gid=10001 --read-only --cap-drop=all --security-opt=no-new-privileges \
  --pids-limit=1024 -v ~/lucem-data:/data:Z -v lucem-cache:/home/lucem/.cache lucem-serve:nvidia \
  serve --bind 0.0.0.0:50051 --name $(hostname) --max-clients 8
```

A pipeline over a whole image (`spec.json` is described in `engine/lucem-pipeline/src/spec.rs`; it is
written next to the output as a record). When the server is running on the same folder, mount it
with `:z` in both (shared SELinux label): `:Z` gives the folder to one container only, and each new
container would take it from the others (the server then cannot read its client registry).

```bash
podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable --userns=keep-id:uid=10001,gid=10001 \
  -v ~/lucem-data:/data:z lucem-serve:nvidia run /data/spec.json /data/slide.ome.zarr /data/mask.ome.zarr
```

## The page (browsers)

`lucem serve --web /usr/share/lucem/web` (the image's default command) serves the page on TCP
8443 and WebTransport on UDP 8443. The page signs in with a pairing code once (`--pair` prints
one at start; `lucem clients pair` makes more) and keeps a token; `lucem clients revoke` ends it.
It views the images under `/data`; the server computes for it (pipelines on tiles, histology,
whole-image runs).

### Routes

| Where the browser is | How | Notes |
|---|---|---|
| This computer | `-p 127.0.0.1:8443:8443/tcp -p 127.0.0.1:8443:8443/udp`, open `http://localhost:8443` | a browser counts localhost as secure: no certificate needed for the page |
| A laptop, the server a cluster node | `ssh -L 8443:localhost:8443 node`, open `http://localhost:8443` | SSH carries TCP only: the page falls back to its WebSocket (no WebTransport) |
| Your tailnet | `tailscale serve --https=443 http://127.0.0.1:8443` | a real certificate for the page; WebTransport needs UDP 8443 reachable too, else the WebSocket |
| A lab network | `--certificate fullchain.pem --private-key key.pem --web-bind 0.0.0.0:8443`, publish both protocols | the page and the socket on HTTPS; WebTransport keeps its own 14-day certificate (restart the server within 14 days: the page reads the new hash on its next load) |

Plain HTTP off loopback is refused unless asked for (`--web-plain-http`, as the image does, its
port published only to 127.0.0.1). Every response carries a Content-Security-Policy (the page's
own scripts only), `Cross-Origin-Opener-Policy` / `-Embedder-Policy` (threads), and
`X-Content-Type-Options: nosniff`; the WebSocket and WebTransport refuse a browser whose
`Origin` is not the page's own (another site open in the same browser cannot use the server).

## Who can connect

Only clients the server has enrolled. The server keeps its own small certificate authority in
`/data/.lucem-auth` (mode 0700; its key 0600).

```bash
(umask 077; podman exec lucem-serve lucem clients enroll alice --out - > alice.pem)
podman exec lucem-serve lucem clients list
podman exec lucem-serve lucem clients revoke alice      # refused from its next connection on
```

A computer can also pair with a one-time code instead of a file. Codes exist only when asked for:
`lucem clients pair` (add `--control` for more than reading) makes one and shows it on the
terminal itself, never on standard output or in a log, and the running server accepts it at once.
The code itself never crosses the network (SPAKE2): it works once, for 10 minutes and 3 tries.
`lucem clients codes` lists the live codes (without their secrets) and `--cancel ID|all` cancels
them, for a code that may have been seen.

```bash
podman exec -it lucem-serve lucem clients pair     # -it: the code is shown on the terminal
podman exec lucem-serve lucem clients codes --cancel all
```

Hand the file over privately: it holds the client's private key. Back up `/data/.lucem-auth`. Clients
open only images under `/data`: hidden folders, links out of it and anything that is not an image are
refused. Clients are read-only.

```bash
lucem fetch <server-ip>:50051 --identity alice.pem --server-name <name> --open slide.ome.zarr
```

## Hardening

The image runs as a non-root user (uid 10001). The service file adds a read-only root, no Linux
capabilities, no privilege gain and a process cap. Open the port only to the network that needs it, or
reach the server through a VPN.

## GPU

- **NVIDIA:** install the NVIDIA Container Toolkit and run once `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`.
- **AMD, Intel:** `--device /dev/dri` with the `mesa` target; rootless containers also need `--group-add keep-groups`.
- **SELinux:** device files are not labelled for containers, so GPU containers use `--security-opt=label=disable`.
