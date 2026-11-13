---
title: lucem serve
description: Serve images to enrolled clients.
sidebar:
  order: 4
---

```text
lucem serve [--data DIR] [--bind ADDR] [--name HOST]... [--max-clients N] [--allow-anonymous]
            [--pair] [--certificate FILE --private-key FILE]
lucem clients pair [--control] [--out FILE]            (--auth DIR)
lucem clients codes [--cancel ID|all]                  (--auth DIR)
lucem clients enroll <name> [--control] --out FILE|-   (--auth DIR)
lucem clients list | revoke <name>                     (--auth DIR)
lucem fetch <host:port> --identity FILE --server-name NAME [--open PATH]
```

`serve` gives enrolled clients tiles of the images under `--data`, over QUIC with TLS 1.3 (UDP
port 50051). Each client gets an identity file from `clients enroll`, or pairs itself with a
one-time code from `clients pair` (`lucem pair <host:port> <code>` on the computer); `revoke`
removes it.

Pairing codes are secrets and are made only when asked for: `clients pair` shows its code on the
terminal itself, never on standard output or in a log (a script passes `--out FILE`: a file only its owner can read), and
`serve --pair` shows one at start. A code works once, for 10 minutes and 3 tries, and never
crosses the network (SPAKE2). `clients codes` lists the live codes without their secrets;
`--cancel ID|all` cancels them.
See [Deploy](/lucem/deploy/).
