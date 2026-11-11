---
title: lumen serve
description: Serve images to enrolled clients.
sidebar:
  order: 4
---

```text
lumen serve [--data DIR] [--bind ADDR] [--name HOST]... [--max-clients N] [--allow-anonymous]
            [--certificate FILE --private-key FILE]
lumen clients enroll <name> [--control] --out FILE|-   (--auth DIR)
lumen clients list | revoke <name>                     (--auth DIR)
lumen fetch <host:port> --identity FILE --server-name NAME [--open PATH]
```

`serve` gives enrolled clients tiles of the images under `--data`, over QUIC with TLS 1.3 (UDP
port 50051). Each client gets an identity file from `clients enroll`; `revoke` removes it.
See [Deploy](/lumen/deploy/).
