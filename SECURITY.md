# Security policy

## Reporting a vulnerability

Please report it privately, not in a public issue: open a
[security advisory](https://github.com/neuro-kitchen/lucem/security/advisories/new) on this
repository. Include what is affected (crate, desktop application, Python package), the version,
and how to reproduce it.

You will get an answer within a week. Fixes are released as a new version and described in the
advisory and the [CHANGELOG](CHANGELOG.md).

## Supported versions

lucem is in early development: only the latest release receives fixes.

## Scope

Of particular interest: file parsers in `lucem-io` and the tile reader reading untrusted images
(OME-Zarr metadata and chunks, TIFF), and project files opened by the desktop application.

## Research data

lucem may process images from human tissue. If a problem could expose such data, say so clearly in
your report so it can be prioritized.
