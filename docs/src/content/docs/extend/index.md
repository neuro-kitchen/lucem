---
title: Extend
description: Add a model plugin, a script step or an operation.
sidebar:
  order: 0
---

## A model plugin

A plugin is a program named `lucem-ml-<family>`. lucem asks it what it offers and sends it work
as JSON lines on its standard input; images and masks travel as raw files in a scratch folder.

| Command or request | Does |
|---|---|
| `<program> info` | Prints the plugin's manifest: its models, their task, weights, licence, runtime |
| `<program> serve` | Stays running and answers requests, one JSON line each |
| `install` | Downloads a model's weights (or the plugin's runtime) |
| `segment` | A mask from prompts (points, boxes) |
| `instances` | Every object of a region, as a label image |

The contract is the `lucem-ml` crate; `plugins/ml/lucem-ml-stardist` is a complete example.

## A script step

A Python or Rust file as a pipeline step: **Script → New Python script…** in the Pipeline. Edit
it in your editor; when you save it, it runs again.

## An operation

Operations live in `crates/lucem-ops` (the GPU kernels, in CubeCL) and are listed in
`engine/lucem-pipeline` (`Op`), where each gets its typed inputs and output. A new one appears
in the menus, the search and `lucem ops` from there.
