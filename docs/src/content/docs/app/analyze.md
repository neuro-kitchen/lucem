---
title: Analyze
description: Tables and plots of what was measured.
sidebar:
  order: 3
---

Analyze lists what was measured on the image shown: its **regions and shapes**, its **cells**
and **tiles** (histology), and each result view's table.

## The table

- Filter each column (`> 10`, `1..20`, `= stroma`), search every column, sort by a header.
- Click a row to select that shape or cell; it shows in Explore.
- **Table → Export CSV** writes the rows shown.
- **Table → Code on the table** runs a Python or Rust step on it.

## Views

The table and plots are views, as in Explore: tabs you drag into splits, arranged from **+** in
a tab. Plots draw the rows the table shows, filters included.

| Plot | Shows |
|---|---|
| Histogram | How one measurement is spread, coloured by a group (class) if chosen |
| Scatter | One measurement against another |
| Box plot | A measurement compared across groups |
| Counts | How many rows each group has |

A plot's tab switches its kind. The hierarchy (left) and the selection and summary pane (right)
show and hide from the ends of the bar.
