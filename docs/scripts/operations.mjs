// Writes the Operations page from `lumen ops --json` (the operations' own documentation, ports
// and defaults), so the page cannot drift from the code. Needs the built command line:
//   cargo build -p lumen-cli && bun docs/scripts/operations.mjs
// (LUMEN_BIN overrides the program's path). The page is committed; rerun when operations change.
import { execFileSync } from 'node:child_process';
import { writeFile } from 'node:fs/promises';

const bin = process.env.LUMEN_BIN ?? new URL('../../target/debug/lumen', import.meta.url).pathname;
const ops = JSON.parse(execFileSync(bin, ['ops', '--json'], { encoding: 'utf8' }));
const families = [...new Set(ops.map((o) => o.family))];
const port = (t) => `\`${t}\``;
const value = (v) => (typeof v === 'number' && !Number.isInteger(v) ? String(v) : JSON.stringify(v));

let md = `---
title: Operations
description: Every pipeline operation, with its inputs, output and parameters.
sidebar:
  order: 1
---

{/* Written by docs/scripts/operations.mjs from \`lumen ops --json\`: don't edit by hand. */}

Every operation the Pipeline, the **Process** menu, the search and \`lumen run\` can use, by
family. Inputs and outputs are typed (${['image', 'mask', 'labels', 'table'].map(port).join(', ')}); a
connection between different types is refused. In a pipeline file an operation is written as
\`{"op": "<name>", …parameters}\`.

`;
for (const f of families) {
  md += `## ${f}\n\n| Operation | Does | In → out | Parameters (default) |\n|---|---|---|---|\n`;
  for (const o of ops.filter((o) => o.family === f)) {
    const ins = o.inputs.map((i) => (o.inputs.length > 1 ? `${i.name}: ${port(i.type)}` : port(i.type))).join(' + ');
    const params = o.params.length ? o.params.map((p) => `\`${p.name}\` ${value(p.default)}`).join(', ') : '—';
    md += `| **${o.title}** \`${o.op}\` | ${o.about.replace(/\|/g, '\\|')} | ${ins} → ${port(o.output)} | ${params} |\n`;
  }
  md += '\n';
}
await writeFile(new URL('../src/content/docs/reference/operations.mdx', import.meta.url), md);
console.log(`operations: ${ops.length} operations written`);
