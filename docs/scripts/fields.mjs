// The landing page's fields of view: for each entry point, a strip of nuclei outlines as a
// segmentation would draw them. Generated (seeded, so every build draws the same fields), not
// taken from any image. Written to src/data/fields.json before every build and dev run.
import { mkdir, writeFile } from 'node:fs/promises';

const WIDTH = 360;
const HEIGHT = 44;
const ROWS = 12;

// Small seeded generator (mulberry32): the same fields on every build.
function random(seed) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// One nucleus: a slightly irregular ellipse as a closed path (12 points, smoothed).
function nucleus(rnd, cx, cy, rx, ry, angle) {
  const n = 12;
  const pts = [];
  for (let i = 0; i < n; i++) {
    const t = (i / n) * Math.PI * 2;
    const wobble = 1 + (rnd() - 0.5) * 0.18;
    const x = Math.cos(t) * rx * wobble;
    const y = Math.sin(t) * ry * wobble;
    pts.push([cx + x * Math.cos(angle) - y * Math.sin(angle), cy + x * Math.sin(angle) + y * Math.cos(angle)]);
  }
  const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  const f = (p) => `${p[0].toFixed(1)} ${p[1].toFixed(1)}`;
  let d = `M${f(mid(pts[n - 1], pts[0]))}`;
  for (let i = 0; i < n; i++) d += ` Q${f(pts[i])} ${f(mid(pts[i], pts[(i + 1) % n]))}`;
  return `${d} Z`;
}

const rows = [];
for (let r = 0; r < ROWS; r++) {
  const rnd = random(1009 * (r + 1));
  const cells = [];
  let x = 6 + rnd() * 10;
  while (x < WIDTH - 10) {
    const rx = 5 + rnd() * 6;
    const ry = rx * (0.6 + rnd() * 0.35);
    const cy = 8 + ry + rnd() * (HEIGHT - 16 - 2 * ry);
    cells.push(nucleus(rnd, x + rx, cy, rx, ry, rnd() * Math.PI));
    x += 2 * rx + 3 + rnd() * 14;
  }
  rows.push(cells);
}

const out = new URL('../src/data/fields.json', import.meta.url);
await mkdir(new URL('.', out), { recursive: true });
await writeFile(out, JSON.stringify({ width: WIDTH, height: HEIGHT, rows }));
console.log(`fields: ${rows.length} fields of view written`);
