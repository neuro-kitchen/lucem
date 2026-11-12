"""Writes the Rust API reference (docs/src/content/docs/reference/rust/) from rustdoc's JSON, so every
public function, type and method is listed with its exact signature and its doc comment, and the
pages cannot drift from the code. lucem is not on docs.rs, so these pages stand in for it.

Needs the nightly toolchain (rustdoc's JSON output is unstable). From the repository root:

    for p in lucem-core lucem-io lucem-ops lucem-ml lucem-hist lucem-pipeline lucem-hub; do
      CARGO_TARGET_DIR=target/rustdoc-json cargo +nightly rustdoc -p $p --lib -- \
        -Z unstable-options --output-format json
    done
    python3 docs/scripts/rust_api.py

The pages are committed; rerun when a public API changes. Standard library only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JSON_DIR = ROOT / "target" / "rustdoc-json" / "doc"
OUT = ROOT / "docs" / "src" / "content" / "docs" / "reference" / "rust"

# (package, what it is). Order is the order of the pages in the sidebar.
CRATES = [
    ("lucem-core", "Device arrays and memory, shapes, compute targets (runtimes), tiling"),
    ("lucem-io", "Reading and writing images: TIFF, OME-Zarr, pyramids, projects"),
    ("lucem-ops", "Image operations on the device: filters, thresholds, morphology, segmentation, measurements"),
    ("lucem-pipeline", "Pipelines as data: the operation catalog, the executor, tables"),
    ("lucem-hist", "Histology: tissue, cells, classifiers, densities, the analysis engine and sessions"),
    ("lucem-ml", "The model-plugin protocol: discovery, client, messages"),
    ("lucem-hub", "Verified downloads of published artifacts (model weights)"),
]
# Crates whose top-level modules each get a page (the rest get one page per crate).
SPLIT = {"lucem-ops", "lucem-hist"}
# Modules that are implementation detail even when public (GPU kernels, test helpers).
SKIP_MODULES = {"kernels", "test_support", "tests"}

FRONT = """---
title: "{title}"
description: "{description}"
sidebar:
  label: "{label}"
  order: {order}
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->
"""


class Crate:
    def __init__(self, package: str):
        self.package = package
        self.name = package.replace("-", "_")
        self.doc = json.loads((JSON_DIR / f"{self.name}.json").read_text())
        self.index = self.doc["index"]

    def item(self, id_):
        return self.index.get(str(id_))

    def root(self):
        return self.item(self.doc["root"])


# ---- types -------------------------------------------------------------------------------------


def ty(t) -> str:
    if t is None:
        return "()"
    if isinstance(t, str):
        return t
    (kind, v), = t.items()
    if kind == "primitive":
        return v
    if kind == "generic":
        return v
    if kind == "resolved_path":
        return path_name(v["path"]) + args(v.get("args"))
    if kind == "borrowed_ref":
        lt = f"{v['lifetime']} " if v.get("lifetime") else ""
        mut = "mut " if v.get("is_mutable") else ""
        return f"&{lt}{mut}{ty(v['type'])}"
    if kind == "raw_pointer":
        return f"*{'mut' if v.get('is_mutable') else 'const'} {ty(v['type'])}"
    if kind == "slice":
        return f"[{ty(v)}]"
    if kind == "array":
        return f"[{ty(v['type'])}; {v['len']}]"
    if kind == "tuple":
        inner = ", ".join(ty(x) for x in v)
        return f"({inner},)" if len(v) == 1 else f"({inner})"
    if kind == "impl_trait":
        return "impl " + " + ".join(bound(b) for b in v)
    if kind == "dyn_trait":
        s = " + ".join(poly(p) for p in v["traits"])
        if v.get("lifetime"):
            s += f" + {v['lifetime']}"
        return f"dyn {s}"
    if kind == "qualified_path":
        self_ty = ty(v["self_type"])
        trait = v.get("trait")
        if trait:
            return f"<{self_ty} as {path_name(trait['path'])}>::{v['name']}{args(v.get('args'))}"
        return f"{self_ty}::{v['name']}{args(v.get('args'))}"
    if kind == "function_pointer":
        sig = v["sig"]
        ins = ", ".join(ty(i[1]) for i in sig["inputs"])
        out = sig.get("output")
        return f"fn({ins})" + (f" -> {ty(out)}" if out else "")
    if kind == "infer":
        return "_"
    if kind == "pat":
        return ty(v["type"])
    return "?"


def path_name(p: str) -> str:
    # Keep the last segment: `std::sync::Arc` reads as `Arc` in a signature.
    return p.split("::")[-1]


def args(a) -> str:
    if not a:
        return ""
    (kind, v), = a.items()
    if kind == "angle_bracketed":
        parts = []
        for g in v.get("args", []):
            (k, x), = g.items()
            parts.append(x if k == "lifetime" else ty(x) if k == "type" else str(x.get("expr", x)) if k == "const" else "_")
        for c in v.get("constraints", []):
            b = c.get("binding", {})
            if "equality" in b:
                eq = b["equality"]
                parts.append(f"{c['name']} = {ty(eq['type']) if isinstance(eq, dict) and 'type' in eq else ty(eq)}")
            elif "constraint" in b:
                parts.append(f"{c['name']}: " + " + ".join(bound(x) for x in b["constraint"]))
        return f"<{', '.join(parts)}>" if parts else ""
    if kind == "parenthesized":
        ins = ", ".join(ty(x) for x in v.get("inputs", []))
        out = v.get("output")
        return f"({ins})" + (f" -> {ty(out)}" if out else "")
    return ""


def poly(p) -> str:
    return path_name(p["trait"]["path"]) + args(p["trait"].get("args"))


def bound(b) -> str:
    (kind, v), = b.items()
    if kind == "trait_bound":
        m = "?" if v.get("modifier") == "maybe" else ""
        return m + path_name(v["trait"]["path"]) + args(v["trait"].get("args"))
    if kind == "outlives":
        return v
    if kind == "use":
        return "use<..>"
    return "?"


def generics(g) -> tuple[str, str]:
    params = []
    for p in g.get("params", []):
        kind = p["kind"]
        if "lifetime" in kind:
            params.append(p["name"])
        elif "type" in kind:
            if kind["type"].get("is_synthetic"):
                continue  # `impl Trait` arguments
            bs = kind["type"].get("bounds", [])
            params.append(p["name"] + (": " + " + ".join(bound(b) for b in bs) if bs else ""))
        elif "const" in kind:
            params.append(f"const {p['name']}: {ty(kind['const']['type'])}")
    wheres = []
    for w in g.get("where_predicates", []):
        if "bound_predicate" in w:
            bp = w["bound_predicate"]
            wheres.append(f"{ty(bp['type'])}: " + " + ".join(bound(b) for b in bp["bounds"]))
    return (f"<{', '.join(params)}>" if params else "", ("\nwhere\n    " + ",\n    ".join(wheres)) if wheres else "")


def fn_sig(name: str, f) -> str:
    sig = f["sig"]
    head = f.get("header", {})
    pre = ("const " if head.get("is_const") else "") + ("async " if head.get("is_async") else "") + ("unsafe " if head.get("is_unsafe") else "")
    gen, where = generics(f.get("generics", {}))
    ins = []
    for arg_name, t in sig["inputs"]:
        if arg_name == "self":
            s = ty(t)
            ins.append("self" if s == "Self" else "&self" if s == "&Self" else "&mut self" if s == "&mut Self" else f"self: {s}")
        else:
            ins.append(f"{arg_name}: {ty(t)}")
    out = sig.get("output")
    one = f"pub {pre}fn {name}{gen}({', '.join(ins)})" + (f" -> {ty(out)}" if out else "")
    if len(one) > 96 and ins:
        one = f"pub {pre}fn {name}{gen}(\n" + "".join(f"    {i},\n" for i in ins) + ")" + (f" -> {ty(out)}" if out else "")
    return one + where


# ---- items -------------------------------------------------------------------------------------


def kind_of(it) -> str:
    return next(iter(it["inner"])) if isinstance(it.get("inner"), dict) else str(it.get("inner"))


def is_public(it) -> bool:
    return it.get("visibility") == "public"


def doc_text(it) -> str:
    d = (it.get("docs") or "").strip()
    # Intra-doc links read as code: [`Foo`] -> `Foo`, [text](path) -> text when the target is a Rust path.
    d = re.sub(r"\[(`[^`]+`)\](?!\()", r"\1", d)
    d = re.sub(r"\[([^\]]+)\]\((?!https?:)[^)]*\)", r"\1", d)
    # Headings inside a doc comment sit below the item's heading.
    d = re.sub(r"^(#+) ", lambda m: "#" * min(len(m.group(1)) + 3, 6) + " ", d, flags=re.M)
    # Doc-test code fences: hide lines starting with `# ` the way rustdoc does; untagged fences are Rust.
    out, in_code = [], False
    for line in d.split("\n"):
        if line.strip().startswith("```"):
            if not in_code:
                tag = line.strip()[3:].strip()
                line = "```rust" if tag in ("", "no_run", "ignore", "should_panic", "compile_fail") or tag.startswith("rust") else line
            in_code = not in_code
        elif in_code and (line.strip() == "#" or line.strip().startswith("# ")):
            continue
        out.append(line)
    return "\n".join(out)


def methods(c: Crate, it) -> list:
    """Public inherent methods (not trait impls) of a struct or enum, in source order."""
    inner = it["inner"][kind_of(it)]
    found = []
    for impl_id in inner.get("impls", []):
        impl = c.item(impl_id)
        if not impl:
            continue
        im = impl["inner"].get("impl", {})
        if im.get("trait") is not None or im.get("is_synthetic") or im.get("blanket_impl"):
            continue
        for mid in im.get("items", []):
            m = c.item(mid)
            if m and is_public(m) and kind_of(m) == "function":
                found.append(m)
    return sorted(found, key=lambda m: (m.get("span") or {}).get("begin", [0])[0])


def trait_impls(c: Crate, it) -> list[str]:
    inner = it["inner"][kind_of(it)]
    names = []
    for impl_id in inner.get("impls", []):
        impl = c.item(impl_id)
        im = impl["inner"].get("impl", {}) if impl else {}
        tr = im.get("trait")
        if tr and not im.get("is_synthetic") and not im.get("blanket_impl"):
            n = path_name(tr["path"])
            if n not in ("StructuralPartialEq",):
                names.append(n)
    return sorted(set(names))


def fields(c: Crate, it) -> list[str]:
    k = kind_of(it)
    lines = []
    if k == "struct":
        sk = it["inner"]["struct"]["kind"]
        if isinstance(sk, dict) and "plain" in sk:
            for fid in sk["plain"]["fields"]:
                f = c.item(fid)
                if f and is_public(f):
                    lines.append((f"{f['name']}: {ty(f['inner']['struct_field'])}", first_line(f)))
        elif isinstance(sk, dict) and "tuple" in sk:
            ts = [ty(c.item(fid)["inner"]["struct_field"]) if fid is not None and c.item(fid) and is_public(c.item(fid)) else "_" for fid in sk["tuple"]]
            lines.append((f"({', '.join(ts)})", ""))
    if k == "enum":
        for vid in it["inner"]["enum"]["variants"]:
            v = c.item(vid)
            if not v:
                continue
            vk = v["inner"]["variant"]["kind"]
            s = v["name"]
            if isinstance(vk, dict) and "tuple" in vk:
                s += "(" + ", ".join(ty(c.item(f)["inner"]["struct_field"]) if f is not None and c.item(f) else "_" for f in vk["tuple"]) + ")"
            elif isinstance(vk, dict) and "struct" in vk:
                s += " { " + ", ".join(f"{c.item(f)['name']}: {ty(c.item(f)['inner']['struct_field'])}" for f in vk["struct"]["fields"] if c.item(f)) + " }"
            lines.append((s, first_line(v)))
    return lines


def first_line(it) -> str:
    d = (it.get("docs") or "").strip().split("\n\n")[0].replace("\n", " ")
    d = re.sub(r"\[(`[^`]+`)\](?!\()", r"\1", d)
    return d


def type_head(it) -> str:
    k = kind_of(it)
    gen, where = generics(it["inner"][k].get("generics", {}))
    word = {"struct": "struct", "enum": "enum", "trait": "trait", "type_alias": "type", "union": "union"}[k]
    if k == "type_alias":
        return f"pub type {it['name']}{gen} = {ty(it['inner'][k]['type'])};"
    return f"pub {word} {it['name']}{gen}{where}"


def render_item(c: Crate, it, out: list[str]) -> None:
    k = kind_of(it)
    name = it["name"]
    if k == "function":
        out += [f"### `{name}`", "", "```rust", fn_sig(name, it["inner"]["function"]), "```", ""]
        d = doc_text(it)
        if d:
            out += [d, ""]
        return
    if k in ("struct", "enum", "union", "type_alias", "trait"):
        out += [f"### `{name}`", "", "```rust", type_head(it), "```", ""]
        d = doc_text(it)
        if d:
            out += [d, ""]
        fs = fields(c, it) if k in ("struct", "enum") else []
        if fs:
            out += ["**Variants**" if k == "enum" else "**Fields**", ""]
            for s, d1 in fs:
                out.append(f"- `{s}`" + (f": {d1}" if d1 else ""))
            out.append("")
        if k == "trait":
            req = []
            for mid in it["inner"]["trait"].get("items", []):
                m = c.item(mid)
                if m and kind_of(m) == "function":
                    req.append(m)
            if req:
                out += ["**Methods**", "", "```rust"]
                out += [fn_sig(m["name"], m["inner"]["function"]).replace("pub fn", "fn") + ";" for m in req]
                out += ["```", ""]
        ms = methods(c, it) if k in ("struct", "enum", "union") else []
        if ms:
            out += ["**Methods**", ""]
            for m in ms:
                out += ["```rust", fn_sig(m["name"], m["inner"]["function"]), "```", ""]
                d1 = doc_text(m)
                if d1:
                    out += [d1, ""]
        impls = trait_impls(c, it) if k in ("struct", "enum", "union") else []
        if impls:
            out += ["Implements " + ", ".join(f"`{n}`" for n in impls) + ".", ""]
        return
    if k == "constant":
        cst = it["inner"]["constant"]
        out += [f"### `{name}`", "", "```rust", f"pub const {name}: {ty(cst['type'])} = {cst['const'].get('expr', '…')};", "```", ""]
        d = doc_text(it)
        if d:
            out += [d, ""]
        return
    if k == "static":
        out += [f"### `{name}`", "", "```rust", f"pub static {name}: {ty(it['inner']['static']['type'])};", "```", ""]
        d = doc_text(it)
        if d:
            out += [d, ""]


ORDER = {"struct": 1, "enum": 2, "trait": 3, "type_alias": 4, "union": 5, "function": 6, "constant": 7, "static": 8}
SECTION = {"struct": "Types", "enum": "Types", "union": "Types", "type_alias": "Types", "trait": "Traits", "function": "Functions", "constant": "Constants", "static": "Constants"}


def module_items(c: Crate, mod):
    """(items defined or re-exported here, submodules), public only."""
    items, subs = [], []
    for iid in mod["inner"]["module"]["items"]:
        it = c.item(iid)
        if not it or not is_public(it):
            continue
        k = kind_of(it)
        if k == "module":
            if it["name"] not in SKIP_MODULES:
                subs.append(it)
        elif k == "use":
            u = it["inner"]["use"]
            target = c.item(u["id"]) if u.get("id") is not None else None
            if target and kind_of(target) in ORDER and not u.get("is_glob"):
                items.append((u["name"], target))
        elif k in ORDER:
            items.append((it["name"], it))
    return items, subs


def render_module(c: Crate, mod, path: str, out: list[str], depth: int, seen: set) -> int:
    items, subs = module_items(c, mod)
    count = 0
    own = [(n, it) for n, it in items if id(it) not in seen]
    if own or depth == 0:
        if depth > 0:
            out += [f"## `{path}`", ""]
            d = first_para(mod)
            if d:
                out += [d, ""]
        last = None
        for name, it in sorted(own, key=lambda x: (ORDER[kind_of(x[1])], src(x[1]))):
            seen.add(id(it))
            sec = SECTION[kind_of(it)]
            if sec != last and depth == 0:
                out += [f"## {sec}", ""]
            last = sec
            it = dict(it, name=name)
            render_item(c, it, out)
            count += 1
    for s in sorted(subs, key=lambda m: m["name"]):
        count += render_module(c, s, f"{path}::{s['name']}", out, depth + 1, seen)
    return count


def src(it) -> tuple:
    """Where an item is defined (file, line): related items (one file) stay together."""
    sp = it.get("span") or {}
    return (sp.get("filename", ""), sp.get("begin", [0])[0])


def first_para(it) -> str:
    d = doc_text(it)
    return d.split("\n\n")[0] if d else ""


def write(rel: str, title: str, description: str, label: str, order: int, body: list[str]) -> None:
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    esc = lambda s: s.replace('"', '\\"')
    p.write_text(FRONT.format(title=esc(title), description=esc(description), label=esc(label), order=order) + "\n" + "\n".join(body).rstrip() + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.rglob("*.md"):
        if old.name != "index.md":
            old.unlink()
    total = 0
    pages = []
    for order, (package, what) in enumerate(CRATES, start=1):
        c = Crate(package)
        root = c.root()
        crate_doc = doc_text(root)
        if package in SPLIT:
            items, subs = module_items(c, root)
            body = [crate_doc, ""] if crate_doc else []
            if items:
                body += ["## At the crate root", ""]
                for name, it in items:
                    render_item(c, dict(it, name=name), body)
                    total += 1
            body += ["## Modules", "", "| Module | What it holds |", "|---|---|"]
            for i, s in enumerate(sorted(subs, key=lambda m: m["name"])):
                body.append(f"| [`{c.name}::{s['name']}`](/lucem/reference/rust/{package}/{s['name']}/) | {first_para(s).replace(chr(10), ' ') or '—'} |")
                sub = []
                d = doc_text(s)
                if d:
                    sub += [d, ""]
                n = render_module(c, s, f"{c.name}::{s['name']}", sub, 0, set())
                total += n
                write(f"{package}/{s['name']}.md", f"{c.name}::{s['name']}", first_line(s) or f"The {s['name']} module of {package}.", s["name"], i + 1, sub)
            write(f"{package}/index.md", c.name, what + ".", "Overview", 0, body)
            pages.append((package, what, f"{package}/"))
        else:
            body = [crate_doc, ""] if crate_doc else []
            total += render_module(c, root, c.name, body, 0, set())
            write(f"{package}.md", c.name, what + ".", package, order, body)
            pages.append((package, what, f"{package}/"))
    print(f"rust api: {total} items on {len(list(OUT.rglob('*.md'))) - 1} pages")


if __name__ == "__main__":
    main()
