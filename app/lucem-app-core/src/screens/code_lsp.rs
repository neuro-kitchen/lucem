//! Built-in LSP `CompletionProvider` and `HoverProvider` for the **Code** workspace.
//!
//! Works identically on desktop and `wasm32` with sub-millisecond latency:
//! - Context-aware dot completions for `lucem.*`, `np.*`, and `scipy.ndimage.*` / `ndimage.*`
//! - `@` decorator completions (`@lucem.op`, `@lucem.transform`, `@lucem.param`)
//! - Live document symbol completions (`def`, function args, `import ... as ...`, local variables)
//! - Hover documentation popovers for Lucem decorators, NumPy, and SciPy image functions

use std::collections::BTreeSet;
use std::rc::Rc;

use anyhow::Result;
use gpui_kit::component::input::{CompletionProvider, HoverProvider, Rope};
use gpui_kit::{App, Entity, Task, Window};
use kitchen_ui::widgets::CodeView;
use lsp_types::{
    CompletionContext, CompletionItem, CompletionItemKind, CompletionResponse, Documentation, Hover, HoverContents,
    MarkupContent, MarkupKind,
};

/// Attaches the Lucem Python/Rust LSP completion & hover provider to a [`CodeView`].
pub fn attach(view: &Entity<CodeView>, cx: &mut App) {
    let editor = view.read(cx).editor().clone();
    let provider = Rc::new(LucemCodeLsp);
    editor.update(cx, |ed, _| {
        let lsp = ed.lsp_mut();
        lsp.completion_provider = Some(provider.clone());
        lsp.hover_provider = Some(provider);
    });
}

struct LucemCodeLsp;

fn item(label: &str, kind: CompletionItemKind, detail: &str, doc: &str, insert: Option<&str>) -> CompletionItem {
    CompletionItem {
        label: label.to_string(),
        kind: Some(kind),
        detail: Some(detail.to_string()),
        documentation: (!doc.is_empty()).then(|| {
            Documentation::MarkupContent(MarkupContent {
                kind: MarkupKind::Markdown,
                value: doc.to_string(),
            })
        }),
        insert_text: insert.map(str::to_string),
        ..Default::default()
    }
}

fn lucem_members() -> Vec<CompletionItem> {
    vec![
        item(
            "op",
            CompletionItemKind::FUNCTION,
            "@lucem.op(family=\"Filter\", halo=12)",
            "Register a Python function as a tiled Lucem pipeline operator. Typed array args become input ports and scalar args with defaults become Inspector parameters.",
            Some("op(family=\"Filter\", halo=12)"),
        ),
        item(
            "transform",
            CompletionItemKind::FUNCTION,
            "@lucem.transform(halo=12)",
            "Alias for `@lucem.op` — wraps a tile function with automatic halo padding, `/dev/shm` transfer, and tile reassembly.",
            Some("transform(halo=12)"),
        ),
        item(
            "param",
            CompletionItemKind::FUNCTION,
            "@lucem.param(\"name\", default)",
            "Explicitly declare an editable numeric parameter in the Pipeline Inspector.",
            Some("param(\"sigma\", 3.0)"),
        ),
        item(
            "input",
            CompletionItemKind::FUNCTION,
            "@lucem.input(\"image\", \"image\")",
            "Explicitly declare a named input port (`image`, `mask`, or `labels`).",
            Some("input(\"image\", \"image\")"),
        ),
        item(
            "custom",
            CompletionItemKind::MODULE,
            "lucem.custom",
            "Low-level tiled walker utilities (`lucem.custom.transform`, `lucem.custom.blocks`).",
            None,
        ),
    ]
}

fn numpy_members() -> Vec<CompletionItem> {
    vec![
        item("ndarray", CompletionItemKind::CLASS, "np.ndarray", "N-dimensional NumPy array (2D `float32` tile for Lucem images).", None),
        item("float32", CompletionItemKind::CLASS, "dtype", "32-bit IEEE floating-point scalar type.", None),
        item("uint32", CompletionItemKind::CLASS, "dtype", "32-bit unsigned integer scalar type (used for masks and label maps).", None),
        item("clip", CompletionItemKind::FUNCTION, "np.clip(a, a_min, a_max)", "Limit array values to `[a_min, a_max]`.", Some("clip(")),
        item("where", CompletionItemKind::FUNCTION, "np.where(cond, x, y)", "Select elements from `x` or `y` depending on `cond`.", Some("where(")),
        item("zeros", CompletionItemKind::FUNCTION, "np.zeros(shape, dtype=np.float32)", "Return a new array filled with zeros.", Some("zeros(")),
        item("zeros_like", CompletionItemKind::FUNCTION, "np.zeros_like(a)", "Return an array of zeros with the same shape and dtype as `a`.", Some("zeros_like(")),
        item("ones", CompletionItemKind::FUNCTION, "np.ones(shape, dtype=np.float32)", "Return a new array filled with ones.", Some("ones(")),
        item("percentile", CompletionItemKind::FUNCTION, "np.percentile(a, q)", "Compute the q-th percentile of the data.", Some("percentile(")),
        item("mean", CompletionItemKind::FUNCTION, "np.mean(a)", "Compute the arithmetic mean.", Some("mean(")),
        item("std", CompletionItemKind::FUNCTION, "np.std(a)", "Compute the standard deviation.", Some("std(")),
        item("median", CompletionItemKind::FUNCTION, "np.median(a)", "Compute the median.", Some("median(")),
        item("min", CompletionItemKind::FUNCTION, "np.min(a)", "Return the minimum of an array.", Some("min(")),
        item("max", CompletionItemKind::FUNCTION, "np.max(a)", "Return the maximum of an array.", Some("max(")),
        item("maximum", CompletionItemKind::FUNCTION, "np.maximum(x1, x2)", "Element-wise maximum of two arrays.", Some("maximum(")),
        item("minimum", CompletionItemKind::FUNCTION, "np.minimum(x1, x2)", "Element-wise minimum of two arrays.", Some("minimum(")),
        item("sqrt", CompletionItemKind::FUNCTION, "np.sqrt(x)", "Return the non-negative square-root of an array, element-wise.", Some("sqrt(")),
        item("log1p", CompletionItemKind::FUNCTION, "np.log1p(x)", "Return `log(1 + x)` element-wise (useful for high-dynamic-range fluorescence).", Some("log1p(")),
        item("abs", CompletionItemKind::FUNCTION, "np.abs(x)", "Element-wise absolute value.", Some("abs(")),
        item("stack", CompletionItemKind::FUNCTION, "np.stack(arrays, axis=0)", "Join a sequence of arrays along a new axis.", Some("stack(")),
        item("histogram", CompletionItemKind::FUNCTION, "np.histogram(a, bins=256)", "Compute the histogram of a dataset.", Some("histogram(")),
    ]
}

fn ndimage_members() -> Vec<CompletionItem> {
    vec![
        item("gaussian_filter", CompletionItemKind::FUNCTION, "gaussian_filter(input, sigma)", "Multidimensional Gaussian filter. Remember to set `halo >= ceil(3 * sigma)` in `@lucem.op`.", Some("gaussian_filter(")),
        item("median_filter", CompletionItemKind::FUNCTION, "median_filter(input, size)", "Calculate a multidimensional median filter.", Some("median_filter(")),
        item("uniform_filter", CompletionItemKind::FUNCTION, "uniform_filter(input, size)", "Multidimensional box/mean filter.", Some("uniform_filter(")),
        item("sobel", CompletionItemKind::FUNCTION, "sobel(input)", "Calculate a Sobel edge filter.", Some("sobel(")),
        item("laplace", CompletionItemKind::FUNCTION, "laplace(input)", "N-D Laplace filter based on approximate second derivatives.", Some("laplace(")),
        item("maximum_filter", CompletionItemKind::FUNCTION, "maximum_filter(input, size)", "Multidimensional maximum filter (grayscale dilation).", Some("maximum_filter(")),
        item("minimum_filter", CompletionItemKind::FUNCTION, "minimum_filter(input, size)", "Multidimensional minimum filter (grayscale erosion).", Some("minimum_filter(")),
        item("binary_erosion", CompletionItemKind::FUNCTION, "binary_erosion(input)", "Multidimensional binary erosion.", Some("binary_erosion(")),
        item("binary_dilation", CompletionItemKind::FUNCTION, "binary_dilation(input)", "Multidimensional binary dilation.", Some("binary_dilation(")),
        item("binary_opening", CompletionItemKind::FUNCTION, "binary_opening(input)", "Multidimensional binary opening.", Some("binary_opening(")),
        item("binary_closing", CompletionItemKind::FUNCTION, "binary_closing(input)", "Multidimensional binary closing.", Some("binary_closing(")),
        item("distance_transform_edt", CompletionItemKind::FUNCTION, "distance_transform_edt(input)", "Exact Euclidean distance transform.", Some("distance_transform_edt(")),
        item("label", CompletionItemKind::FUNCTION, "label(input)", "Label connected components in a binary mask.", Some("label(")),
    ]
}

fn array_method_members() -> Vec<CompletionItem> {
    vec![
        item("astype", CompletionItemKind::METHOD, ".astype(np.float32, copy=False)", "Copy/cast array to a specified dtype.", Some("astype(np.float32, copy=False)")),
        item("shape", CompletionItemKind::PROPERTY, "tuple[int, ...]", "Tuple of array dimensions `(height, width)`.", None),
        item("dtype", CompletionItemKind::PROPERTY, "np.dtype", "Data-type of the array's elements.", None),
        item("min", CompletionItemKind::METHOD, ".min()", "Return the minimum value.", Some("min()")),
        item("max", CompletionItemKind::METHOD, ".max()", "Return the maximum value.", Some("max()")),
        item("mean", CompletionItemKind::METHOD, ".mean()", "Return the arithmetic mean.", Some("mean()")),
        item("std", CompletionItemKind::METHOD, ".std()", "Return the standard deviation.", Some("std()")),
        item("clip", CompletionItemKind::METHOD, ".clip(min, max)", "Clip values between `min` and `max`.", Some("clip(")),
        item("copy", CompletionItemKind::METHOD, ".copy()", "Return a copy of the array.", Some("copy()")),
    ]
}

fn extract_doc_symbols(src: &str) -> Vec<CompletionItem> {
    let mut seen = BTreeSet::new();
    let mut items = Vec::new();
    for line in src.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("def ") {
            if let Some(name) = rest.split('(').next().map(str::trim)
                && is_ident(name)
                && seen.insert(name.to_string())
            {
                items.push(item(name, CompletionItemKind::FUNCTION, "function", "Defined in this script.", None));
            }
            if let Some(open) = rest.find('(')
                && let Some(close) = rest.rfind(')')
            {
                for arg in rest[open + 1..close].split(',') {
                    let param = arg.split(':').next().unwrap_or(arg).split('=').next().unwrap_or(arg).trim();
                    if is_ident(param) && seen.insert(param.to_string()) {
                        items.push(item(param, CompletionItemKind::VARIABLE, "parameter", "Function parameter.", None));
                    }
                }
            }
        } else if t.starts_with("import ") || t.starts_with("from ") {
            if let Some((_, alias)) = t.rsplit_once(" as ") {
                let a = alias.trim();
                if is_ident(a) && seen.insert(a.to_string()) {
                    items.push(item(a, CompletionItemKind::MODULE, "import", "Imported alias.", None));
                }
            } else if let Some((_, imported)) = t.split_once(" import ") {
                for sym in imported.split(',') {
                    let s = sym.trim();
                    if is_ident(s) && seen.insert(s.to_string()) {
                        items.push(item(s, CompletionItemKind::FUNCTION, "import", "Imported symbol.", None));
                    }
                }
            }
        } else if let Some((lhs, _)) = t.split_once('=')
            && !lhs.contains('(')
            && !lhs.contains('[')
            && !t.starts_with('#')
            && !t.starts_with('@')
        {
            let name = lhs.split(':').next().unwrap_or(lhs).trim();
            if is_ident(name) && seen.insert(name.to_string()) {
                items.push(item(name, CompletionItemKind::VARIABLE, "local", "Local variable.", None));
            }
        }
    }
    items
}

fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    let Some(first) = chars.next() else { return false };
    (first.is_ascii_alphabetic() || first == '_') && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn prefix_context(src: &str, offset: usize) -> (&str, &str) {
    let clamped = offset.min(src.len());
    let before = &src[..clamped];
    let line_start = before.rfind('\n').map_or(0, |i| i + 1);
    let line = &before[line_start..];
    // Find current identifier token and optional qualifier before `.` or `@`
    let ident_start = line
        .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .map_or(0, |i| i + 1);
    let prefix = &line[ident_start..];
    let head = &line[..ident_start];
    if let Some(before_dot) = head.strip_suffix('.') {
        let qual_start = before_dot
            .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '.'))
            .map_or(0, |i| i + 1);
        return (&before_dot[qual_start..], prefix);
    }
    if head.ends_with('@') {
        return ("@", prefix);
    }
    ("", prefix)
}

fn word_at(src: &str, offset: usize) -> String {
    let clamped = offset.min(src.len());
    let bytes = src.as_bytes();
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_' || b == b'.';
    let mut start = clamped;
    while start > 0 && is_word(bytes[start - 1]) {
        start -= 1;
    }
    let mut end = clamped;
    while end < bytes.len() && is_word(bytes[end]) {
        end += 1;
    }
    src[start..end].to_string()
}

fn hover_doc(symbol: &str) -> Option<&'static str> {
    let tail = symbol.rsplit('.').next().unwrap_or(symbol);
    match symbol {
        "lucem.op" | "op" => Some("**`@lucem.op(family=\"Filter\", halo=12)`**\n\nRegisters a Python function as a tiled Lucem pipeline operator.\n- Typed array arguments (`np.ndarray`, `Mask`, `Labels`) become graph input ports.\n- Scalar arguments with defaults (`sigma: float = 3.0`) become Inspector scrubbers.\n- `halo` adds border overlap in pixels around every tile and crops it back automatically."),
        "lucem.transform" | "transform" => Some("**`@lucem.transform(halo=12)`**\n\nAlias for `@lucem.op`. Lucem automatically reads each tile with `halo` padding, transfers it via `/dev/shm` zero-copy `np.memmap`, and reassembles the output."),
        "lucem.param" => Some("**`@lucem.param(name, default)`**\n\nExplicitly declares a numeric parameter shown in the Pipeline Inspector."),
        "np.ndarray" | "ndarray" => Some("**`np.ndarray`**\n\n2D `float32` tile array (`shape = (height, width)`) passed zero-copy from Lucem's tile engine."),
        _ => match tail {
            "gaussian_filter" => Some("**`scipy.ndimage.gaussian_filter(input, sigma)`**\n\nApplies a Gaussian blur. Set `halo >= ceil(3 * sigma)` in `@lucem.op(...)` to eliminate tile seam artifacts."),
            "median_filter" => Some("**`scipy.ndimage.median_filter(input, size)`**\n\nApplies a multidimensional median filter. Set `halo >= size // 2`."),
            "clip" => Some("**`np.clip(a, a_min, a_max)`**\n\nClamps pixel values into `[a_min, a_max]`."),
            "percentile" => Some("**`np.percentile(a, q)`**\n\nComputes the `q`-th percentile across the tile."),
            "astype" => Some("**`arr.astype(np.float32, copy=False)`**\n\nEnsures contiguous `float32` output for Lucem's tile buffer."),
            "halo" => Some("**`halo: int`**\n\nNumber of extra border pixels Lucem reads around each tile before calling your function, then crops away after — eliminating seam artifacts on spatial filters."),
            _ => None,
        },
    }
}

impl CompletionProvider for LucemCodeLsp {
    fn completions(
        &self,
        text: &Rope,
        offset: usize,
        _trigger: CompletionContext,
        _window: &mut Window,
        _cx: &mut App,
    ) -> Task<Result<CompletionResponse>> {
        let src = text.to_string();
        let (qualifier, prefix) = prefix_context(&src, offset);
        let prefix_lower = prefix.to_ascii_lowercase();
        let mut candidates = match qualifier {
            "lucem" => lucem_members(),
            "lucem.custom" => vec![
                item("transform", CompletionItemKind::FUNCTION, "@lucem.custom.transform(halo=12)", "Wrap a tile transform with automatic halo overlap.", Some("transform(halo=12)")),
                item("blocks", CompletionItemKind::FUNCTION, "lucem.custom.blocks(pyramid, halo=12)", "Iterate halo-padded tiles across a pyramid.", Some("blocks(")),
            ],
            "np" | "numpy" => numpy_members(),
            "ndimage" | "scipy.ndimage" => ndimage_members(),
            "@" => vec![
                item("lucem.op", CompletionItemKind::SNIPPET, "@lucem.op(family=\"Filter\", halo=12)", "Declare a tiled Lucem operator from the decorated function.", Some("lucem.op(family=\"Filter\", halo=12)")),
                item("lucem.transform", CompletionItemKind::SNIPPET, "@lucem.transform(halo=12)", "Declare a tiled Lucem tile transform.", Some("lucem.transform(halo=12)")),
                item("lucem.param", CompletionItemKind::SNIPPET, "@lucem.param(\"sigma\", 3.0)", "Declare a numeric parameter for the Inspector.", Some("lucem.param(\"sigma\", 3.0)")),
            ],
            q if !q.is_empty() => array_method_members(),
            _ => {
                let mut list = extract_doc_symbols(&src);
                list.extend([
                    item("lucem", CompletionItemKind::MODULE, "module", "Lucem Python plugin API (`@lucem.op`, `@lucem.transform`).", None),
                    item("np", CompletionItemKind::MODULE, "numpy", "NumPy array library.", None),
                    item("gaussian_filter", CompletionItemKind::FUNCTION, "scipy.ndimage.gaussian_filter", "Gaussian smoothing filter.", Some("gaussian_filter(")),
                    item("def", CompletionItemKind::KEYWORD, "keyword", "Define a Python function.", None),
                    item("return", CompletionItemKind::KEYWORD, "keyword", "Return the processed tile array.", None),
                    item("import", CompletionItemKind::KEYWORD, "keyword", "Import a Python module.", None),
                    item("from", CompletionItemKind::KEYWORD, "keyword", "Import symbols from a module.", None),
                    item("float", CompletionItemKind::CLASS, "builtin", "Floating-point scalar type.", None),
                    item("int", CompletionItemKind::CLASS, "builtin", "Integer scalar type.", None),
                    item("bool", CompletionItemKind::CLASS, "builtin", "Boolean scalar type.", None),
                ]);
                list
            }
        };
        if !prefix_lower.is_empty() {
            candidates.retain(|c| c.label.to_ascii_lowercase().starts_with(&prefix_lower));
        }
        Task::ready(Ok(CompletionResponse::Array(candidates)))
    }

    fn is_completion_trigger(&self, _offset: usize, new_text: &str, _cx: &mut App) -> bool {
        matches!(new_text, "." | "@")
            || new_text.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
    }
}

impl HoverProvider for LucemCodeLsp {
    fn hover(
        &self,
        text: &Rope,
        offset: usize,
        _window: &mut Window,
        _cx: &mut App,
    ) -> Task<Result<Option<Hover>>> {
        let src = text.to_string();
        let sym = word_at(&src, offset);
        let hover = hover_doc(&sym).map(|md| Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: md.to_string(),
            }),
            range: None,
        });
        Task::ready(Ok(hover))
    }
}
