// The page's loader (a file of its own: the page's Content-Security-Policy allows no inline
// script).
// The loader: before the app (several MB) is downloaded, find what this browser offers.
//
// - Drawing: WebGPU where it is presented fast, else WebGL2; never in software.
// - Threads: the page must be cross-origin isolated (`lucem serve --web` serves it so;
//   elsewhere a service worker of this page adds the headers, after one reload).
// - The connection: WebTransport (below, `window.lucemTransport`, which the app's
//   `lucem_stream::web_client` drives), else the app falls back to a WebSocket.
// - `?gpu=webgl` / `?gpu=webgpu` force the drawing backend (for measuring only).

// ---- WebTransport for the app ------------------------------------------------------------
// Messages are a 4-byte big-endian length, then the bytes. `open(url, hash)` connects (a
// self-signed certificate is accepted by its SHA-256 only) and opens the control stream
// (who the page is, as JSON); `request(bytes)` opens a stream for one request and resolves
// with its answer. Bytes handed in are copies outside the app's shared memory.
const frame = (bytes) => {
  const out = new Uint8Array(4 + bytes.length);
  new DataView(out.buffer).setUint32(0, bytes.length);
  out.set(bytes, 4);
  return out;
};
const framed = (reader) => {
  let buf = new Uint8Array(0);
  const need = async (n) => {
    while (buf.length < n) {
      const { value, done } = await reader.read();
      if (done) throw new Error("the stream closed");
      const grown = new Uint8Array(buf.length + value.length);
      grown.set(buf);
      grown.set(value, buf.length);
      buf = grown;
    }
  };
  return {
    async next() {
      await need(4);
      const len = new DataView(buf.buffer, buf.byteOffset).getUint32(0);
      await need(4 + len);
      const out = buf.slice(4, 4 + len);
      buf = buf.slice(4 + len);
      return out;
    },
  };
};
if ("WebTransport" in window) {
  window.lucemTransport = {
    async open(url, hash) {
      const options = hash && hash.length ? { serverCertificateHashes: [{ algorithm: "sha-256", value: hash.slice() }] } : {};
      const wt = new WebTransport(url, options);
      await wt.ready;
      const control = await wt.createBidirectionalStream();
      const writer = control.writable.getWriter();
      const reader = framed(control.readable.getReader());
      const encoder = new TextEncoder(), decoder = new TextDecoder();
      return {
        control: {
          send: async (text) => writer.write(frame(encoder.encode(text))),
          next: async () => decoder.decode(await reader.next()),
        },
        async request(bytes) {
          const stream = await wt.createBidirectionalStream();
          const w = stream.writable.getWriter();
          await w.write(frame(bytes));
          // Our side ends here, not awaited: a server that answered at once (a cached tile) may
          // close the stream first, and the close then "fails" with STOP_SENDING although the
          // answer is there
          w.close().catch(() => {});
          return framed(stream.readable.getReader()).next();
        },
        close() {
          wt.close();
        },
      };
    },
  };
}

// ---- picking files (a page that runs alone: the docs viewer) ---------------------------------
// Resolve to [{path: "<folder>/<path inside>", file}] or null (closed). Nothing is read: a File is
// a handle until a tile needs its bytes. dsp-kitchen's picker awaited getFile() for each of
// ~20,000 files in turn before anything showed; here the names are listed first and the Files
// made in parallel batches.
const pickWithInput = (folder) =>
  new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    if (folder) input.webkitdirectory = true;
    input.onchange = () => resolve([...input.files].map((f) => ({ path: f.webkitRelativePath || f.name, file: f })));
    input.oncancel = () => resolve(null);
    input.click();
  });
window.lucemPickFiles = () => pickWithInput(false);
window.lucemPickFolder = async () => {
  if (!window.showDirectoryPicker) return pickWithInput(true);
  let root;
  try {
    root = await window.showDirectoryPicker({ mode: "read" });
  } catch {
    return null;
  }
  const handles = [];
  const MAX = 200000, DEPTH = 10;
  const walk = async (dir, prefix, depth) => {
    for await (const [name, handle] of dir.entries()) {
      if (handles.length >= MAX) return;
      if (handle.kind === "file") handles.push([`${prefix}/${name}`, handle]);
      else if (!name.startsWith(".") && depth < DEPTH) await walk(handle, `${prefix}/${name}`, depth + 1);
    }
  };
  await walk(root, root.name, 0);
  const out = [];
  for (let i = 0; i < handles.length; i += 512) {
    out.push(...(await Promise.all(handles.slice(i, i + 512).map(async ([path, h]) => ({ path, file: await h.getFile() })))));
  }
  return out;
};

// ---- the browser -------------------------------------------------------------------------
const note = document.getElementById("note");
const say = (html) => (note.firstElementChild.innerHTML = html);
const params = new URLSearchParams(location.search);

/** Hardware WebGL2: refused when the browser would draw in software. */
function webgl() {
  const gl = document.createElement("canvas").getContext("webgl2", { failIfMajorPerformanceCaveat: true });
  if (!gl) return null;
  const ext = gl.getExtension("WEBGL_debug_renderer_info");
  const renderer = String(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
  gl.getExtension("WEBGL_lose_context")?.loseContext();
  return /swiftshader|llvmpipe|softpipe|software|basic render/i.test(renderer) ? null : renderer;
}

/** A hardware WebGPU adapter, or null. */
async function webgpu() {
  if (!navigator.gpu) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter || adapter.isFallbackAdapter || adapter.info?.isFallbackAdapter) return null;
    return `${adapter.info?.vendor ?? ""} ${adapter.info?.architecture ?? ""}`.trim() || "WebGPU";
  } catch {
    return null;
  }
}

if (!window.crossOriginIsolated && "serviceWorker" in navigator && location.protocol !== "file:") {
  const tried = sessionStorage.getItem("lucem-isolate");
  if (!tried) {
    sessionStorage.setItem("lucem-isolate", "1");
    await navigator.serviceWorker.register("./isolate-sw.js");
    await navigator.serviceWorker.ready;
    location.reload();
    await new Promise(() => {});
  }
}

const [gl, gpu] = [webgl(), await webgpu()];
// Chrome on Linux presents a WebGPU canvas slowly (dsp-kitchen measured it, 2026-11-15)
const linuxChrome = /Linux/.test(navigator.userAgent) && !/Android/.test(navigator.userAgent) && /Chrome\//.test(navigator.userAgent);
let drawing = params.get("gpu") ?? (gpu && !linuxChrome ? "webgpu" : gl ? "webgl" : gpu ? "webgpu" : null);
const threads = window.crossOriginIsolated === true;
const served = await fetch("./lucem-serve.json", { cache: "no-store" })
  .then((r) => (r.ok ? r.json() : {}))
  .catch(() => ({}));

if (!threads) {
  say(`<h1>This browser cannot run Lucem here</h1>
    <p>Lucem reads images on several threads, which a browser allows only to a page that is
    cross-origin isolated, and this one is not.</p>
    <p>Open it from <code>lucem serve --web</code>, in a recent Chrome, Edge or Firefox.</p>`);
} else if (!drawing && !params.has("software")) {
  say(`<h1>This browser draws without the graphics card</h1>
    <p>Lucem draws every frame on the GPU; in software it would be too slow to use.</p>
    <ul>
      <li>Chrome / Edge: Settings → System → <em>Use graphics acceleration when available</em>, then restart.</li>
      <li>Firefox: Settings → Performance → <em>Use recommended performance settings</em>.</li>
    </ul>
    <p><a href="?software">Start anyway (slow)</a></p>`);
} else {
  drawing ??= "webgl";
  // What the app reads at start (`lucem_app_web::backend`)
  window.lucemWeb = { drawing, server: served.server ?? null, webtransport: "lucemTransport" in window };
  say(`<p>Loading Lucem (${drawing === "webgpu" ? "WebGPU" : "WebGL2"})…</p>`);
  try {
    const app = await import("./mt/lucem_app_web.js");
    await app.default();
    note.remove();
  } catch (e) {
    // A Rust panic arrives here as "unreachable": the app's panic hook left what it was
    const why = window.lucemLastPanic ?? String(e);
    say(`<h1>Lucem did not start</h1><p><code>${why.replace(/</g, "&lt;")}</code></p>`);
  }
}
