#!/usr/bin/env bash
# Builds the page into app/lucem-app-web/dist/ (served as it is: `lucem serve --web <dist>`):
#
#   index.html     the loader: checks the browser (hardware drawing, WebGPU, threads) before
#                  anything large is downloaded, holds the WebTransport shim, starts the app
#   mt/            the app, threaded (tile reads wait for the server on web workers); needs the
#                  page cross-origin isolated (`lucem serve --web` sends the headers)
#   isolate-sw.js  the service worker that isolates the page where the host cannot
#
# Each .wasm and .js is also written Brotli-compressed (.br), which `lucem serve` sends to
# browsers that accept it. As dsp-kitchen's app/dspk-app-web/build.sh.
#
# Needs: the pinned nightly below with rust-src and the wasm32-unknown-unknown target (GPUI's web
# platform needs nightly, and threads need -Z build-std), wasm-bindgen-cli of the version in
# Cargo.lock, wasm-opt (binaryen) and brotli.
set -euo pipefail

# Pinned so a build can be repeated (the same as dsp-kitchen's: one toolchain for both)
TOOLCHAIN="${LUCEM_WEB_TOOLCHAIN:-nightly-2026-11-14}"

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
dist="$here/dist"
target="${CARGO_TARGET_DIR:-$root/target}"

bindgen_version=$(awk '/^name = "wasm-bindgen"$/ { getline; gsub(/version = |"/, ""); print; exit }' "$root/Cargo.lock")
bindgen=$(command -v wasm-bindgen || true)
if [ -z "$bindgen" ] || ! "$bindgen" --version | grep -q " $bindgen_version\$"; then
    bindgen=$(ls ~/.cache/.wasm-pack/wasm-bindgen-*/wasm-bindgen 2>/dev/null | while read -r b; do "$b" --version | grep -q " $bindgen_version\$" && echo "$b" && break; done || true)
fi
[ -n "$bindgen" ] || { echo "wasm-bindgen $bindgen_version not found: cargo binstall wasm-bindgen-cli --version $bindgen_version"; exit 1; }
wasm_opt=$(command -v wasm-opt || ls -t ~/.cache/.wasm-pack/wasm-opt-*/bin/wasm-opt 2>/dev/null | head -1 || true)
[ -n "$wasm_opt" ] || { echo "wasm-opt not found (binaryen)"; exit 1; }
command -v brotli > /dev/null || { echo "brotli not found"; exit 1; }
rustup run "$TOOLCHAIN" rustc --version > /dev/null 2>&1 || rustup toolchain install "$TOOLCHAIN" --profile minimal --component rust-src --target wasm32-unknown-unknown

mkdir -p "$dist"
cp "$here/index.html" "$here/loader.js" "$here/isolate-sw.js" "$dist/"
# A page on its own says so; `lucem serve` answers this address itself
echo '{"server": null}' > "$dist/lucem-serve.json"

# Shared memory and atomics, and a standard library built with them (-Z build-std); the
# exports wasm-bindgen's thread support needs
export CARGO_TARGET_DIR="$target/web-mt"
(cd "$root" && RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128' \
    rustup run "$TOOLCHAIN" cargo rustc -p lucem-app-web --lib --target wasm32-unknown-unknown --release \
    -Z build-std=std,panic_abort -- \
    -C link-arg=--shared-memory -C link-arg=--import-memory -C link-arg=--max-memory=4294967296 \
    -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base)
wasm="$CARGO_TARGET_DIR/wasm32-unknown-unknown/release/lucem_app_web.wasm"
rm -rf "$dist/mt"
"$bindgen" "$wasm" --out-dir "$dist/mt" --target web
"$wasm_opt" -O2 --enable-threads --enable-bulk-memory --enable-mutable-globals --enable-simd --enable-nontrapping-float-to-int --enable-sign-ext \
    "$dist/mt/lucem_app_web_bg.wasm" -o "$dist/mt/lucem_app_web_bg.wasm"
du -h "$dist/mt/lucem_app_web_bg.wasm" | sed 's/^/   /'
find "$dist" \( -name '*.wasm' -o -name '*.js' \) | while read -r f; do brotli -f -q 11 "$f" -o "$f.br"; done
echo "Built into $dist (serve it: lucem serve --data <images> --web $dist)"
