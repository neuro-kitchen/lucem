#!/usr/bin/env bash
# Smoke test of the lucem server image, end to end, with nothing on the host but Podman: the image
# is the server and the client.
#
#   build → doctor → data → serve → enroll → fetch (and refusals) → run a pipeline → stop
#
# Usage (from the repository root):
#   deploy/smoke-test.sh                 # target picked from the host: nvidia (CDI), mesa (/dev/dri), cpu
#   deploy/smoke-test.sh --target cpu    # or nvidia, mesa
#   deploy/smoke-test.sh --no-build      # test the image already built
#
# Settings (environment): PORT=50079, WORK=$XDG_CACHE_HOME/lucem-smoke-test
# Exits non-zero at the first failure; its containers are always removed.
set -euo pipefail

TARGET=""
BUILD=1
while [ $# -gt 0 ]; do
    case "$1" in
        --target) TARGET="$2"; shift 2 ;;
        --no-build) BUILD=0; shift ;;
        -h | --help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
PORT="${PORT:-50079}"
WORK="${WORK:-${XDG_CACHE_HOME:-$HOME/.cache}/lucem-smoke-test}"
NAME="lucem-smoke-$$"
# The working folder is removed and recreated: refuse anything not named for this test
case "$(basename "$WORK")" in *lucem-smoke*) ;; *) echo "WORK must be a folder named *lucem-smoke*: $WORK" >&2; exit 2 ;; esac

if [ -z "$TARGET" ]; then
    if [ -e /etc/cdi/nvidia.yaml ] || [ -e /var/run/cdi/nvidia.yaml ]; then TARGET=nvidia
    elif [ -e /dev/dri ]; then TARGET=mesa
    else TARGET=cpu; fi
fi
case "$TARGET" in
    nvidia) DEVICE=(--device nvidia.com/gpu=all --security-opt=label=disable) ;;
    mesa) DEVICE=(--device /dev/dri --group-add keep-groups --security-opt=label=disable) ;;
    cpu) DEVICE=() ;;
    *) echo "unknown target: $TARGET" >&2; exit 2 ;;
esac
IMAGE="lucem-serve:$TARGET"
step() { echo "--- $*"; }
pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*" >&2; exit 1; }
cleanup() { podman rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

if [ "$BUILD" = 1 ]; then
    step "build $IMAGE"
    podman build -f deploy/Containerfile --ignorefile deploy/.containerignore --target "$TARGET" -t "$IMAGE" .
fi
rm -rf "$WORK"
mkdir -p "$WORK"
# The data folder is shared by several containers (the server, its clients): `:z`, a label they
# all may use. `:Z` (private) would relabel it for each new container and lock the server out of
# its own client registry.
RUN=(podman run --rm --userns=keep-id:uid=10001,gid=10001 "${DEVICE[@]}" -v "$WORK:/data:z" -v lucem-smoke-cache:/home/lucem/.cache "$IMAGE")

step "doctor"
"${RUN[@]}" doctor && pass "doctor" || fail "doctor found no usable runtime"

step "data"
"${RUN[@]}" sample /data/image.ome.zarr --size 256 || fail "sample"
pass "synthetic image"

step "serve (hardened)"
podman run -d --name "$NAME" -p "$PORT:50051/udp" --userns=keep-id:uid=10001,gid=10001 "${DEVICE[@]}" \
    --read-only --cap-drop=all --security-opt=no-new-privileges --pids-limit=1024 \
    -v "$WORK:/data:z" -v lucem-smoke-cache:/home/lucem/.cache "$IMAGE" serve --bind 0.0.0.0:50051 >/dev/null
sleep 3
podman ps --filter "name=$NAME" --format '{{.Status}}' | grep -q Up || { podman logs "$NAME"; fail "the server did not start"; }
pass "serving"

step "enroll"
(umask 077; podman exec "$NAME" lucem clients enroll alice --out - > "$WORK/alice.pem")
[ -s "$WORK/alice.pem" ] || fail "enroll"
pass "enrolled"

FETCH=(podman run --rm --network=host --userns=keep-id:uid=10001,gid=10001 -v "$WORK:/data:z" "$IMAGE" fetch "127.0.0.1:$PORT" --server-name localhost --identity /data/alice.pem)
step "fetch"
"${FETCH[@]}" | grep -q image.ome.zarr && pass "listed" || fail "the image is not listed"
"${FETCH[@]}" --open image.ome.zarr | grep -q "tile 0,0" && pass "read a tile" || fail "tile"

step "refusals"
if "${FETCH[@]}" --open ../etc/passwd >/dev/null 2>&1; then fail "a path out of the data folder was served"; fi
pass "a path out of /data is refused"
if "${FETCH[@]}" --open .lucem-auth/ca.key.pem >/dev/null 2>&1; then fail "the authority's key was served"; fi
pass "the authority's key is refused"
podman exec "$NAME" lucem clients revoke alice
if "${FETCH[@]}" >/dev/null 2>&1; then fail "a revoked client was served"; fi
pass "a revoked client is refused"

step "run a pipeline"
cat > "$WORK/spec.json" <<'SPEC'
{"sources": {"image": 0},
 "nodes": {"blur": {"op": {"op": "gaussian_blur", "sigma": 1.0}, "inputs": ["image"]},
           "mask": {"op": {"op": "threshold", "value": 0.3}, "inputs": ["blur"]},
           "objects": {"op": {"op": "label"}, "inputs": ["mask"]}},
 "target": "objects"}
SPEC
"${RUN[@]}" run /data/spec.json /data/image.ome.zarr /data/objects.ome.zarr | tee "$WORK/run.log"
grep -q objects "$WORK/run.log" && [ -f "$WORK/objects.ome.zarr.spec.json" ] && pass "pipeline and its spec" || fail "pipeline"

step "stop"
podman stop -t 10 "$NAME" >/dev/null && pass "stopped cleanly"
rm -rf "$WORK"
echo "All passed ($TARGET)."
