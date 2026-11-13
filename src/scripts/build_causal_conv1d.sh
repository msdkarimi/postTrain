#!/usr/bin/env bash
# build_causal_conv1d.sh - compile causal-conv1d ONCE, keep the wheel, reuse it on every later VM in seconds.
#
# Why: the project publishes no prebuilt wheel for torch 2.11, so `pip install causal-conv1d` compiles from source,
# and its setup.py compiles for SIX GPU architectures (sm_75/80/87/90/100/120 with CUDA 12.8). You only need your own
# GPU's architecture (sm_80 on an A100), so this script patches setup.py to build just that one.
#
# Usage (on a VM where bootstrap.sh has already run):
#     bash src/scripts/build_causal_conv1d.sh
# then copy the wheel folder to your own machine so later VMs can reuse it:
#     scp -r ubuntu@<vm-address>:~/wheels data/
# and on a new VM, before bootstrap.sh:
#     scp -r data/wheels ubuntu@<vm-address>:~/
# bootstrap.sh installs a matching cached wheel automatically (it is keyed by torch version, CUDA and GPU arch).
#
# The wheel only works with the SAME torch version it was built against (a compiled extension): keep torch pinned.
# Env: VENV_DIR, WHEEL_DIR (default ~/wheels), CAUSAL_CONV1D_VERSION (default 1.7.0), ARCH (default: this GPU, e.g. 80),
#      MAX_JOBS (default: number of CPU cores)
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/venv}"
WHEEL_DIR="${WHEEL_DIR:-$HOME/wheels}"
VERSION="${CAUSAL_CONV1D_VERSION:-1.7.0}"
export PATH="$HOME/.local/bin:$PATH"
source "$VENV_DIR/bin/activate"

ARCH="${ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')}"
TORCH_VER="$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
CUDA_VER="$(python -c 'import torch; print(torch.version.cuda)')"
TAG="torch${TORCH_VER}_cu${CUDA_VER}_sm${ARCH}"
OUT="$WHEEL_DIR/$TAG"

if ls "$OUT"/causal_conv1d-*.whl >/dev/null 2>&1; then
  echo "already built: $(ls "$OUT"/causal_conv1d-*.whl)"; exit 0
fi

if [ -z "${CUDA_HOME:-}" ] && ! command -v nvcc >/dev/null 2>&1 && [ -d /usr/local/cuda ]; then
  export CUDA_HOME=/usr/local/cuda; export PATH="$CUDA_HOME/bin:$PATH"
fi
command -v nvcc >/dev/null 2>&1 || { echo "ERROR: nvcc (CUDA toolkit) not found; cannot compile"; exit 1; }

echo "==> building causal-conv1d $VERSION for $TAG (sm_$ARCH only), nvcc: $(nvcc --version | tail -1)"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
URL="$(curl -fsSL "https://pypi.org/pypi/causal-conv1d/$VERSION/json" | python -c \
  "import sys,json; d=json.load(sys.stdin); print(next(u['url'] for u in d['urls'] if u['packagetype']=='sdist'))")"
curl -fsSL "$URL" -o "$WORK/src.tar.gz"
tar -xzf "$WORK/src.tar.gz" -C "$WORK"
cd "$WORK"/causal_conv1d-*/

# Patch: after the hard-coded -gencode list, keep only the architecture we need.
python - <<'EOF'
import re, sys
p = "setup.py"
lines = open(p).read().split("\n")
marker = [i for i, l in enumerate(lines) if "arch=compute_121,code=sm_121" in l]
if not marker:
    sys.exit("setup.py layout changed: cannot find the -gencode block; build without the arch patch (slow) or edit this script")
i = marker[-1]
patch = '''        _only = os.environ.get("CAUSAL_CONV1D_ONLY_ARCH")   # added by build_causal_conv1d.sh
        if _only:
            _kept, _k = [], 0
            while _k < len(cc_flag):
                if cc_flag[_k] == "-gencode":
                    if f"sm_{_only}" in cc_flag[_k + 1]:
                        _kept += cc_flag[_k:_k + 2]
                    _k += 2
                else:
                    _kept.append(cc_flag[_k]); _k += 1
            cc_flag = _kept
            print(f"[patched] compiling only sm_{_only}: {cc_flag}")'''.split("\n")
lines[i + 1:i + 1] = patch
open(p, "w").write("\n".join(lines))
EOF

uv pip install -q packaging ninja wheel setuptools
mkdir -p "$OUT"
CAUSAL_CONV1D_FORCE_BUILD=TRUE CAUSAL_CONV1D_ONLY_ARCH="$ARCH" MAX_JOBS="${MAX_JOBS:-$(nproc)}" \
  python setup.py bdist_wheel -d "$OUT"
echo "==> wheel saved: $(ls "$OUT"/*.whl)"
uv pip install --no-deps --force-reinstall "$OUT"/causal_conv1d-*.whl
python -c "import causal_conv1d; print('causal_conv1d imports OK:', causal_conv1d.__version__)"
echo "Now copy $WHEEL_DIR to your own machine (see the header of this script) so you never compile it again."
