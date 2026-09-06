#!/usr/bin/env bash
#
# Build the distributable Mirume .dmg.
#
#   1. Freeze the Python backend with PyInstaller  -> backend/dist/mirume-backend/
#   2. Stage the bundled seed data                 -> backend/packaging/backend-data/
#   3. Build + ad-hoc-sign the Tauri app and .dmg  -> frontend/src-tauri/target/release/bundle/
#
# Requires: the backend virtualenv (backend/venv) with requirements.txt +
# pyinstaller installed, Node modules installed in frontend/, and a Rust
# toolchain. macOS only.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="$REPO_ROOT/backend"
FRONTEND="$REPO_ROOT/frontend"
VENV_PY="$BACKEND/venv/bin/python"
SEED_DIR="$BACKEND/packaging/backend-data"

echo "==> 1/3  Freezing the backend (PyInstaller)"
cd "$BACKEND"
rm -rf build dist
"$VENV_PY" -m PyInstaller packaging/mirume-backend.spec --noconfirm --clean
test -x dist/mirume-backend/mirume-backend

echo "==> 2/3  Staging bundled seed data"
mkdir -p "$SEED_DIR"
if [ ! -f "$SEED_DIR/jmdict.db.gz" ] || [ "$BACKEND/../data/jmdict.db" -nt "$SEED_DIR/jmdict.db.gz" ]; then
  echo "    gzip data/jmdict.db -> $SEED_DIR/jmdict.db.gz"
  gzip -c -6 "$REPO_ROOT/data/jmdict.db" > "$SEED_DIR/jmdict.db.gz"
fi
cp "$REPO_ROOT/models/lid.176.ftz" "$SEED_DIR/lid.176.ftz"
ls -lh "$SEED_DIR"

echo "==> 3a/3  Ad-hoc signing the frozen backend"
# Tauri's `signingIdentity: "-"` signs the app wrapper + main binary but not the
# nested backend resources. Sign every Mach-O in the one-dir build so macOS
# doesn't stall validating unsigned dylibs on first launch.
find "$BACKEND/dist/mirume-backend" \( -name '*.so' -o -name '*.dylib' \) -print0 \
  | xargs -0 -n1 codesign --force --sign - --timestamp=none 2>/dev/null || true
codesign --force --sign - --timestamp=none "$BACKEND/dist/mirume-backend/mirume-backend"

echo "==> 3b/3  Building the Tauri app + .dmg"
cd "$FRONTEND"
# CI=true makes Tauri's bundle_dmg.sh skip the Finder/AppleScript step that
# arranges the disk-image window — that step hangs indefinitely in a
# non-interactive shell (no Finder automation). The .dmg is produced without
# the custom icon layout, which is cosmetic only.
CI=true npm run tauri build -- --bundles dmg

echo
echo "Done. Artifacts:"
find "$FRONTEND/src-tauri/target/release/bundle" -maxdepth 2 -name '*.dmg' -o -maxdepth 2 -name '*.app' | sed 's/^/  /'
