# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the bundled Mirume backend.

Produces ``dist/mirume-backend/`` (a one-dir build: the ``mirume-backend``
launcher plus an ``_internal/`` folder of Python + native libraries). The
Tauri bundler copies that whole folder into
``Mirume.app/Contents/Resources/`` via ``bundle.resources`` in
``tauri.conf.json``; ``frontend/src-tauri/src/lib.rs`` spawns the launcher on
app start and kills it on exit.

One-dir rather than one-file: a one-file build unpacks its entire payload to a
temp directory on every launch (slow, and fragile with the MeCab / fastText
native extensions), whereas one-dir starts immediately and is straightforward
to inspect.

Build from the ``backend/`` directory with its virtualenv active:

    pyinstaller packaging/mirume-backend.spec --noconfirm --clean
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = []
binaries = []
hiddenimports = []

# fugashi auto-discovers the dictionary by importing `unidic_lite` and reading
# `unidic_lite.DICDIR` — from inside its compiled extension, which PyInstaller's
# static analysis can't see. collect_all pulls in the package's __init__.py
# (defines DICDIR) *and* its ~50 MB dicdir/ data. fasttext / fugashi likewise
# have native extensions that need their bundled libs.
for pkg in ("unidic_lite", "fugashi", "fasttext"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden
hiddenimports += ["unidic_lite", "fugashi"]

# uvicorn resolves its loop / protocol / lifespan implementations by string at
# runtime, so they never show up as imports.
hiddenimports += collect_submodules("uvicorn")

# pyobjc frameworks used by accessibility.py and ocr.py. The imported names are
# picked up from source, but list them explicitly so a build never silently
# drops the OCR path.
hiddenimports += [
    "objc",
    "Foundation",
    "AppKit",
    "Quartz",
    "CoreText",
    "ApplicationServices",
    "Vision",
    "CoreML",
]

# anthropic / deepl are optional at runtime but cheap to include, and pulling
# them in avoids a broken build if a user does configure a key.
hiddenimports += ["anthropic", "deepl"]


a = Analysis(
    ["../mirume_server.py"],
    pathex=[".."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Never needed at runtime; keep them out of the bundle.
        "tkinter",
        "watchfiles",
        "pytest",
        "IPython",
        # PaddleOCR was replaced by macOS Vision — make sure a stale install
        # in the build venv can't creep back in.
        "paddle",
        "paddleocr",
        "paddlex",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="mirume-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="mirume-backend",
)
