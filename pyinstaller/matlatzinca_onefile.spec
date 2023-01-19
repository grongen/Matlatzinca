# -*- mode: python ; coding: utf-8 -*-
import os
import sys
import shutil
from pathlib import Path


block_cipher = None

# Import list of binaries to exclude
with open("binaries_to_remove.txt", "r") as f:
    binaries_to_remove = [file.lower() for file in f.read().split("\n") if not file.startswith("#")]

# Collect documentation
docpaths = []
docbase = "../doc/build/html"
for root, dirs, files in os.walk(docbase):
    docpaths += [os.path.join(root, file) for file in files]
docdata = [(path, os.path.split(path.replace(docbase, "doc"))[0]) for path in docpaths]

# Add icon
docdata.append(("../matlatzinca/data/icon.ico", "data"))
docdata.append(("../matlatzinca/data/cond_prob_rho.npy", "data"))

# Find qwindows.dll
for path in sys.path:
    path = Path(path)
    possible_path1 = path / "Lib/site-packages/PyQt5/Qt5/plugins/platforms/qwindows.dll"
    if possible_path1.exists():
        print('Found at' , possible_path1)
        qwindowsdll = possible_path1
        break
    
    possible_path2 = path / "PyQt5/Qt5/plugins/platforms/qwindows.dll"
    if possible_path2.exists():
        print('Found at' , possible_path2)
        qwindowsdll = possible_path2
        break

else:
    raise OSError("Could not find qwindows.dll")

# Analysis class
a = Analysis(
    ["../matlatzinca/__main__.py"],
    pathex=[],
    binaries=[(qwindowsdll, "platforms")],
    datas=docdata,
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Remove binaries
to_remove = []
for i, (file, _, cat) in enumerate(a.binaries):
    if (file.strip().lower() in binaries_to_remove) or (file.strip().lower().split("\\")[-1] in binaries_to_remove):
        to_remove.append(i)
for i in reversed(to_remove):
    del a.binaries[i]

# Remove data to exclude
to_remove = []
for i, (path, _, _) in enumerate(a.datas):
    if ("PyQt5\\qt\\qml" in path.lower()) or ("PyQt5\\qt5\\qml" in path.lower()):
        to_remove.append(i)
    elif ("PyQt5\\translations" in path.lower()) or ("PyQt5\\qt5\\translations" in path.lower()):
        to_remove.append(i)
    elif "PyQt5\\qt\\bin" in path.lower():
        to_remove.append(i)
    elif "mpl-data\\sample_data" in path.lower():
        to_remove.append(i)
    elif "mpl-data\\fonts" in path.lower() and ("DejaVuSans.ttf" not in path):
        to_remove.append(i)

for i in reversed(to_remove):
    del a.datas[i]

# Create archive
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Create executable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="Matlatzinca",
    debug=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    icon="../matlatzinca/data/icon.ico",
)
