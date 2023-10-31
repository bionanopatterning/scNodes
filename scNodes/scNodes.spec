# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['__main__.py'],
    pathex=['../venv/lib/site-packages/glfw', '../venv/lib/site-packages/pygpufit', '../venv/lib/site-packages/', '../venv/lib/site-packages/pyGPUreg'],
    binaries=[],
    datas=[("./icons", "icons"), ("./shaders", "shaders"), ("./nodes", "nodes"), ("./ceplugins", "ceplugins"), ("./models", "models"),
    ("./core/particlefitting.py", "scNodes/core"), ("./nodes/pysofi", "scNodes/nodes/pysofi"),
    ("../venv/lib/site-packages/pygpufit/Gpufit.dll", "pygpufit"),
    ("../venv/lib/site-packages/pyGPUreg", "pyGPUreg"),
    ("./nodesetups", "nodesetups")],
    hiddenimports=['pywt', 'scNodes.core.particlefitting', 'pystackreg', 'cv2', 'scNodes.nodes.pysofi', 'pygpufit', 'pyGPUreg'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='scNodes',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
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
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='__main__',
)
