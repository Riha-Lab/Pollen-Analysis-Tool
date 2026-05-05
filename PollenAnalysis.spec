# -*- mode: python ; coding: utf-8 -*-
# PollenAnalysis.spec — used by PyInstaller for all platforms

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect cellpose data files (model configs, etc.)
cellpose_datas = collect_data_files('cellpose')
scipy_datas    = collect_data_files('scipy')
statsmodels_datas = collect_data_files('statsmodels')

# Platform-specific icon
if sys.platform == 'darwin':
    icon_file = os.path.join('assets', 'icon.icns')
elif sys.platform == 'win32':
    icon_file = os.path.join('assets', 'icon.ico')
else:
    icon_file = os.path.join('assets', 'icon.png')

a = Analysis(
    ['PollenAnalysis_Trainer_PyQt6_v17.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('assets', 'assets'),
        *cellpose_datas,
        *scipy_datas,
        *statsmodels_datas,
    ],
    hiddenimports=[
        # PyQt6
        'PyQt6.sip',
        'PyQt6.QtPrintSupport',
        # matplotlib
        'matplotlib.backends.backend_qtagg',
        'matplotlib.backends.backend_agg',
        # cellpose / torch
        'cellpose',
        'cellpose.models',
        'cellpose.io',
        'torch',
        'torch.nn',
        'torchvision',
        # scipy / stats
        'scipy.stats',
        'scipy.special',
        'scipy.linalg',
        'scipy._lib.messagestream',
        # statsmodels
        'statsmodels.stats.multicomp',
        # huggingface
        'huggingface_hub',
        # reportlab
        'reportlab.platypus',
        'reportlab.lib.pagesizes',
        'reportlab.lib.styles',
        'reportlab.lib.units',
        'reportlab.graphics',
    ],
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
    name='PollenAnalysisTool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,       # no terminal window
    disable_windowed_traceback=False,
    target_arch=None,    # let CI set this per matrix job
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PollenAnalysisTool',
)

# macOS .app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='PollenAnalysisTool.app',
        icon=icon_file,
        bundle_identifier='com.rihalab.pollenanalysis',
        info_plist={
            'NSPrincipalClass': 'NSApplication',
            'NSHighResolutionCapable': True,
            'CFBundleShortVersionString': '1.0.0',
            'CFBundleName': 'Pollen Analysis Tool',
            'NSHumanReadableCopyright': '© Riha Lab',
        },
    )
