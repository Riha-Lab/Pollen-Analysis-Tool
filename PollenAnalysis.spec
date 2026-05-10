# -*- mode: python ; coding: utf-8 -*-
# PollenAnalysis.spec — used by PyInstaller for all platforms

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect data files for packages with non-Python assets
cellpose_datas    = collect_data_files('cellpose')
scipy_datas       = collect_data_files('scipy')
statsmodels_datas = collect_data_files('statsmodels')
torch_datas       = collect_data_files('torch')
matplotlib_datas  = collect_data_files('matplotlib')

# Collect ALL matplotlib backends as hidden imports.
# savefig() lazily imports the backend that matches the file extension at
# runtime (e.g. backend_pdf for .pdf, backend_svg for .svg).  PyInstaller
# cannot see these dynamic imports, so we collect every backend explicitly.
matplotlib_backends = collect_submodules('matplotlib.backends')

# Platform-specific icon
if sys.platform == 'darwin':
    icon_file = os.path.join('assets', 'icon.icns')
elif sys.platform == 'win32':
    icon_file = os.path.join('assets', 'icon.ico')
else:
    icon_file = os.path.join('assets', 'icon.png')

a = Analysis(
    ['pollen_analysis_app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('assets', 'assets'),
        *cellpose_datas,
        *scipy_datas,
        *statsmodels_datas,
        *torch_datas,
        *matplotlib_datas,
    ],
    hiddenimports=[
        # PyQt6
        'PyQt6.sip',
        'PyQt6.QtPrintSupport',
        # matplotlib — collect_submodules covers every backend (pdf, svg, agg,
        # qtagg, etc.) so savefig() never hits a missing-module error at runtime
        *matplotlib_backends,
        'matplotlib.figure',
        'matplotlib.font_manager',
        'matplotlib.ticker',
        'matplotlib.patches',
        'matplotlib.colors',
        'matplotlib.transforms',
        'matplotlib.artist',
        'matplotlib.axis',
        'matplotlib.scale',
        # cellpose / torch
        'cellpose',
        'cellpose.models',
        'cellpose.io',
        'torch',
        'torch.nn',
        'torchvision',
        # opencv
        'cv2',
        # PIL / Pillow
        'PIL',
        'PIL.Image',
        # pandas
        'pandas',
        'pandas._libs.tslibs.timedeltas',
        # scipy / stats
        'scipy.stats',
        'scipy.special',
        'scipy.special._ufuncs',
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
        # stdlib used at runtime
        'requests',
        'multiprocessing',
        'concurrent.futures',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

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
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
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
    # Don't UPX-compress Qt/torch binaries — causes crashes on Windows
    upx_exclude=[
        'vcruntime140.dll',
        'python*.dll',
        'Qt6*.dll',
        'torch_python*.pyd',
        '_C*.pyd',
    ],
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
