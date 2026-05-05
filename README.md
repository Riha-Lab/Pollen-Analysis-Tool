# 🌸 Pollen Analysis Tool

**Automated pollen viability analysis powered by Cellpose-SAM deep learning.**
Built by [Riha Lab](https://github.com/Riha-Lab) for biologists — no coding required.

[![Build & Release](https://github.com/Riha-Lab/Pollen-Analysis-Tool/actions/workflows/build-release.yml/badge.svg)](https://github.com/Riha-Lab/Pollen-Analysis-Tool/actions/workflows/build-release.yml)
[![Docker](https://img.shields.io/docker/pulls/rihalab/pollen-analysis)](https://hub.docker.com/r/rihalab/pollen-analysis)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📥 Installation — choose your method

### Option A — Native desktop app (recommended for most users)

| Platform | Download |
|----------|----------|
| 🪟 **Windows** 10/11 (Intel or AMD) | [`PollenAnalysisTool-Setup-Windows-x64.exe`](https://github.com/Riha-Lab/Pollen-Analysis-Tool/releases/latest) |
| 🍎 **macOS** Apple Silicon (M1 / M2 / M3) | [`PollenAnalysisTool-macOS-arm64.dmg`](https://github.com/Riha-Lab/Pollen-Analysis-Tool/releases/latest) |
| 🍎 **macOS** Intel | [`PollenAnalysisTool-macOS-x86_64.dmg`](https://github.com/Riha-Lab/Pollen-Analysis-Tool/releases/latest) |

**Windows — step by step**
1. Download `PollenAnalysisTool-Setup-Windows-x64.exe` from the link above.
2. Double-click the installer → click **Next** → **Install**.
3. A shortcut appears on your desktop. Launch it and you're done.

> ⚠️ Windows may show a SmartScreen warning for unsigned software. Click **More info → Run anyway**.

**macOS — step by step**
1. Download the `.dmg` that matches your Mac (Apple Silicon or Intel).
2. Double-click the `.dmg` → drag **PollenAnalysisTool** into **Applications**.
3. First launch: right-click the app → **Open** → **Open** (bypasses Gatekeeper).

---

### Option B — Docker (Linux, macOS, Windows · CPU or GPU)

Docker lets you run the tool in an isolated container with a single command.
No Python installation needed.

#### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows / macOS) or Docker Engine (Linux)
- An X11 server to display the GUI:
  - **Linux** — already included
  - **macOS** — install [XQuartz](https://www.xquartz.org/) and enable *"Allow connections from network clients"* in Preferences → Security
  - **Windows** — install [VcXsrv](https://vcxsrv.sourceforge.io/) (check *Disable access control* when launching)

#### Start the app

```bash
# 1. Clone the repository (or just download docker-compose.yml)
git clone https://github.com/Riha-Lab/Pollen-Analysis-Tool.git
cd Pollen-Analysis-Tool

# 2. (macOS / Linux) Allow Docker to connect to your display
xhost +local:docker

# 3. Launch
docker compose up
```

The first run downloads the container image (~3 GB) and the Cellpose-SAM model weights.
Subsequent runs start in seconds.

**Your data** — place image files in `~/PollenData/` on your computer.
That folder is automatically mounted inside the container as `/data/`.

#### GPU acceleration (NVIDIA only)

```bash
docker compose --profile gpu up
```

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

#### Useful Docker commands

```bash
# Run in background
docker compose up -d

# Stop the container
docker compose down

# Update to the latest version
docker compose pull && docker compose up

# Remove downloaded weights cache (frees ~2 GB)
docker volume rm pollen-analysis-tool_pollen_weights
```

---

### Option C — Run from source (developers / advanced users)

```bash
# Requires Python 3.10 or 3.11
git clone https://github.com/Riha-Lab/Pollen-Analysis-Tool.git
cd Pollen-Analysis-Tool

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python PollenAnalysis_Trainer_PyQt6_v17.py
```

---

## 🖥️ Features

- **Automated segmentation** using Cellpose-SAM, Cellpose, or your own custom model
- **Viability classification** with Alexander stain colour channels
- **Batch processing** of entire image folders
- **Statistical analysis** — ANOVA / Kruskal-Wallis with Tukey post-hoc
- **PDF & CSV reports** generated automatically
- **Training mode** — annotate images and fine-tune a custom model

---

## 🗂️ Repo structure

```
Pollen-Analysis-Tool/
├── PollenAnalysis_Trainer_PyQt6_v17.py   # main application
├── requirements.txt                       # Python dependencies
├── PollenAnalysis.spec                    # PyInstaller build spec
├── Dockerfile                             # CPU Docker image
├── Dockerfile.gpu                         # GPU Docker image
├── docker-compose.yml                     # one-command launcher
├── assets/
│   ├── icon.ico                           # Windows icon
│   ├── icon.icns                          # macOS icon
│   ├── icon.png                           # Linux / Docker icon
│   └── dmg_background.png                 # macOS DMG background
├── installer/
│   ├── windows/installer.nsi              # NSIS Windows installer script
│   └── mac/
│       ├── build_dmg.sh                   # DMG creation script
│       └── entitlements.plist             # macOS code-signing entitlements
└── .github/
    └── workflows/
        └── build-release.yml              # CI/CD pipeline
```

---

## 🚀 Releasing a new version

```bash
# Tag a new version — GitHub Actions builds everything automatically
git tag v1.2.0
git push origin v1.2.0
```

GitHub Actions will:
1. Build native installers for Windows x64, macOS Intel, and macOS Apple Silicon
2. Build and push a multi-arch Docker image to Docker Hub
3. Create a GitHub Release with all files attached

---

## 🔧 GitHub Secrets required for CI/CD

Go to **Settings → Secrets and variables → Actions** and add:

| Secret | Description |
|--------|-------------|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token ([create here](https://hub.docker.com/settings/security)) |

Optional (for signed macOS builds):
| Secret | Description |
|--------|-------------|
| `CODESIGN_IDENTITY` | Apple Developer ID (e.g. `Developer ID Application: Your Name (TEAMID)`) |
| `APPLE_CERTIFICATE` | Base64-encoded `.p12` certificate |
| `APPLE_CERTIFICATE_PASSWORD` | Password for the `.p12` file |

---

## 📄 License

MIT © Riha Lab — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgements

- [Cellpose](https://github.com/MouseLand/cellpose) — cell segmentation framework
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) — GUI framework
- [PyInstaller](https://pyinstaller.org/) — cross-platform packaging
