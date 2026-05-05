FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxcb-xinerama0 \
        libxcb1 \
        libx11-xcb1 \
        libxrender1 \
        libxext6 \
        libxi6 \
        libdbus-1-3 \
        libfontconfig1 \
        libfreetype6 \
        libx11-6 \
        x11-apps \
        xauth \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .

# Install CPU-only PyTorch first (keeps image smaller; GPU users override via compose)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# ── App source ────────────────────────────────────────────────────────────
COPY pollen_analysis_app.py .
COPY assets/ assets/

# ── Qt platform plugin for X11 forwarding ─────────────────────────────────
ENV QT_QPA_PLATFORM=xcb
ENV DISPLAY=:0

# Pre-download model weights into the image so first run is instant
RUN python - <<'EOF'
import os, sys
try:
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id="mouseland/cellpose-sam", filename="cpsam")
    print("Cellpose-SAM weights cached.")
except Exception as e:
    print(f"Weight pre-download skipped (will download on first run): {e}")
EOF

CMD ["python", "pollen_analysis_app.py"]
