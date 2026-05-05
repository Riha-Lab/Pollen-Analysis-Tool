import sys
import io
import os

# 1. Create dummy streams so libraries like tqdm/cellpose 
# don't crash when sys.stdout/stderr are None in windowed mode.
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

# 2. Tell Cellpose NOT to check for/download models at startup.
# This forces it to wait until you manually provide a path in the GUI.
os.environ["CELLPOSE_LOCAL_MODELS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

# Now your original imports...
import re
import warnings
import re
import warnings
import sys
import re
import warnings
import traceback
import numpy as np
import cv2
from matplotlib.figure import Figure as _MplFigure
from matplotlib.backends.backend_agg import FigureCanvasAgg as _FigureCanvasAgg
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QGraphicsView, QGraphicsScene,
    QSplitter, QTextEdit, QComboBox, QLineEdit,
    QRadioButton, QButtonGroup, QProgressBar, QCheckBox,
    QSpinBox, QDoubleSpinBox, QListWidget, QListWidgetItem, QTableWidget,
    QTableWidgetItem, QHeaderView, QSlider, QAbstractItemView,
    QFrame, QScrollArea, QStackedWidget, QInputDialog
)
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QPolygonF
from PyQt6.QtCore import Qt, QRectF, QThread, QTimer, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FixedLocator, MaxNLocator
import matplotlib.font_manager as _fm
from cellpose import models
try:
    from cellpose.io import imsave as _imsave
except ImportError:
    _imsave = None
from scipy.stats import shapiro, kruskal, f_oneway, mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd
import os
from datetime import datetime
import string
from collections import defaultdict, OrderedDict
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Table, TableStyle, Image as RLImage,
                                PageBreak, HRFlowable, KeepTogether)
from reportlab.lib.units import cm
import io
import multiprocessing
import concurrent.futures
from PIL import Image
import torch
from huggingface_hub import hf_hub_download
import requests

# Channel extraction for Alexander stain
CHANNEL_OPTIONS = {
    "RGB (trained on colour images)": lambda img: img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB),
    "Red Channel (viable pollen)": lambda img: img[:, :, 0] if img.ndim == 3 else img,
    "Red minus Green (max contrast)": lambda img: np.clip(
        img[:, :, 0].astype(np.int16) - img[:, :, 1].astype(np.int16), 0, 255
    ).astype(np.uint8) if img.ndim == 3 else img,
    "Grayscale": lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img,
}
DEFAULT_CHANNEL = "RGB (trained on colour images)"

# Model weights download functions
def download_weights_cpsam():
    try:
        return hf_hub_download(repo_id="mouseland/cellpose-sam", filename="cpsam")
    except Exception as e:
        print(f"Error downloading Cellpose-SAM weights: {e}")
        raise

def download_weights_cpsam_old():
    import tempfile
    cache_dir = os.path.join(tempfile.gettempdir(), "cellpose_weights")
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, "cpsam")
    url = "https://osf.io/d7c8e/download"
    if not os.path.isfile(fname):
        for ntries in range(10):
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == requests.codes.ok:
                    with open(fname, "wb") as fid:
                        fid.write(r.content)
                    return fname
                else:
                    print(f"!!! HTTP {r.status_code} downloading Cellpose-SAM weights (attempt {ntries+1}/10) !!!")
            except Exception as e:
                print(f"!!! Failed to download Cellpose-SAM weights: {e} (attempt {ntries+1}/10) !!!")
        raise Exception("Failed to download Cellpose-SAM weights after 10 attempts")
    return fname

# PyTorch 2.6: patch torch.load globally at startup so that model weights
# saved without weights_only=True can still be loaded. The patch is removed
# after model loading is complete.
_orig_torch_load = torch.load
def _permissive_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)  # noqa: F821 – deleted after use below
torch.load = _permissive_load

# Load models
try:
    cpsam_path = download_weights_cpsam()
    model_cpsam = models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model=cpsam_path)
except Exception as e:
    print(f"Error loading Cellpose-SAM model: {e}")
    try:
        print("Attempting fallback download for Cellpose-SAM...")
        cpsam_path = download_weights_cpsam_old()
        model_cpsam = models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model=cpsam_path)
    except Exception as e:
        print(f"Cellpose-SAM fallback failed: {e}")
        model_cpsam = None

model_cellpose = models.CellposeModel(gpu=torch.cuda.is_available())

torch.load = _orig_torch_load
del _orig_torch_load, _permissive_load

# Custom model registry
model_custom = None
model_custom_name = ""

# ---------------------------------------------------------------------------
# Utility functions ported from V5
# ---------------------------------------------------------------------------
def _compute_min_grain_area(diameter_setting, masks):
    if diameter_setting and diameter_setting > 0:
        return max(1, int(np.pi * (diameter_setting / 2) ** 2 * 0.25))
    ids = np.unique(masks)
    ids = ids[ids != 0]
    if len(ids) == 0:
        return 1
    areas = np.array([np.sum(masks == n) for n in ids], dtype=np.float32)
    diameters = 2.0 * np.sqrt(areas / np.pi)
    med_d = float(np.median(diameters))
    return max(1, int(np.pi * (med_d / 2) ** 2 * 0.25))

def image_resize(img, resize=1000):
    ny, nx = img.shape[:2]
    if max(ny, nx) > resize:
        if ny > nx:
            nx_new = int(nx / ny * resize)
            ny_new = resize
        else:
            ny_new = int(ny / nx * resize)
            nx_new = resize
        img_resized = cv2.resize(img, (nx_new, ny_new), interpolation=cv2.INTER_AREA)
        return img_resized.astype(np.uint8)
    return img.astype(np.uint8)

def normalize99(img, lower_p=1.0, upper_p=99.0):
    if img.ndim == 3:
        out = np.zeros_like(img, dtype=np.float32)
        for c in range(img.shape[2]):
            ch = img[:, :, c].astype(np.float32)
            p_lo, p_hi = np.percentile(ch, [lower_p, upper_p])
            out[:, :, c] = (ch - p_lo) / (1e-10 + p_hi - p_lo)
        return out
    X = img.astype(np.float32)
    p_lo, p_hi = np.percentile(X, [lower_p, upper_p])
    return (X - p_lo) / (1e-10 + p_hi - p_lo)

def plot_overlay(img, masks, alpha=0.5):
    try:
        out = img.copy().astype(np.float32)
        GOLDEN = 0.6180339887          
        n_cells = int(masks.max())
        hues = np.mod(np.arange(1, n_cells + 1) * GOLDEN, 1.0)
        
        for idx in range(n_cells):
            cell_id = idx + 1
            pix = masks == cell_id
            if not pix.any():
                continue
            # Overlay the assigned hue directly onto the original RGB image
            c = np.array(hsv_to_rgb((hues[idx], 0.9, 0.9))) * 255.0
            out[pix] = out[pix] * (1.0 - alpha) + c * alpha
            
        return Image.fromarray(out.astype(np.uint8))
    except Exception as e:
        print(f"Error in plot_overlay: {e}")
        return Image.fromarray(img.astype(np.uint8))

def plot_outlines(img, masks):
    try:
        img_n = np.clip(normalize99(img), 0, 1)
        contours, _ = cv2.findContours(
            masks.astype(np.int32), mode=cv2.RETR_FLOODFILL,
            method=cv2.CHAIN_APPROX_SIMPLE
        )
        outpix = []
        for c in contours:
            pix = c.astype(int).squeeze()
            if len(pix) > 4:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.001 * peri, True)[:, 0, :]
                outpix.append(approx)
        ar = img_n.shape[0] / img_n.shape[1]
        fig = plt.figure(figsize=(6, 6 * ar) if ar >= 1 else (6 / ar, 6), facecolor='k')
        ax = fig.add_subplot(111)
        ax.set_xlim([0, img_n.shape[1]])
        ax.set_ylim([0, img_n.shape[0]])
        ax.imshow(img_n[::-1], origin='upper', aspect='auto')
        for o in outpix:
            ax.plot(o[:, 0], img_n.shape[0] - o[:, 1], color=[1, 0, 0], lw=1)
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches='tight')
        buf.seek(0)
        pil_img = Image.open(buf).copy()
        plt.close(fig)
        return pil_img
    except Exception as e:
        print(f"Error in plot_outlines: {e}")
        return Image.fromarray(np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8))

def _pil_to_bytesio(pil_img):
    buf = io.BytesIO()
    if pil_img.mode in ("RGBA", "P"):
        pil_img = pil_img.convert("RGB")
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Parallel overlay/outline worker — must be a module-level function so it is
# picklable by ProcessPoolExecutor (lambdas and nested functions are not).
# Returns (index, orig_pil, overlay_pil, outline_pil) so results can be
# re-ordered after parallel collection.
# ---------------------------------------------------------------------------
def _render_overlays_worker(args):
    """Called in a worker process. args = (index, img_np, mask_np)."""
    idx, img_np, mask_np = args
    overlay_pil = plot_overlay(img_np, mask_np)
    outline_pil = plot_outlines(img_np, mask_np)
    orig_pil    = Image.fromarray(img_np)
    return idx, orig_pil, overlay_pil, outline_pil

def generate_pdf_report(image_data, output_path, stat_report="", boxplot_path=None, df_counts=None, run_settings=None, sample_order=None):
    PAGE_W, PAGE_H = A4
    MARGIN    = 1.8 * cm
    CONTENT_W = PAGE_W - 2 * MARGIN

    styles    = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=18, leading=24, spaceAfter=4, textColor=colors.HexColor("#1a3a5c"))
    h1_style = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=13, leading=16, textColor=colors.HexColor("#1a3a5c"), spaceBefore=10, spaceAfter=3)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=10, leading=13, textColor=colors.HexColor("#2c5f8a"), spaceBefore=6, spaceAfter=2)
    caption_style = ParagraphStyle("Caption", parent=styles["Normal"], fontSize=7.5, leading=9.5, textColor=colors.HexColor("#555555"))
    mono_style = ParagraphStyle("Mono", parent=styles["Code"], fontSize=7.5, leading=10, fontName="Courier", textColor=colors.HexColor("#333333"))

    doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN)
    story = []
    now_str = datetime.now().strftime("%d %B %Y  %H:%M")

    story.append(Spacer(1, 1.2 * cm))
    story.append(Paragraph("Pollen Analysis Tool", title_style))
    story.append(Paragraph("Alexander Stain · Cellpose Segmentation", h2_style))
    story.append(Paragraph(f"Generated: {now_str}", caption_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a3a5c"), spaceAfter=10))

    sample_stats = defaultdict(list)
    for sample_name, _, _, _, _, count in image_data:
        sample_stats[sample_name].append(count)

    tbl_data = [["Sample", "n images", "Viable mean", "Viable SD", "Min", "Max"]]
    for sname, vals in sample_stats.items():
        arr = np.array(vals)
        tbl_data.append([sname, str(len(arr)), f"{arr.mean():.1f}", f"{arr.std():.1f}", str(arr.min()), str(arr.max())])

    col_widths = [CONTENT_W * w for w in [0.32, 0.12, 0.16, 0.14, 0.13, 0.13]]
    tbl = Table(tbl_data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",      (0, 0), (-1, 0),  colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR",       (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",        (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",        (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS",  (0, 1), (-1, -1), [colors.HexColor("#f0f4f8"), colors.white]),
        ("GRID",            (0, 0), (-1, -1), 0.4, colors.HexColor("#aaaaaa")),
        ("ALIGN",           (1, 0), (-1, -1), "CENTER"),
        ("VALIGN",          (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",      (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",   (0, 0), (-1, -1), 4),
    ]))
    story.append(Paragraph("Summary", h1_style))
    story.append(tbl)
    story.append(PageBreak())

    GAP = 0.2 * cm
    IMG_W = (CONTENT_W - 2 * GAP) / 3      
    IMG_H = IMG_W * (9 / 16)               
    COL_WIDTHS = [IMG_W + GAP/2, IMG_W + GAP/2, IMG_W]

    samples_ordered = list(OrderedDict.fromkeys(r[0] for r in image_data))
    by_sample = defaultdict(list)
    for row in image_data:
        by_sample[row[0]].append(row)

    for sample_name in samples_ordered:
        rows = by_sample[sample_name]
        heading_block = [
            Paragraph(f"Sample: {sample_name}", h1_style),
            HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#1a3a5c"), spaceAfter=5),
        ]
        col_header = Table(
            [[Paragraph("<b>Original</b>", caption_style), Paragraph("<b>Overlay</b>", caption_style), Paragraph("<b>Outlines</b>", caption_style)]],
            colWidths=COL_WIDTHS,
            style=[("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("BOTTOMPADDING", (0,0), (-1,-1), 2)]
        )

        image_rows = []
        for (sname, img_file, orig_img, overlay_img, outline_img, count) in rows:
            fname = os.path.basename(img_file)
            orig_rl = RLImage(_pil_to_bytesio(orig_img), width=IMG_W, height=IMG_H)
            overlay_rl = RLImage(_pil_to_bytesio(overlay_img), width=IMG_W, height=IMG_H)
            outline_rl = RLImage(_pil_to_bytesio(outline_img), width=IMG_W, height=IMG_H)

            img_row = Table(
                [[orig_rl, overlay_rl, outline_rl]], colWidths=COL_WIDTHS,
                style=[
                    ("ALIGN",          (0,0), (-1,-1), "CENTER"),
                    ("VALIGN",         (0,0), (-1,-1), "MIDDLE"),
                    ("LEFTPADDING",    (0,0), (-1,-1), 2),
                    ("RIGHTPADDING",   (0,0), (-1,-1), 2),
                    ("TOPPADDING",     (0,0), (-1,-1), 2),
                    ("BOTTOMPADDING",  (0,0), (-1,-1), 2),
                    ("BOX",            (0,0), (0,0), 0.8, colors.HexColor("#aaaaaa")),
                    ("BOX",            (1,0), (1,0), 0.8, colors.HexColor("#aaaaaa")),
                    ("BOX",            (2,0), (2,0), 0.8, colors.HexColor("#aaaaaa")),
                ]
            )
            label_row = Table(
                [[Paragraph(f"{fname}", caption_style), Paragraph("", caption_style), Paragraph(f"<b>Viable: {count}</b>", caption_style)]],
                colWidths=COL_WIDTHS,
                style=[("ALIGN", (0,0), (0,0), "LEFT"), ("ALIGN", (2,0), (2,0), "RIGHT"), ("TOPPADDING", (0,0), (-1,-1), 1), ("BOTTOMPADDING", (0,0), (-1,-1), 6)]
            )
            image_rows.extend([img_row, label_row])

        first_two = image_rows[:2] if image_rows else []
        story.append(KeepTogether(heading_block + [col_header] + first_two))
        if len(image_rows) > 2:
            story.extend(image_rows[2:])
        story.append(Spacer(1, 0.4 * cm))

    if stat_report:
        story.append(PageBreak())
        story.append(Paragraph("Statistical Analysis", h1_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a3a5c"), spaceAfter=8))

        skip_desc = False
        for line in stat_report.split("\n"):
            stripped = line.strip()
            if "=== Descriptive Statistics ===" in stripped:
                skip_desc = True
                continue
            if skip_desc:
                if stripped.startswith("==="):
                    skip_desc = False
                    story.append(Paragraph(stripped.replace("=","").strip(), h2_style))
                continue
            if stripped.startswith("==="):
                story.append(Paragraph(stripped.replace("=","").strip(), h2_style))
            elif stripped:
                story.append(Paragraph(stripped, mono_style))
            else:
                story.append(Spacer(1, 0.10*cm))
        story.append(Spacer(1, 0.3*cm))

        if df_counts is not None and len(df_counts) > 0:
            story.append(Paragraph("Descriptive Statistics", h2_style))
            desc_header = ["Sample", "n", "Mean", "SD", "Median", "Min", "Max"]
            desc_data = [desc_header]
            for g in df_counts["Sample"].unique():
                vals = df_counts.loc[df_counts["Sample"] == g, "Count"].values
                desc_data.append([str(g), str(len(vals)), f"{vals.mean():.1f}", f"{vals.std():.1f}", f"{float(np.median(vals)):.1f}", str(int(vals.min())), str(int(vals.max()))])
            col_ws = [CONTENT_W * w for w in [0.28, 0.08, 0.12, 0.12, 0.14, 0.12, 0.14]]
            st = Table(desc_data, colWidths=col_ws)
            st.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a3a5c")), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 8.5),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f0f4f8"), colors.white]), ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#aaaaaa")),
                ("ALIGN", (1,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ]))
            story.append(st)

    if boxplot_path and os.path.exists(boxplot_path):
        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph("Viable Pollen Count — All Samples", h1_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#aaaaaa"), spaceAfter=6))
        story.append(RLImage(boxplot_path, width=CONTENT_W, height=CONTENT_W * 0.6))

    doc.build(story)
    return output_path

def _p_label(p):
    if p < 0.0001: return "****"
    elif p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return "ns"

def _p_str(p):
    return f"p={p:.2e}" if p < 0.0001 else f"p={p:.4f}"

def run_full_statistics(df_counts):
    if df_counts is None or len(df_counts["Sample"].unique()) < 2:
        return "Need at least 2 samples for statistical analysis.", [], "N/A"
    try:
        from scikit_posthocs import posthoc_dunn
        _has_dunn = True
    except ImportError:
        _has_dunn = False

    groups = list(df_counts["Sample"].unique())
    group_data = {g: df_counts.loc[df_counts["Sample"] == g, "Count"].values for g in groups}
    report = []
    report.append("=== Normality (Shapiro-Wilk) ===")
    all_normal = True
    for g, vals in group_data.items():
        if len(vals) >= 3:
            if np.ptp(vals) == 0:
                report.append(f"  {g}: all counts identical ({vals[0]}) — normality assumed")
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stat, p = shapiro(vals)
                normal = p > 0.05
                if not normal:
                    all_normal = False
                report.append(f"  {g}: W={stat:.4f}, p={_p_str(p)}  {'✓ normal' if normal else '✗ non-normal'}")
        else:
            report.append(f"  {g}: n<3, normality assumed (too few images)")

    arrays = list(group_data.values())
    sig_pairs = []

    if all_normal:
        report.append("\n=== One-Way ANOVA ===")
        f_stat, p_anova = f_oneway(*arrays)
        report.append(f"  F={f_stat:.4f}, p={_p_str(p_anova)}")
        method = "ANOVA + Tukey HSD"
        if p_anova < 0.05:
            report.append("\n=== Post-hoc: Tukey HSD ===")
            tukey = pairwise_tukeyhsd(df_counts["Count"], df_counts["Sample"])
            tdf = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            for _, row in tdf.iterrows():
                g1, g2 = row["group1"], row["group2"]
                p = float(row["p-adj"])
                diff = float(row["meandiff"])
                report.append(f"  {g1} vs {g2}: Δmean={diff:.2f}, {_p_str(p)} {_p_label(p)}")
                if p < 0.05: sig_pairs.append((g1, g2, p))
        else:
            report.append("  ANOVA not significant – no post-hoc needed.")
    else:
        report.append("\n=== Kruskal-Wallis Test ===")
        h_stat, p_kw = kruskal(*arrays)
        report.append(f"  H={h_stat:.4f}, p={_p_str(p_kw)}")
        method = "Kruskal-Wallis + Dunn (Bonferroni)"
        if p_kw < 0.05:
            if _has_dunn:
                report.append("\n=== Post-hoc: Dunn's Test (Bonferroni) ===")
                dunn = posthoc_dunn(df_counts, val_col="Count", group_col="Sample", p_adjust="bonferroni")
                for i, g1 in enumerate(groups):
                    for g2 in groups[i+1:]:
                        p = dunn.loc[g1, g2]
                        report.append(f"  {g1} vs {g2}: {_p_str(p)} {_p_label(p)}")
                        if p < 0.05: sig_pairs.append((g1, g2, float(p)))
            else:
                report.append("  Post-hoc: scikit-posthocs not installed. Falling back to pairwise Mann-Whitney U (Bonferroni corrected).")
                n_pairs = sum(1 for i in range(len(groups)) for j in range(i+1, len(groups)))
                for i, g1 in enumerate(groups):
                    for g2 in groups[i+1:]:
                        u, p_raw = mannwhitneyu(group_data[g1], group_data[g2], alternative='two-sided')
                        p = min(p_raw * n_pairs, 1.0)
                        report.append(f"  {g1} vs {g2}: U={u:.0f}, {_p_str(p)} {_p_label(p)} (Bonferroni)")
                        if p < 0.05: sig_pairs.append((g1, g2, float(p)))
        else:
            report.append("  Kruskal-Wallis not significant – no post-hoc needed.")

    report.append("\n=== Descriptive Statistics ===")
    report.append(f"  {'Sample':<20} {'n':>4} {'Mean':>8} {'SD':>8} {'Median':>8} {'Min':>6} {'Max':>6}")
    report.append("  " + "-" * 62)
    for g, vals in group_data.items():
        report.append(f"  {g:<20} {len(vals):>4} {vals.mean():>8.1f} {vals.std():>8.1f} {np.median(vals):>8.1f} {vals.min():>6} {vals.max():>6}")

    return "\n".join(report), sig_pairs, method

def _compact_letter_display(sig_pairs, groups):
    letters = list(string.ascii_lowercase)
    n = len(groups)
    group_idx = {g: i for i, g in enumerate(groups)}
    diff_set = set()
    for g1, g2, _ in sig_pairs:
        i, j = group_idx.get(g1, -1), group_idx.get(g2, -1)
        if i >= 0 and j >= 0:
            diff_set.add((min(i,j), max(i,j)))

    assigned = [[] for _ in range(n)]
    letter_id = 0
    for i in range(n):
        placed = False
        for li in range(letter_id + 1):
            conflict = False
            for j in range(n):
                if li < len(assigned[j]) or letters[li] in assigned[j]:
                    pair = (min(i,j), max(i,j))
                    if pair in diff_set:
                        conflict = True
                        break
            if not conflict:
                if letters[li] not in assigned[i]:
                    assigned[i].append(letters[li])
                placed = True
                break
        if not placed:
            letter_id += 1
            assigned[i].append(letters[letter_id])

    return {"".join(sorted(assigned[i])): groups[i] for i in range(n)}, {groups[i]: "".join(sorted(assigned[i])) for i in range(n)}

def plot_publication_figure(df_counts, sig_pairs, method, model_type, output_dir, sample_order=None, custom_title=None, custom_ylabel=None, pt_size=28, show_jitter=True, show_mean=True):
    _avail = {f.name for f in _fm.fontManager.ttflist}
    _font = "Arial" if "Arial" in _avail else "DejaVu Sans"
    with plt.rc_context({
        "font.family": _font, "font.size": 11, "axes.linewidth": 1.4,
        "axes.edgecolor": "black", "axes.facecolor": "white", "figure.facecolor": "white",
        "xtick.major.width": 1.4, "ytick.major.width": 1.4,
        "xtick.direction": "out", "ytick.direction": "out", "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        all_groups = list(df_counts["Sample"].unique())
        if sample_order:
            ordered = [s for s in sample_order if s in all_groups]
            remaining = [s for s in all_groups if s not in ordered]
            groups = ordered + remaining
        else:
            groups = all_groups
        n_groups = len(groups)
        FILLS = ["#CADDED", "#F7C8A0", "#B6DDB8", "#E8C5E5", "#FAEAA0", "#C8DDD0", "#F2B8B8"]
        EDGES = ["#2166AC", "#D6600A", "#1A7A2E", "#762A83", "#B8860B", "#2E6B4F", "#B22222"]
        fills = [FILLS[i % len(FILLS)] for i in range(n_groups)]
        edges = [EDGES[i % len(EDGES)] for i in range(n_groups)]

        fig_w = max(3.8, 1.85 * n_groups)
        fig = _MplFigure(figsize=(fig_w, 5.8), dpi=150)
        _FigureCanvasAgg(fig)   # attach non-interactive canvas so savefig works
        ax = fig.add_subplot(111)
        BW = 0.38          
        rng = np.random.default_rng(42)
        y_all_max = max(float(df_counts["Count"].max()), 1.0)

        for i, g in enumerate(groups):
            vals = df_counts.loc[df_counts["Sample"] == g, "Count"].values.astype(float)
            n = len(vals)
            fill, edge = fills[i], edges[i]
            q1, med, q3 = float(np.percentile(vals, 25)), float(np.median(vals)), float(np.percentile(vals, 75))
            iqr = q3 - q1
            lo = float(max(vals.min(), q1 - 1.5 * iqr))
            hi = float(min(vals.max(), q3 + 1.5 * iqr))
            mn, sd = float(vals.mean()), float(vals.std(ddof=1)) if n > 1 else 0.0

            rect = mpatches.FancyBboxPatch((i - BW, q1), 2 * BW, iqr, boxstyle="square,pad=0", facecolor=fill, edgecolor=edge, linewidth=1.6, zorder=3)
            ax.add_patch(rect)
            ax.plot([i - BW, i + BW], [med, med], color="black", lw=2.2, solid_capstyle="butt", zorder=4)
            ax.plot([i, i], [q3, hi], color=edge, lw=1.4, zorder=3)
            ax.plot([i, i], [q1, lo], color=edge, lw=1.4, zorder=3)
            cw = BW * 0.50
            ax.plot([i - cw, i + cw], [hi, hi], color=edge, lw=1.4, zorder=3)
            ax.plot([i - cw, i + cw], [lo, lo], color=edge, lw=1.4, zorder=3)

            if show_jitter:
                jit = rng.uniform(-BW * 0.55, BW * 0.55, size=n)
                ax.scatter(np.full(n, i) + jit, vals, color=edge, s=pt_size, alpha=0.85, edgecolors="white", linewidths=0.5, zorder=5, clip_on=True)

            if show_mean:
                GREY = "#333333"
                hw = BW * 0.55
                ax.plot([i - hw, i + hw], [mn, mn], color=GREY, lw=1.8, zorder=6, solid_capstyle="butt", linestyle="--")
                if sd > 0:
                    ax.plot([i, i], [mn - sd, mn + sd], color=GREY, lw=1.2, zorder=6)
                    sc = BW * 0.25
                    ax.plot([i - sc, i + sc], [mn + sd, mn + sd], color=GREY, lw=1.2, zorder=6)
                    ax.plot([i - sc, i + sc], [mn - sd, mn - sd], color=GREY, lw=1.2, zorder=6)

        _, cld = _compact_letter_display(sig_pairs, groups)
        y_top_data = y_all_max
        cld_y = y_top_data * 1.07
        for i, g in enumerate(groups):
            ltr = cld.get(g, "")
            if ltr: ax.text(i, cld_y, ltr, ha="center", va="bottom", fontsize=12, fontweight="bold", color="#222222")

        n_sig = len(sig_pairs) if sig_pairs else 0
        if sig_pairs and n_sig <= 4:
            brk_base = y_top_data * 1.18
            brk_step = y_top_data * 0.12
            brk_h = y_top_data * 0.022
            gpos = {g: i for i, g in enumerate(groups)}
            for k, (g1, g2, p) in enumerate(sig_pairs):
                x1, x2 = gpos.get(g1, -1), gpos.get(g2, -1)
                if x1 < 0 or x2 < 0: continue
                yb = brk_base + k * brk_step
                ax.plot([x1 + 0.04, x1 + 0.04, x2 - 0.04, x2 - 0.04], [yb, yb + brk_h, yb + brk_h, yb], lw=1.2, color="#222222", clip_on=False)
                ax.text((x1 + x2) / 2, yb + brk_h + y_top_data * 0.008, _p_label(p), ha="center", va="bottom", fontsize=11, color="#222222")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.4)
        ax.spines["bottom"].set_linewidth(1.4)
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)

        y_headroom = 1.30 if n_sig > 0 else 1.18
        ax.set_ylim(0, y_top_data * y_headroom)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        
        ax.set_xlim(-0.65, n_groups - 0.35)
        x_labels = [f"{g}\n(n={len(df_counts[df_counts['Sample']==g])})" for g in groups]
        ax.xaxis.set_major_locator(FixedLocator(range(n_groups)))
        ax.set_xticklabels(x_labels, fontsize=11)
        ax.tick_params(axis="x", length=0, pad=5)

        ax.set_ylabel(custom_ylabel or "Viable Pollen Count (per image)", fontsize=12, labelpad=8)
        ax.set_title(custom_title or "Viable Pollen Count \u2013 Alexander Stain", fontsize=13, fontweight="bold", pad=16, color="#111111")

        fig.text(0.5, -0.01, f"Statistics: {method}. Box = IQR, whiskers = 1.5×IQR. Letters indicate homogeneous groups (α = 0.05).", ha="center", fontsize=7.5, color="#666666", style="italic")
        fig.tight_layout(rect=[0, 0.04, 1, 1])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(output_dir, f"pollen_viability_{model_type}_{timestamp}.png")
        plot_path_pdf = plot_path.replace(".png", ".pdf")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(plot_path_pdf, dpi=300, bbox_inches="tight", facecolor="white")
        return plot_path, plot_path_pdf, fig

class StreamRedirector(io.StringIO):
    def __init__(self, signal):
        super().__init__()
        self.signal = signal
    def write(self, text):
        self.signal.emit(text)
    def flush(self):
        pass

class TrainingThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, model, train_data, train_labels, save_dir, model_name, epochs, lr):
        super().__init__()
        self.model = model
        self.train_data = train_data
        self.train_labels = train_labels
        self.save_dir = save_dir
        self.model_name = model_name
        self.epochs = epochs
        self.lr = lr

    def run(self):
        old_stdout = sys.stdout
        sys.stdout = StreamRedirector(self.log_signal)
        try:
            self.log_signal.emit("Initializing training (this may take a moment to start)...\n")

            # Check if we are on a version of Cellpose that still has model.train()
            if hasattr(self.model, 'train'):
                new_model_path = self.model.train(
                    self.train_data, self.train_labels,
                    channels=[0, 0],
                    save_path=self.save_dir,
                    n_epochs=self.epochs,
                    learning_rate=self.lr,
                    normalize=True,
                    model_name=self.model_name
                )
            else:
                # Cellpose v4.x removed .train() from the model class
                from cellpose import train as _cp_train
                if hasattr(_cp_train, 'train_seg'):
                    new_model_path = _cp_train.train_seg(
                        self.model.net,
                        train_data=self.train_data,
                        train_labels=self.train_labels,
                        save_path=self.save_dir,
                        n_epochs=self.epochs,
                        learning_rate=self.lr,
                        model_name=self.model_name
                    )
                else:
                    raise AttributeError(
                        "Your version of Cellpose does not support this training API."
                    )

            self.finished_signal.emit(str(new_model_path))
        except Exception as e:
            self.finished_signal.emit(f"ERROR: {str(e)}")
        finally:
            sys.stdout = old_stdout

class ZoomGraphicsView(QGraphicsView):
    zoom_applied = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor(30, 30, 30))
        self._is_panning = False
        self._pan_start = None
        self.scene_ref = QGraphicsScene(self)
        self.setScene(self.scene_ref)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            factor = 1.15
        else:
            factor = 1 / 1.15
        self.scale(factor, factor)
        self.zoom_applied.emit(factor)

    def apply_zoom(self, factor):
        old_anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.scale(factor, factor)
        self.setTransformationAnchor(old_anchor)

    def load_image(self, image_rgb):
        self.scene_ref.clear()
        arr_contiguous = np.ascontiguousarray(image_rgb, dtype=np.uint8)
        h, w, ch = arr_contiguous.shape
        qimg = QImage(arr_contiguous.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.img_item = self.scene_ref.addPixmap(QPixmap.fromImage(qimg.copy()))
        self.setSceneRect(self.img_item.boundingRect())

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_panning and self._pan_start:
            delta = event.pos() - self._pan_start
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._pan_start = event.pos()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

class InteractiveGraphicsView(QGraphicsView):
    zoom_applied = pyqtSignal(float)
    masks_updated = pyqtSignal(object)
    action_logged = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene_ref = QGraphicsScene(self)
        self.setScene(self.scene_ref)
        
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setBackgroundBrush(QColor(30, 30, 30))
        
        self.img_item = None
        self.mask_item = None
        self.masks = None
        self.mask_opacity = 127
        self.draw_mode = "Ellipse"
        
        self._pan_active = False
        self._pan_start_pos = None
        self._draw_active = False
        self._draw_start_pos = None
        self._draw_polygon = None
        self._draw_item = None

    def set_draw_mode(self, mode):
        self.draw_mode = mode

    def set_mask_opacity(self, val):
        self.mask_opacity = int((val / 100.0) * 255)
        self.update_mask_overlay(emit_signal=False)   # opacity change doesn't alter mask data

    def apply_zoom(self, factor):
        old_anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.scale(factor, factor)
        self.setTransformationAnchor(old_anchor)

    def create_safe_qpixmap(self, arr, format_type):
        arr_contiguous = np.ascontiguousarray(arr, dtype=np.uint8)
        h, w, ch = arr_contiguous.shape
        qimg = QImage(arr_contiguous.data, w, h, ch * w, format_type)
        return QPixmap.fromImage(qimg.copy())

    def load_image_and_masks(self, image_rgb, masks):
        self.scene_ref.clear()
        self.masks = masks.copy() if masks is not None else np.zeros(image_rgb.shape[:2], dtype=np.uint16)
        
        safe_pixmap = self.create_safe_qpixmap(image_rgb, QImage.Format.Format_RGB888)
        self.img_item = self.scene_ref.addPixmap(safe_pixmap)
        
        self.mask_item = None
        self.update_mask_overlay()
        
        self.setSceneRect(self.img_item.boundingRect())

    def update_mask_overlay(self, emit_signal=True):
        if self.masks is None: return

        if self.mask_item is not None and self.mask_item in self.scene_ref.items():
            self.scene_ref.removeItem(self.mask_item)

        h, w = self.masks.shape
        # ── Fully-vectorised colourmap (no Python loop over mask IDs) ─────────
        GOLDEN = 0.6180339887
        max_id = int(self.masks.max())
        if max_id > 0:
            ids = np.arange(max_id + 1, dtype=np.float32)
            hues = np.mod(ids * GOLDEN, 1.0)
            # HLS → RGB via fast numpy path (l=0.55, s=0.9)
            l, s = 0.55, 0.90
            q = np.where(l < 0.5, l * (1 + s), l + s - l * s).astype(np.float32)
            p = (2 * l - q).astype(np.float32)
            def _chan(t):
                t = t % 1.0
                return np.where(t < 1/6, p + (q-p)*6*t,
                       np.where(t < 1/2, q,
                       np.where(t < 2/3, p + (q-p)*(2/3-t)*6, p)))
            r_lut = (_chan(hues + 1/3) * 255).astype(np.uint8)
            g_lut = (_chan(hues      ) * 255).astype(np.uint8)
            b_lut = (_chan(hues - 1/3) * 255).astype(np.uint8)
            # Build RGBA via LUT indexing — O(pixels) not O(pixels * n_ids)
            flat = self.masks.ravel()
            rgba = np.empty((flat.shape[0], 4), dtype=np.uint8)
            rgba[:, 0] = r_lut[flat]
            rgba[:, 1] = g_lut[flat]
            rgba[:, 2] = b_lut[flat]
            alpha_lut = np.zeros(max_id + 1, dtype=np.uint8)
            alpha_lut[1:] = self.mask_opacity
            rgba[:, 3] = alpha_lut[flat]
            rgba = rgba.reshape(h, w, 4)
        else:
            rgba = np.zeros((h, w, 4), dtype=np.uint8)

        safe_pixmap = self.create_safe_qpixmap(rgba, QImage.Format.Format_RGBA8888)
        self.mask_item = self.scene_ref.addPixmap(safe_pixmap)
        self.mask_item.setZValue(1)
        if emit_signal:
            self.masks_updated.emit(self.masks)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1/1.15
        self.scale(factor, factor)
        self.zoom_applied.emit(factor)

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_active = True
            self._pan_start_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            
        elif event.button() == Qt.MouseButton.RightButton:
            self._draw_active = True
            self._draw_start_pos = scene_pos
            if self.draw_mode == "Freehand Polygon":
                self._draw_polygon = QPolygonF([scene_pos])
                self._draw_item = self.scene_ref.addPolygon(self._draw_polygon, QPen(Qt.GlobalColor.yellow, 2))
            else: 
                self._draw_item = self.scene_ref.addEllipse(QRectF(scene_pos, scene_pos), QPen(QColor(0, 255, 255), 2, Qt.PenStyle.DashLine))
            self._draw_item.setZValue(2)
            event.accept()
            
        elif event.button() == Qt.MouseButton.LeftButton and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            if self.masks is not None:
                x, y = int(scene_pos.x()), int(scene_pos.y())
                if 0 <= x < self.masks.shape[1] and 0 <= y < self.masks.shape[0]:
                    mask_id = self.masks[y, x]
                    if mask_id != 0:
                        self.masks[self.masks == mask_id] = 0
                        self.update_mask_overlay()
                        self.action_logged.emit(f"Status: Removed mask {mask_id}")
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        
        if self._pan_active:
            delta = event.pos() - self._pan_start_pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._pan_start_pos = event.pos()
            event.accept()
            
        elif self._draw_active and self._draw_item is not None:
            if self.draw_mode == "Freehand Polygon":
                self._draw_polygon.append(scene_pos)
                self._draw_item.setPolygon(self._draw_polygon)
            else:
                rect = QRectF(self._draw_start_pos, scene_pos).normalized()
                self._draw_item.setRect(rect)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_active = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            
        elif event.button() == Qt.MouseButton.RightButton and self._draw_active:
            self._draw_active = False
            if self.draw_mode == "Freehand Polygon":
                if self._draw_polygon.count() > 2:
                    self._burn_polygon_to_mask(self._draw_polygon)
            else:
                rect = self._draw_item.rect()
                if rect.width() > 0 and rect.height() > 0:
                    self._burn_ellipse_to_mask(rect)
                    
            if self._draw_item in self.scene_ref.items():
                self.scene_ref.removeItem(self._draw_item)
            self._draw_item = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def _burn_polygon_to_mask(self, polygon):
        if self.masks is None: return
        h, w = self.masks.shape
        temp_img = QImage(w, h, QImage.Format.Format_Grayscale8)
        temp_img.fill(0)
        
        painter = QPainter(temp_img)
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(polygon)
        painter.end()
        
        ptr = temp_img.bits()
        ptr.setsize(h * w)
        drawn_mask = np.array(ptr).reshape((h, w)).copy()
        
        new_id = int(self.masks.max()) + 1 if self.masks.max() > 0 else 1
        self.masks[drawn_mask > 0] = new_id
        self.update_mask_overlay()
        self.action_logged.emit(f"Status: Added drawn mask {new_id}")

    def _burn_ellipse_to_mask(self, rect):
        if self.masks is None: return
        h, w = self.masks.shape
        
        cx = int(rect.center().x())
        cy = int(rect.center().y())
        rx = int(rect.width() / 2)
        ry = int(rect.height() / 2)
        
        new_id = int(self.masks.max()) + 1 if self.masks.max() > 0 else 1
        cv2.ellipse(self.masks, (cx, cy), (rx, ry), 0, 0, 360, new_id, -1)
        self.update_mask_overlay()
        self.action_logged.emit(f"Status: Added drawn ellipse {new_id}")


class SegmentationThread(QThread):
    """Runs Cellpose segmentation off the main thread to keep the UI responsive."""
    finished = pyqtSignal(object, object)   # masks_filtered, img_resized
    error    = pyqtSignal(str)
    progress = pyqtSignal(int, int)         # current, total  (for batch)

    def __init__(self, model, images_params, parent=None):
        """images_params: list of dicts with keys: image, diameter, flow_thresh,
           cellprob, max_iter, norm_low, norm_up, resize_val, channel_func"""
        super().__init__(parent)
        self.model = model
        self.images_params = images_params

    def run(self):
        try:
            results = []
            total = len(self.images_params)
            for i, p in enumerate(self.images_params):
                img_resized = image_resize(p['image'], p['resize_val'])
                processed   = p['channel_func'](img_resized)
                diameter    = p['diameter']
                normalize_params = {"percentile": [p['norm_low'], p['norm_up']]}
                try:
                    result = self.model.eval(
                        processed,
                        diameter=diameter if diameter > 0 else None,
                        flow_threshold=p['flow_thresh'],
                        cellprob_threshold=p['cellprob'],
                        normalize=normalize_params,
                        niter=p['max_iter'],
                        batch_size=8,
                    )
                    masks = result[0]
                except Exception as seg_err:
                    print(f"Warning: segmentation failed for image {i+1}: {seg_err}")
                    masks = np.zeros(img_resized.shape[:2], dtype=np.uint16)

                min_grain_area = _compute_min_grain_area(diameter, masks)
                counts_per_id  = np.bincount(masks.ravel())
                valid_ids      = [n for n in np.unique(masks)[1:] if counts_per_id[n] >= min_grain_area]
                masks_filtered = np.zeros_like(masks)
                if valid_ids:
                    id_map = np.zeros(int(masks.max()) + 1, dtype=np.uint32)
                    for new_id, old_id in enumerate(valid_ids, start=1):
                        id_map[old_id] = new_id
                    masks_filtered = id_map[masks].astype(np.uint16)

                results.append((masks_filtered, img_resized))
                self.progress.emit(i + 1, total)

            self.finished.emit(results, None)
        except Exception as e:
            self.error.emit(str(e))


class AnalysisThread(QThread):
    """Runs overlay generation, statistics, plot and PDF off the main thread."""
    progress      = pyqtSignal(int, int, str)   # current, total, message
    finished      = pyqtSignal(object)           # dict of results
    error         = pyqtSignal(str)

    def __init__(self, validation_entries, output_dir, timestamp,
                 save_int, plot_order, batch_folder_path=None, parent=None):
        super().__init__(parent)
        self.validation_entries = validation_entries
        self.output_dir         = output_dir
        self.timestamp          = timestamp
        self.save_int           = save_int
        self.plot_order         = plot_order
        self.batch_folder_path  = batch_folder_path

    def run(self):
        try:
            counts     = []
            total      = len(self.validation_entries)

            # ── Determine worker count: half of available CPUs, minimum 1 ──────
            try:
                cpu_count  = multiprocessing.cpu_count()
            except Exception:
                cpu_count  = 2
            n_workers = max(1, cpu_count // 2)

            # ── Build the work list and per-entry metadata in one pass ──────────
            work_items  = []   # (index, img_np, mask_np) sent to workers
            entry_meta  = []   # (sname, img_file, num_pollen) kept on main thread

            for i, entry in enumerate(self.validation_entries):
                img_file = entry.get("file", "Unknown")
                img_orig = entry["image"]
                mask     = entry["mask"]
                if mask is None:
                    continue

                dirname = os.path.dirname(img_file)
                if self.batch_folder_path and dirname == self.batch_folder_path:
                    sname = "Sample"
                else:
                    sname = os.path.basename(dirname) if dirname else "Sample"
                if not sname or sname == "Unknown":
                    sname = "Sample"

                num_pollen = int(mask.max())
                counts.append({"Sample": sname, "Count": num_pollen,
                                "Image": os.path.basename(img_file)})

                work_items.append((i, img_orig, mask))
                entry_meta.append((sname, img_file, num_pollen))

            if not counts:
                self.finished.emit({"empty": True})
                return

            # ── Parallel overlay + outline rendering ────────────────────────────
            # Try ProcessPoolExecutor first (true CPU parallelism).
            # Fall back to ThreadPoolExecutor if spawning processes fails
            # (e.g. frozen/packaged builds, some HPC environments).
            self.progress.emit(0, total, f"Rendering overlays with {n_workers} workers…")
            render_results = [None] * len(work_items)

            def _collect_parallel(executor_cls, **kw):
                futures = {}
                with executor_cls(max_workers=n_workers, **kw) as ex:
                    for item in work_items:
                        fut = ex.submit(_render_overlays_worker, item)
                        futures[fut] = item[0]  # map future → original entry index
                done_count = 0
                for fut in concurrent.futures.as_completed(futures):
                    idx, orig_pil, overlay_pil, outline_pil = fut.result()
                    # find position in work_items list
                    pos = next(j for j, w in enumerate(work_items) if w[0] == idx)
                    render_results[pos] = (orig_pil, overlay_pil, outline_pil)
                    done_count += 1
                    self.progress.emit(done_count, total,
                                       f"Rendered overlays {done_count}/{total}…")

            # Use ThreadPoolExecutor only: overlay rendering is numpy/PIL I/O that
            # releases the GIL, so threads give good parallelism without spawning
            # new processes.  ProcessPoolExecutor with the "spawn" start method
            # re-executes this module's top-level code (including model loading)
            # in each worker, which fails because the torch.load patch is no
            # longer active at that point.
            _collect_parallel(concurrent.futures.ThreadPoolExecutor)

            # ── Assemble image_data and optionally save intermediates ───────────
            image_data = []
            for pos, (orig_pil, overlay_pil, outline_pil) in enumerate(render_results):
                if orig_pil is None:
                    continue
                sname, img_file, num_pollen = entry_meta[pos]
                image_data.append((sname, img_file, orig_pil,
                                   overlay_pil, outline_pil, num_pollen))

                if self.save_int:
                    sd   = os.path.join(self.output_dir, sname)
                    os.makedirs(sd, exist_ok=True)
                    base = os.path.splitext(os.path.basename(img_file))[0]
                    overlay_pil.save(os.path.join(sd, f"{base}_overlay.png"))
                    outline_pil.save(os.path.join(sd, f"{base}_outlines.png"))
                    try:
                        mask = self.validation_entries[
                            next(j for j, e in enumerate(self.validation_entries)
                                 if e.get("file", "") == img_file)
                        ]["mask"]
                        if _imsave is not None:
                            _imsave(os.path.join(sd, f"{base}_masks.tif"), mask)
                        else:
                            # Fallback: save as PNG if cellpose imsave unavailable
                            cv2.imwrite(
                                os.path.join(sd, f"{base}_masks.png"),
                                np.clip(mask, 0, 255).astype(np.uint8)
                            )
                    except Exception:
                        pass

            df_counts = pd.DataFrame(counts)
            csv_path  = os.path.join(self.output_dir,
                                     f"pollen_counts_{self.timestamp}.csv")
            df_counts.to_csv(csv_path, index=False)

            self.progress.emit(total, total, "Running statistics…")
            stat_report, sig_pairs, stat_method = run_full_statistics(df_counts)

            self.progress.emit(total, total, "Generating plot…")
            plot_path, plot_path_pdf, fig = plot_publication_figure(
                df_counts, sig_pairs, stat_method, "pyqt6",
                self.output_dir, sample_order=self.plot_order
            )

            self.progress.emit(total, total, "Building PDF report…")
            pdf_path = os.path.join(self.output_dir,
                                    f"pollen_counts_report_{self.timestamp}.pdf")
            generate_pdf_report(image_data, pdf_path, stat_report=stat_report,
                                 boxplot_path=plot_path, df_counts=df_counts,
                                 sample_order=self.plot_order)

            self.finished.emit({
                "empty":       False,
                "df_counts":   df_counts,
                "csv_path":    csv_path,
                "stat_report": stat_report,
                "sig_pairs":   sig_pairs,
                "stat_method": stat_method,
                "plot_path":   plot_path,
                "pdf_path":    pdf_path,
                "fig":         fig,
                "image_data":  image_data,
            })
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")

class PollenAnalysisApp(QMainWindow):
    def __init__(self, font_scale=1.0):
        super().__init__()
        self._font_scale = font_scale
        self._theme = "Forest (default)"
        self.setWindowTitle("Pollen Analysis Tool")
        self.setGeometry(100, 100, 1440, 920)
        self.setMinimumSize(1100, 740)

        # Initialize variables
        self.image = None
        self.image_display = None
        self.mask = None
        self.current_model = model_cellpose
        self._active_model_display_name = (
            "cellpose-SAM" if model_cpsam is not None else "cellpose (built-in)"
        )
        self.batch_files = []
        self.batch_results = []
        
        self.stop_requested = False
        self.validation_entries = []
        self.current_idx = -1
        self.draw_current_item = None
        self.output_dir = None

        # Debounce timer for contrast/brightness sliders — avoids flooding the
        # main thread with full redraws on every slider tick
        self._display_timer = QTimer(self)
        self._display_timer.setSingleShot(True)
        self._display_timer.setInterval(40)   # 40 ms → max ~25 fps while dragging
        self._display_timer.timeout.connect(self._do_display_image)

        self.init_ui()

    # ── theme helpers ──────────────────────────────────────────────────────────
    def _build_stylesheet(self, scale=1.0, theme=None):
        """scale=1.0 → default sizes. Pass e.g. 1.3 for +30% larger text."""
        THEMES = {
            # (BG, CARD, SIDE, TXT, MUTE, ACC, ACC2, DARK, BDR_SOLID, HI_MAIN,
            #  BTN_TXT, SIDE_TXT, SIDE_MUTE, SIDE_HI, HDR_BG)

            # ── Built-in default ──────────────────────────────────────────────
            "Forest (default)": (
                "#F4F1EA", "#EDEAE1", "#1E2D1E", "#1A271A", "#6B7D60",
                "#3A6B35", "#4E8A47", "#2A5226", "#C8D4C0", "#DFF0D9",
                "#FFFFFF", "#C8DDB8", "#7A9B6A", "#EEF7E8", "#EAE6DC",
            ),

            # ── Palette 1: Linen & Spice ──────────────────────────────────────
            # BG=linen, CARD=off-white, SIDE=warm dark brown, TXT=deep brown
            # MUTE=warm tan, ACC=spice, ACC2=muted teal, DARK=darker spice
            # BDR=warm border, HI=light spice tint, SIDE_TXT=pale linen
            # SIDE_MUTE=mid-spice, SIDE_HI=bright linen, HDR=warm grey
            "Linen & Spice": (
                "#EDE9E6", "#E4DED9", "#5C4F4A", "#2E1F1A", "#8A7060",
                "#C9996B", "#5C766D", "#A0703A", "#D4C4B4", "#F0E0CC",
                "#FFFFFF", "#EDE9E6", "#C4A080", "#FFF8F0", "#E0D8D0",
            ),

            # ── Palette 2: Oat & Moss ─────────────────────────────────────────
            # BG=warm cream, CARD=ivory, SIDE=mid-tan, TXT=deep brown
            # MUTE=warm grey, ACC=sage-green, ACC2=lighter sage, DARK=darker tan
            # BDR=warm border, HI=sage tint, SIDE_TXT=pale oat, HDR=cream
            "Oat & Moss": (
                "#F3E4C9", "#EDD8B6", "#A98B76", "#3A2818", "#8A7A62",
                "#7A8A60", "#BABF94", "#7A6448", "#D4BFA0", "#E8F0D4",
                "#FFFFFF", "#F3E4C9", "#C4A882", "#FFF8EE", "#EDE0CC",
            ),

            # ── Palette 3: Ivory & Ember ──────────────────────────────────────
            # BG=ivory, CARD=warm white, SIDE=deep amber-brown, TXT=near-black
            # MUTE=mid brown, ACC=ember orange, ACC2=pale ember, DARK=deeper brown
            # BDR=warm cream, HI=pale ember tint, SIDE_TXT=warm cream, HDR=off-white
            "Ivory & Ember": (
                "#FFFDF1", "#FFF5DC", "#562F00", "#2A1400", "#8A6040",
                "#FF9644", "#FFCE99", "#CC6600", "#E8D0A8", "#FFE8CC",
                "#2A1400", "#FFCE99", "#CC8844", "#FFFDF1", "#FFF5E0",
            ),

            # ── Palette 4: Moss & Ochre ───────────────────────────────────────
            # BG=light grey, CARD=white, SIDE=olive-moss, TXT=dark olive
            # MUTE=warm grey, ACC=ochre orange, ACC2=parchment, DARK=darker moss
            # BDR=warm border, HI=pale parchment, SIDE_TXT=parchment, HDR=off-white
            "Moss & Ochre": (
                "#EEEEEE", "#E6E4DC", "#5C6F2B", "#1E2408", "#6A7050",
                "#DE802B", "#D8C9A7", "#3A4A10", "#C8C4A8", "#F0E4CC",
                "#FFFFFF", "#D8C9A7", "#8A9A50", "#F8F4E8", "#E4E0D4",
            ),

            # ── Palette 5: Linen & Forest ─────────────────────────────────────
            # BG=linen, CARD=warm white, SIDE=deep forest, TXT=near-black
            # MUTE=mid green, ACC=burnt orange, ACC2=lighter orange, DARK=darker forest
            # BDR=warm border, HI=pale orange tint, SIDE_TXT=pale linen
            # SIDE_MUTE=mid forest, SIDE_HI=bright linen, HDR=warm linen
            "Linen & Forest": (
                "#EBE1D1", "#E2D6C2", "#0D4715", "#1A1A0A", "#6A7A52",
                "#E9762B", "#F0A060", "#A04010", "#C8BBA8", "#FFE4CC",
                "#FFFFFF", "#D4E8CC", "#5A8A60", "#F0FFE8", "#E0D4C0",
            ),
        }
        theme_name = theme if theme is not None else getattr(self, '_theme', "Forest (default)")
        (BG, CARD, SIDE, TXT, MUTE, ACC, ACC2, DARK, BDR_SOLID, HI_MAIN,
         BTN_TXT, SIDE_TXT, SIDE_MUTE, SIDE_HI, HDR_BG) = THEMES[theme_name]
        return f"""
* {{
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, Arial, sans-serif;
    color: {TXT}; outline: none;
}}
QMainWindow, QDialog {{ background: {BG}; }}
QScrollArea {{ background: transparent; border: none; }}
QWidget#ContentStack {{ background: {BG}; }}

QFrame#LogoFrame {{
    background: {SIDE};
    border: none;
    min-height: 52px;
}}

QFrame#Sidebar {{
    background: {SIDE};
    border-right: 1px solid rgba(0,0,0,0.18);
}}
QLabel#LogoTitle {{
    font-size: {round(12*scale)}px; font-weight: bold; color: {SIDE_HI};
    margin: 0; padding: 0;
}}
QLabel#LogoSub {{
    font-size: {round(8*scale)}px; font-weight: normal; color: {SIDE_MUTE};
    letter-spacing: 1.2px;
}}
QFrame#SidebarSep {{
    background: rgba(255,255,255,0.07); border: none;
    max-height: 1px; min-height: 1px;
}}

QScrollBar:vertical {{ background: transparent; width: 7px; margin: 0; }}
QScrollBar::handle:vertical {{
    background: rgba(100,140,90,0.30); border-radius: 3px; min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: {ACC}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ height: 7px; background: transparent; }}
QScrollBar::handle:horizontal {{
    background: rgba(100,140,90,0.30); border-radius: 3px; min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{ background: {ACC}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

QLabel#NavSection {{
    font-size: {round(8*scale)}px; font-weight: bold; color: {SIDE_MUTE};
    letter-spacing: 2px;
    padding: 12px 14px 4px 16px;
}}

QPushButton#NavBtn {{
    background: transparent; border: none; border-radius: 6px;
    color: {SIDE_TXT}; font-size: {round(12*scale)}px; font-weight: normal;
    text-align: left; padding: 5px 10px 5px 14px;
    min-height: 32px;
}}
QPushButton#NavBtn:hover {{
    background: rgba(255,255,255,0.07);
    color: {SIDE_HI};
}}
QPushButton#NavBtn:checked {{
    background: rgba(58,107,53,0.28);
    color: {SIDE_HI}; font-weight: bold;
    border-left: 3px solid {ACC2};
    padding-left: 12px;
}}

QLabel#SidebarStatus {{
    font-size: {round(10*scale)}px; color: {SIDE_TXT};
    padding: 7px 12px 8px 14px;
    line-height: 1.5;
}}

QFrame#Sidebar QScrollArea {{ background: transparent; border: none; }}
QFrame#Sidebar QScrollArea QWidget {{ background: transparent; }}

QFrame#PageHeader {{
    background: {HDR_BG};
    border-bottom: 1px solid {BDR_SOLID};
}}
QLabel#PageTitle {{
    font-size: {round(15*scale)}px; font-weight: bold;
    color: {TXT}; letter-spacing: -0.3px;
}}
QLabel#PageSubtitle {{
    font-size: {round(10*scale)}px; color: {MUTE}; font-weight: normal;
    font-style: italic;
}}

QFrame#Card {{
    background: {CARD};
    border: 1px solid {BDR_SOLID};
    border-radius: 10px;
}}
QLabel#CardTitle {{
    font-size: {round(9*scale)}px; font-weight: bold;
    color: {MUTE}; letter-spacing: 1.8px;
    padding-bottom: 4px;
    margin-bottom: 2px;
    border-bottom: 1px solid {BDR_SOLID};
}}
QLabel#HintLabel {{
    font-size: {round(11*scale)}px; color: {MUTE};
    line-height: 1.5;
}}
QLabel#FormLabel {{
    font-size: {round(11*scale)}px; color: {TXT}; font-weight: normal;
}}

QLabel#TrainBaseLabel {{
    font-size: {round(11*scale)}px; font-weight: bold;
    color: {ACC};
    background: rgba(58,107,53,0.08);
    border: 1px solid rgba(58,107,53,0.28);
    border-radius: 6px;
    padding: 6px 10px;
}}

QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit, QComboBox {{
    background: {BG}; border: 1px solid {BDR_SOLID};
    border-radius: 6px; padding: 4px 8px;
    font-size: {round(12*scale)}px; color: {TXT};
    selection-background-color: {HI_MAIN};
}}
QSpinBox:focus, QDoubleSpinBox:focus,
QLineEdit:focus, QTextEdit:focus, QComboBox:focus {{
    border-color: {ACC}; outline: none;
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {CARD}; border: 1px solid {BDR_SOLID};
    selection-background-color: {HI_MAIN}; outline: none;
    border-radius: 6px;
}}
QRadioButton {{ font-size: {round(12*scale)}px; color: {TXT}; spacing: 6px; }}
QCheckBox   {{ font-size: {round(12*scale)}px; color: {TXT}; spacing: 6px; }}
QCheckBox#OptionCheck {{ font-size: {round(12*scale)}px; }}

QPushButton#PrimaryBtn {{
    background: {ACC}; color: {BTN_TXT};
    border: none; border-bottom: 2px solid {DARK};
    border-radius: 8px; padding: 0 16px;
    font-size: {round(12*scale)}px; font-weight: bold;
    min-height: 34px;
}}
QPushButton#PrimaryBtn:hover  {{ background: {DARK}; }}
QPushButton#PrimaryBtn:pressed {{ border-bottom-width: 1px; padding-top: 1px; }}
QPushButton#PrimaryBtn:disabled {{
    background: {HI_MAIN}; color: {MUTE}; border-bottom-color: {BDR_SOLID};
}}

QPushButton#ActionBtn {{
    background: {CARD}; color: {TXT};
    border: 1px solid {BDR_SOLID}; border-bottom: 2px solid {BDR_SOLID};
    border-radius: 8px; padding: 0 12px;
    font-size: {round(12*scale)}px; font-weight: 500; min-height: 30px;
}}
QPushButton#ActionBtn:hover  {{ background: {HI_MAIN}; border-color: {ACC}; }}
QPushButton#ActionBtn:pressed {{ background: {HI_MAIN}; border-bottom-width: 1px; padding-top: 1px; }}
QPushButton#ActionBtn:disabled {{ color: {MUTE}; }}

QPushButton#SecondaryBtn {{
    background: {CARD}; color: {TXT};
    border: 1px solid {BDR_SOLID}; border-bottom: 2px solid {BDR_SOLID};
    border-radius: 8px; padding: 0 12px;
    font-size: {round(12*scale)}px; font-weight: 500; min-height: 30px;
}}
QPushButton#SecondaryBtn:hover  {{ background: {HI_MAIN}; border-color: {ACC}; }}
QPushButton#SecondaryBtn:pressed {{ border-bottom-width: 1px; padding-top: 1px; }}

QPushButton#DangerBtn {{
    background: transparent; color: #C94040;
    border: 1px solid rgba(201,64,64,0.30);
    border-bottom: 2px solid rgba(201,64,64,0.30);
    border-radius: 8px; padding: 0 12px;
    font-size: {round(12*scale)}px; font-weight: bold; min-height: 30px;
}}
QPushButton#DangerBtn:hover {{
    background: rgba(201,64,64,0.08);
    border-color: rgba(201,64,64,0.60);
}}
QPushButton#DangerBtn:pressed {{ border-bottom-width: 1px; padding-top: 1px; }}

QTableWidget {{
    background: {CARD}; border: 1px solid {BDR_SOLID};
    border-radius: 10px; gridline-color: {BDR_SOLID};
    color: {TXT}; font-size: {round(12*scale)}px; outline: none;
    alternate-background-color: {BG};
}}
QHeaderView::section {{
    background: {SIDE}; color: {SIDE_TXT};
    font-size: {round(9*scale)}px; font-weight: bold; letter-spacing: 1.5px;
    padding: 7px 10px; border: none;
    border-bottom: 1px solid rgba(0,0,0,0.20);
}}
QTableWidget::item {{ padding: 5px 10px; border: none; }}
QTableWidget::item:selected {{ background: {HI_MAIN}; color: {ACC}; }}

QListWidget {{
    background: {BG}; border: 1px solid {BDR_SOLID};
    border-radius: 10px; outline: none;
    alternate-background-color: {CARD};
    padding: 4px;
}}
QListWidget::item {{
    padding: 7px 12px; border-radius: 6px;
    color: {TXT}; font-size: {round(12*scale)}px; min-height: 28px;
}}
QListWidget::item:hover   {{ background: {HI_MAIN}; }}
QListWidget::item:selected {{
    background: {HI_MAIN}; color: {ACC};
    font-weight: bold; border: none;
}}

QTabWidget::pane {{ border: none; background: {BG}; }}
QTabBar::tab {{
    background: {SIDE}; color: {SIDE_TXT};
    padding: 9px 18px; border: none;
    border-top-left-radius: 8px; border-top-right-radius: 8px;
    font-weight: 500; font-size: {round(12*scale)}px; margin-right: 3px;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ background: {CARD}; color: {ACC}; border-bottom: 2px solid {ACC}; }}
QTabBar::tab:hover    {{ color: {TXT}; background: rgba(58,107,53,0.12); }}

QProgressBar {{
    background: {CARD}; border: 1px solid {BDR_SOLID};
    border-radius: 4px; min-height: 6px; max-height: 6px; text-align: center;
}}
QProgressBar::chunk {{ background: {ACC}; border-radius: 4px; }}

QToolTip {{
    background: #1A271A; border: none; border-radius: 6px;
    color: #EEF7E8; padding: 7px 11px;
    font-size: {round(11*scale)}px; font-weight: normal;
}}

QFrame#FontScalePanel {{
    background: transparent; border: none;
    padding: 6px 14px 8px 14px;
}}
QLabel#FontScaleLabel {{
    font-size: {round(8*scale)}px; font-weight: bold; color: {SIDE_MUTE};
    letter-spacing: 1.5px;
}}
QLabel#FontScaleValue {{
    font-size: {round(10*scale)}px; font-weight: normal; color: {SIDE_TXT};
}}
QSlider#FontScaleSlider::groove:horizontal {{
    height: 4px; background: rgba(255,255,255,0.12); border-radius: 2px;
}}
QSlider#FontScaleSlider::handle:horizontal {{
    background: {ACC}; border: none;
    width: 14px; height: 14px; border-radius: 7px; margin: -5px 0;
}}
QSlider#FontScaleSlider::handle:horizontal:hover {{ background: {SIDE_HI}; }}
QSlider#FontScaleSlider::sub-page:horizontal {{ background: {ACC}; border-radius: 2px; }}

QSlider::groove:horizontal {{
    height: 4px; background: {BDR_SOLID}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACC}; border: 2px solid {CARD};
    width: 14px; height: 14px; border-radius: 7px; margin: -5px 0;
}}
QSlider::handle:horizontal:hover {{ background: {DARK}; }}
QSlider::sub-page:horizontal {{ background: {ACC}; border-radius: 2px; }}

QComboBox#ThemeCombo {{
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 5px;
    color: {SIDE_TXT};
    font-size: {round(10*scale)}px;
    padding: 3px 8px;
}}
QComboBox#ThemeCombo::drop-down {{ border: none; width: 18px; }}
QComboBox#ThemeCombo QAbstractItemView {{
    background: {SIDE};
    border: 1px solid rgba(255,255,255,0.14);
    color: {SIDE_TXT};
    selection-background-color: rgba(255,255,255,0.12);
    outline: none;
}}

QTextEdit#TrainLog, QTextEdit#ResultsLog {{
    background: #1A1F1A;
    color: #B8DDB0;
    border: 1px solid rgba(58,107,53,0.30);
    border-radius: 8px;
    font-family: "Courier New", Courier, monospace;
    font-size: {round(11*scale)}px;
    selection-background-color: rgba(58,107,53,0.40);
    padding: 6px;
}}

QSplitter#RootSplitter::handle {{ background: {BDR_SOLID}; width: 4px; }}
QSplitter::handle {{ background: {BDR_SOLID}; }}
        """

    def _on_font_scale_changed(self, value):
        """Live-update all font sizes when the sidebar slider moves."""
        self._font_scale = value / 100.0
        self._font_scale_value_lbl.setText(f"{value}%")
        # Rebuild and reapply stylesheet with new scale
        self.setStyleSheet(self._build_stylesheet(self._font_scale))
        # Also update the QApplication base font so non-stylesheet widgets scale too
        app = QApplication.instance()
        if app:
            _screen = app.primaryScreen()
            _dpi    = _screen.logicalDotsPerInch() if _screen else 96.0
            _pt     = max(7, int(round(10 * self._font_scale * (_dpi / 96.0))))
            f = app.font()
            f.setPointSize(_pt)
            app.setFont(f)

    def _make_card(self, title=""):
        """Return (QFrame, QVBoxLayout) card container."""
        frame = QFrame()
        frame.setObjectName("Card")
        lyt = QVBoxLayout(frame)
        lyt.setContentsMargins(16, 12, 16, 14)
        lyt.setSpacing(10)
        if title:
            lbl = QLabel(title.upper())
            lbl.setObjectName("CardTitle")
            lyt.addWidget(lbl)
        return frame, lyt

    def _page_header(self, title, subtitle=""):
        hdr = QFrame()
        hdr.setObjectName("PageHeader")
        hdr.setFixedHeight(48)
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(24, 0, 24, 0)
        t = QLabel(title)
        t.setObjectName("PageTitle")
        hl.addWidget(t)
        if subtitle:
            dot = QLabel("·")
            dot.setObjectName("PageSubtitle")
            dot.setContentsMargins(8, 0, 8, 0)
            hl.addWidget(dot)
            s = QLabel(subtitle)
            s.setObjectName("PageSubtitle")
            hl.addWidget(s)
            hl.addStretch()
        return hdr

    def _form_row(self, label_text, widget, label_width=160):
        row = QHBoxLayout()
        row.setSpacing(10)
        lbl = QLabel(label_text)
        lbl.setObjectName("FormLabel")
        lbl.setFixedWidth(label_width)
        row.addWidget(lbl)
        row.addWidget(widget, 1)
        return row

    def _switch_page(self, index):
        self._stack.setCurrentIndex(index)
        # Update nav button checked states
        for btn in self._nav_buttons:
            btn.setChecked(btn.property("navIndex") == index)

    # ── main init_ui ──────────────────────────────────────────────────────────
    def init_ui(self):
        self.setStyleSheet(self._build_stylesheet(self._font_scale))

        central = QWidget()
        self.setCentralWidget(central)
        root_lyt = QHBoxLayout(central)
        root_lyt.setContentsMargins(0, 0, 0, 0)
        root_lyt.setSpacing(0)

        shell = QSplitter(Qt.Orientation.Horizontal)
        shell.setObjectName("RootSplitter")
        shell.setHandleWidth(4)
        root_lyt.addWidget(shell)

        # ══════════════════════════════════════════════════════
        # SIDEBAR
        # ══════════════════════════════════════════════════════
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setMinimumWidth(210)
        sidebar.setMaximumWidth(300)
        sb_lyt = QVBoxLayout(sidebar)
        sb_lyt.setContentsMargins(0, 0, 0, 0)
        sb_lyt.setSpacing(0)

        # Logo
        # Logo — no icon; pure text scales at any font size
        logo_frame = QFrame()
        logo_frame.setObjectName("LogoFrame")
        logo_lyt = QVBoxLayout(logo_frame)
        logo_lyt.setContentsMargins(16, 14, 12, 12)
        logo_lyt.setSpacing(3)
        title_lbl = QLabel("Pollen Analysis Tool")
        title_lbl.setObjectName("LogoTitle")
        title_lbl.setWordWrap(True)
        sub_lbl = QLabel("Cellpose · Alexander Stain")
        sub_lbl.setObjectName("LogoSub")
        logo_lyt.addWidget(title_lbl)
        logo_lyt.addWidget(sub_lbl)
        logo_lyt.addStretch()
        sb_lyt.addWidget(logo_frame)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("SidebarSep")
        sb_lyt.addWidget(sep)

        # Nav
        nav_scroll = QScrollArea()
        nav_scroll.setWidgetResizable(True)
        nav_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        nav_inner = QWidget()
        nav_lyt = QVBoxLayout(nav_inner)
        nav_lyt.setContentsMargins(8, 8, 8, 8)
        nav_lyt.setSpacing(3)
        nav_scroll.setWidget(nav_inner)

        def nav_section(text):
            lbl = QLabel(text)
            lbl.setObjectName("NavSection")
            nav_lyt.addWidget(lbl)

        def nav_btn(icon, label, index):
            btn = QPushButton(f"{icon}  {label}")
            btn.setObjectName("NavBtn")
            btn.setCheckable(True)
            btn.setProperty("navIndex", index)
            btn.setFixedHeight(36)
            btn.clicked.connect(lambda _, i=index: self._switch_page(i))
            nav_lyt.addWidget(btn)
            return btn

        nav_section("WORKFLOW")
        nb0 = nav_btn("⬡", "Data Input",         0)
        nb1 = nav_btn("⚙", "Parameters",          1)
        nb2 = nav_btn("🔬", "Segment & Validate", 2)
        nb3 = nav_btn("🧬", "Train Custom Model",  3)
        nb4 = nav_btn("📊", "Results & Export",    4)
        nav_section("VIEWER")
        nb5 = nav_btn("🖼", "Validation Editor",   5)
        nb6 = nav_btn("📈", "Results Dashboard",   6)

        nav_lyt.addStretch()
        sb_lyt.addWidget(nav_scroll, 1)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setObjectName("SidebarSep")
        sb_lyt.addWidget(sep2)

        # ── Font-size slider ──────────────────────────────────────────────────
        font_panel = QFrame()
        font_panel.setObjectName("FontScalePanel")
        fp_lyt = QVBoxLayout(font_panel)
        fp_lyt.setContentsMargins(14, 6, 14, 4)
        fp_lyt.setSpacing(3)

        fp_top = QHBoxLayout()
        fp_top.setContentsMargins(0, 0, 0, 0)
        fp_lbl = QLabel("TEXT SIZE")
        fp_lbl.setObjectName("FontScaleLabel")
        self._font_scale_value_lbl = QLabel(f"{int(self._font_scale * 100)}%")
        self._font_scale_value_lbl.setObjectName("FontScaleValue")
        self._font_scale_value_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        fp_top.addWidget(fp_lbl)
        fp_top.addStretch()
        fp_top.addWidget(self._font_scale_value_lbl)
        fp_lyt.addLayout(fp_top)

        self._font_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self._font_scale_slider.setObjectName("FontScaleSlider")
        self._font_scale_slider.setRange(70, 200)          # 70% – 200%
        self._font_scale_slider.setValue(int(self._font_scale * 100))
        self._font_scale_slider.setSingleStep(5)
        self._font_scale_slider.setPageStep(10)
        self._font_scale_slider.setFixedHeight(22)
        self._font_scale_slider.valueChanged.connect(self._on_font_scale_changed)
        fp_lyt.addWidget(self._font_scale_slider)
        sb_lyt.addWidget(font_panel)
        # ─────────────────────────────────────────────────────────────────────

        sep_theme = QFrame(); sep_theme.setFrameShape(QFrame.Shape.HLine)
        sep_theme.setObjectName("SidebarSep")
        sb_lyt.addWidget(sep_theme)

        # ── Theme switcher ────────────────────────────────────────────────────
        theme_panel = QFrame()
        theme_panel.setObjectName("FontScalePanel")
        tp_lyt = QVBoxLayout(theme_panel)
        tp_lyt.setContentsMargins(14, 5, 14, 5)
        tp_lyt.setSpacing(4)
        tp_top = QHBoxLayout()
        tp_top.setContentsMargins(0, 0, 0, 0)
        tp_lbl = QLabel("COLOUR THEME")
        tp_lbl.setObjectName("FontScaleLabel")
        tp_top.addWidget(tp_lbl)
        tp_lyt.addLayout(tp_top)
        self._theme_combo = QComboBox()
        self._theme_combo.addItems([
            "Forest (default)",
            "Linen & Spice",
            "Oat & Moss",
            "Ivory & Ember",
            "Moss & Ochre",
            "Linen & Forest",
        ])
        self._theme_combo.setObjectName("ThemeCombo")
        self._theme_combo.currentTextChanged.connect(self._on_theme_changed)
        tp_lyt.addWidget(self._theme_combo)
        sb_lyt.addWidget(theme_panel)
        # ─────────────────────────────────────────────────────────────────────

        sep3 = QFrame(); sep3.setFrameShape(QFrame.Shape.HLine)
        sep3.setObjectName("SidebarSep")
        sb_lyt.addWidget(sep3)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("SidebarStatus")
        self.status_label.setWordWrap(True)
        sb_lyt.addWidget(self.status_label)

        shell.addWidget(sidebar)

        self._nav_buttons = [nb0, nb1, nb2, nb3, nb4, nb5, nb6]

        # ══════════════════════════════════════════════════════
        # CONTENT STACK
        # ══════════════════════════════════════════════════════
        self._stack = QStackedWidget()
        self._stack.setObjectName("ContentStack")
        shell.addWidget(self._stack)
        shell.setStretchFactor(0, 0)
        shell.setStretchFactor(1, 1)
        shell.setSizes([240, 9999])

        # ─────────────────────────────────────────────────────
        # PAGE 0 — DATA INPUT
        # ─────────────────────────────────────────────────────
        p0 = QWidget()
        p0_lyt = QVBoxLayout(p0)
        p0_lyt.setContentsMargins(0, 0, 0, 0)
        p0_lyt.setSpacing(0)
        p0_lyt.addWidget(self._page_header("Data Input", "Load single images or a batch folder"))

        p0_scroll = QScrollArea()
        p0_scroll.setWidgetResizable(True)
        p0_inner = QWidget()
        p0_inner_lyt = QVBoxLayout(p0_inner)
        p0_inner_lyt.setContentsMargins(28, 24, 28, 28)
        p0_inner_lyt.setSpacing(14)

        # Single image card
        single_card, single_lyt = self._make_card("Single Image")
        self.load_btn = QPushButton("📷  Load Image File")
        self.load_btn.setObjectName("ActionBtn")
        self.load_btn.setFixedHeight(38)
        self.load_btn.clicked.connect(self.load_image)
        single_lyt.addWidget(self.load_btn)
        p0_inner_lyt.addWidget(single_card)

        # Batch card
        batch_card, batch_lyt = self._make_card("Batch Processing")
        self.load_batch_btn = QPushButton("📂  Load Batch Folder")
        self.load_batch_btn.setObjectName("ActionBtn")
        self.load_batch_btn.setFixedHeight(42)
        self.load_batch_btn.clicked.connect(self.load_batch_folder)
        batch_lyt.addWidget(self.load_batch_btn)

        order_lbl = QLabel("Sample plot order — drag to rearrange:")
        order_lbl.setObjectName("HintLabel")
        batch_lyt.addWidget(order_lbl)
        self.sample_order_list = QListWidget()
        self.sample_order_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.sample_order_list.setFixedHeight(110)
        batch_lyt.addWidget(self.sample_order_list)

        batch_files_lbl = QLabel("Files in batch:")
        batch_files_lbl.setObjectName("HintLabel")
        batch_lyt.addWidget(batch_files_lbl)
        self.batch_list = QListWidget()
        self.batch_list.setMinimumHeight(160)
        batch_lyt.addWidget(self.batch_list)
        p0_inner_lyt.addWidget(batch_card)

        # Output directory card
        out_card, out_lyt = self._make_card("Output Directory")
        self.btn_output_dir = QPushButton("💾  Select Output Folder")
        self.btn_output_dir.setObjectName("SecondaryBtn")
        self.btn_output_dir.setFixedHeight(38)
        self.btn_output_dir.clicked.connect(self.set_output_folder)
        out_lyt.addWidget(self.btn_output_dir)
        self.lbl_output_dir = QLabel("Default: same folder as input images")
        self.lbl_output_dir.setObjectName("HintLabel")
        out_lyt.addWidget(self.lbl_output_dir)
        p0_inner_lyt.addWidget(out_card)
        p0_inner_lyt.addStretch()

        p0_scroll.setWidget(p0_inner)
        p0_lyt.addWidget(p0_scroll)
        self._stack.addWidget(p0)

        # ─────────────────────────────────────────────────────
        # PAGE 1 — PARAMETERS
        # ─────────────────────────────────────────────────────
        p1 = QWidget()
        p1_lyt = QVBoxLayout(p1)
        p1_lyt.setContentsMargins(0, 0, 0, 0)
        p1_lyt.setSpacing(0)
        p1_lyt.addWidget(self._page_header("Parameters", "Model & segmentation settings"))

        p1_scroll = QScrollArea()
        p1_scroll.setWidgetResizable(True)
        p1_inner = QWidget()
        p1_inner_lyt = QVBoxLayout(p1_inner)
        p1_inner_lyt.setContentsMargins(28, 24, 28, 28)
        p1_inner_lyt.setSpacing(14)

        # Model card
        model_card, model_lyt = self._make_card("Model Selection")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["cellpose", "cellpose-sam"])
        if model_cpsam is not None:
            self.model_combo.setCurrentText("cellpose-sam")
        self.model_combo.currentTextChanged.connect(self.change_model)
        model_lyt.addLayout(self._form_row("Segmentation model:", self.model_combo))

        self.channel_combo = QComboBox()
        self.channel_combo.addItems(list(CHANNEL_OPTIONS.keys()))
        self.channel_combo.setCurrentText(DEFAULT_CHANNEL)
        self.channel_combo.setToolTip("RGB passes all channels. Red-Minus-Green extracts maximum viability contrast.")
        model_lyt.addLayout(self._form_row("Image channel:", self.channel_combo))

        custom_row = QHBoxLayout()
        self.custom_model_btn = QPushButton("📁  Load Custom Model")
        self.custom_model_btn.setObjectName("SecondaryBtn")
        self.custom_model_btn.setFixedHeight(36)
        self.custom_model_btn.clicked.connect(self.load_custom_model)
        self.custom_model_label = QLabel("No custom model loaded")
        self.custom_model_label.setObjectName("HintLabel")
        custom_row.addWidget(self.custom_model_btn)
        custom_row.addWidget(self.custom_model_label, 1)
        model_lyt.addLayout(custom_row)
        p1_inner_lyt.addWidget(model_card)

        # Diameter card
        diam_card, diam_lyt = self._make_card("Grain Diameter")
        diam_group = QButtonGroup(self)
        self.auto_diameter = QRadioButton("Auto-detect diameter")
        self.manual_diameter = QRadioButton("Manual diameter (px):")
        self.manual_diameter.setChecked(True)
        diam_group.addButton(self.auto_diameter)
        diam_group.addButton(self.manual_diameter)
        diam_lyt.addWidget(self.auto_diameter)
        diam_row = QHBoxLayout()
        diam_row.addWidget(self.manual_diameter)
        self.diameter_spin = QDoubleSpinBox()
        self.diameter_spin.setRange(0, 500)
        self.diameter_spin.setValue(40)
        self.diameter_spin.setEnabled(True)
        self.diameter_spin.setFixedWidth(110)
        self.diameter_spin.setToolTip("Set to typical pixel diameter to help the model filter out small debris.")
        self.manual_diameter.toggled.connect(lambda: self.diameter_spin.setEnabled(self.manual_diameter.isChecked()))
        diam_row.addWidget(self.diameter_spin)
        diam_row.addStretch()
        diam_lyt.addLayout(diam_row)
        p1_inner_lyt.addWidget(diam_card)

        # Advanced card (always visible, collapsible content)
        adv_card, adv_outer_lyt = self._make_card("Advanced Settings")
        self.adv_btn = QPushButton("▶  Show Advanced Settings")
        self.adv_btn.setObjectName("SecondaryBtn")
        self.adv_btn.setCheckable(True)
        self.adv_btn.setFixedHeight(36)
        self.adv_btn.toggled.connect(self.toggle_advanced_settings)
        adv_outer_lyt.addWidget(self.adv_btn)

        self.adv_widget = QWidget()
        adv_layout = QVBoxLayout(self.adv_widget)
        adv_layout.setContentsMargins(0, 4, 0, 0)
        adv_layout.setSpacing(8)

        self.resize_spin = QSpinBox()
        self.resize_spin.setRange(100, 4000)
        self.resize_spin.setValue(1980)
        self.resize_spin.setToolTip("Scales longest edge to this size. Larger = finer detail but slower and uses more RAM.")
        adv_layout.addLayout(self._form_row("Max resize (px):", self.resize_spin))

        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(0, 2000)
        self.iter_spin.setValue(500)
        self.iter_spin.setToolTip("Gradient descent steps. Higher values (1000+) help separate tightly clumped grains.")
        adv_layout.addLayout(self._form_row("Max iterations:", self.iter_spin))

        self.flow_spin = QDoubleSpinBox()
        self.flow_spin.setRange(0.0, 1.0)
        self.flow_spin.setSingleStep(0.05)
        self.flow_spin.setValue(0.3)
        self.flow_spin.setToolTip("Strictness of grain shape. Lower = keeps more masks. Raise to discard irregular shapes (debris).")
        adv_layout.addLayout(self._form_row("Flow threshold:", self.flow_spin))

        self.cellprob_spin = QDoubleSpinBox()
        self.cellprob_spin.setRange(-6.0, 6.0)
        self.cellprob_spin.setSingleStep(0.5)
        self.cellprob_spin.setValue(0.0)
        self.cellprob_spin.setToolTip("Signal strength needed to seed a cell. Lower = more permissive (finds faint grains).")
        adv_layout.addLayout(self._form_row("Cellprob threshold:", self.cellprob_spin))

        self.norm_low_spin = QDoubleSpinBox()
        self.norm_low_spin.setRange(0.0, 100.0)
        self.norm_low_spin.setValue(1.0)
        self.norm_low_spin.setToolTip("Auto-contrast black point (percentile).")
        adv_layout.addLayout(self._form_row("Norm lower %:", self.norm_low_spin))

        self.norm_up_spin = QDoubleSpinBox()
        self.norm_up_spin.setRange(0.0, 100.0)
        self.norm_up_spin.setValue(99.0)
        self.norm_up_spin.setToolTip("Auto-contrast white point (percentile).")
        adv_layout.addLayout(self._form_row("Norm upper %:", self.norm_up_spin))

        self.adv_widget.setVisible(False)
        adv_outer_lyt.addWidget(self.adv_widget)
        p1_inner_lyt.addWidget(adv_card)
        p1_inner_lyt.addStretch()

        p1_scroll.setWidget(p1_inner)
        p1_lyt.addWidget(p1_scroll)
        self._stack.addWidget(p1)

        # ─────────────────────────────────────────────────────
        # PAGE 2 — SEGMENT & VALIDATE
        # ─────────────────────────────────────────────────────
        p2 = QWidget()
        p2_lyt = QVBoxLayout(p2)
        p2_lyt.setContentsMargins(0, 0, 0, 0)
        p2_lyt.setSpacing(0)
        p2_lyt.addWidget(self._page_header("Segment & Validate", "Run segmentation and review masks"))

        p2_scroll = QScrollArea()
        p2_scroll.setWidgetResizable(True)
        p2_inner = QWidget()
        p2_inner_lyt = QVBoxLayout(p2_inner)
        p2_inner_lyt.setContentsMargins(28, 24, 28, 28)
        p2_inner_lyt.setSpacing(14)

        # ── Step 1: Segmentation card ────────────────────────────────────────
        seg_card, seg_lyt = self._make_card("Step 1 — Segmentation")

        self.segment_single_btn = QPushButton("🔬  Segment Current Image")
        self.segment_single_btn.setObjectName("ActionBtn")
        self.segment_single_btn.setFixedHeight(42)
        self.segment_single_btn.clicked.connect(self.run_segmentation)
        seg_lyt.addWidget(self.segment_single_btn)

        self.segment_batch_btn = QPushButton("🔬  Segment Batch for Validation")
        self.segment_batch_btn.setObjectName("PrimaryBtn")
        self.segment_batch_btn.setFixedHeight(42)
        self.segment_batch_btn.setToolTip("Segments all batch images and opens the Validation Editor so you can review and correct masks before counting.")
        self.segment_batch_btn.clicked.connect(lambda: self.run_batch_analysis(autorun=False))
        seg_lyt.addWidget(self.segment_batch_btn)

        autorun_hint = QLabel("Skip validation — segments and counts immediately:")
        autorun_hint.setObjectName("HintLabel")
        seg_lyt.addWidget(autorun_hint)
        self.autorun_btn = QPushButton("🚀  Segment + Count (No Validation)")
        self.autorun_btn.setObjectName("SecondaryBtn")
        self.autorun_btn.setFixedHeight(38)
        self.autorun_btn.setToolTip("Re-segments all batch images from scratch and counts immediately — skips the validation step entirely.")
        self.autorun_btn.clicked.connect(lambda: self.run_batch_analysis(autorun=True))
        seg_lyt.addWidget(self.autorun_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        seg_lyt.addWidget(self.progress_bar)

        stop_row = QHBoxLayout()
        self.stop_btn = QPushButton("■  Stop Processing")
        self.stop_btn.setObjectName("DangerBtn")
        self.stop_btn.setFixedHeight(36)
        self.stop_btn.clicked.connect(self.stop_processing)
        stop_row.addWidget(self.stop_btn)
        stop_row.addStretch()
        seg_lyt.addLayout(stop_row)
        p2_inner_lyt.addWidget(seg_card)

        # ── Step 2: Count after validation card ──────────────────────────────
        count_card, count_lyt = self._make_card("Step 2 — Count Validated Images")

        self.validated_count_lbl = QLabel("No validated images yet.")
        self.validated_count_lbl.setObjectName("HintLabel")
        count_lyt.addWidget(self.validated_count_lbl)

        self.count_validated_btn = QPushButton("📊  Count & Generate Report from Validated Images")
        self.count_validated_btn.setObjectName("PrimaryBtn")
        self.count_validated_btn.setFixedHeight(46)
        self.count_validated_btn.setToolTip("Uses the masks you have already validated — no re-segmentation.")
        self.count_validated_btn.setEnabled(False)
        self.count_validated_btn.clicked.connect(self._count_validated)
        count_lyt.addWidget(self.count_validated_btn)

        p2_inner_lyt.addWidget(count_card)

        # ── Step 3: Save single-image results card ────────────────────────────
        save_card, save_lyt = self._make_card("Step 3 — Save Current Image (optional)")
        save_hint = QLabel("Save the current image mask, overlay, outlines and per-grain CSV.")
        save_hint.setObjectName("HintLabel")
        save_lyt.addWidget(save_hint)
        self.save_btn = QPushButton("💾  Save Current Image Results")
        self.save_btn.setObjectName("SecondaryBtn")
        self.save_btn.setFixedHeight(38)
        self.save_btn.clicked.connect(self.save_results)
        save_lyt.addWidget(self.save_btn)
        p2_inner_lyt.addWidget(save_card)

        # Hidden state-bearing widgets — these are never shown on page 2 but are
        # kept alive so that signal sync logic between page 2 and the page 5
        # toolbar (opacity_slider_v, contrast_slider_v, etc.) can use blockSignals
        # without crashing. All user-visible controls live in the toolbar on p5.
        self.btn_prev = QPushButton()
        self.btn_prev.clicked.connect(self.nav_prev)
        self.btn_next = QPushButton()
        self.btn_next.clicked.connect(self.nav_next)
        self.lbl_nav = QLabel("Image 0 / 0")
        self.draw_mode_combo = QComboBox()
        self.draw_mode_combo.addItems(["Ellipse", "Freehand Polygon"])
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(10, 300)
        self.contrast_slider.setValue(100)
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        # None of the above are added to p2_inner_lyt — all visible controls live in the toolbar

        p2_inner_lyt.addStretch()

        p2_scroll.setWidget(p2_inner)
        p2_lyt.addWidget(p2_scroll)
        self._stack.addWidget(p2)

        # ─────────────────────────────────────────────────────
        # PAGE 3 — TRAIN CUSTOM MODEL
        # ─────────────────────────────────────────────────────
        p3 = QWidget()
        p3_lyt = QVBoxLayout(p3)
        p3_lyt.setContentsMargins(0, 0, 0, 0)
        p3_lyt.setSpacing(0)
        p3_lyt.addWidget(self._page_header("Train Custom Model", "Fine-tune Cellpose on your corrected masks"))

        p3_scroll = QScrollArea()
        p3_scroll.setWidgetResizable(True)
        p3_inner = QWidget()
        p3_inner_lyt = QVBoxLayout(p3_inner)
        p3_inner_lyt.setContentsMargins(28, 24, 28, 28)
        p3_inner_lyt.setSpacing(14)

        # Instructions card
        info_card, info_lyt = self._make_card("How to Train")
        steps_lbl = QLabel(
            "1. Load your images and run segmentation.\n"
            "2. Review and correct masks in the Validation Editor.\n"
            "3. Set training parameters below and click Train.\n\n"
            "Training always fine-tunes the model that is currently active in Parameters.\n"
            "If you loaded a custom model, that model is what gets trained further."
        )
        steps_lbl.setObjectName("HintLabel")
        steps_lbl.setWordWrap(True)
        info_lyt.addWidget(steps_lbl)

        # Live indicator — shows which model will be trained
        self.train_base_model_lbl = QLabel()
        self.train_base_model_lbl.setObjectName("TrainBaseLabel")
        self.train_base_model_lbl.setWordWrap(True)
        info_lyt.addWidget(self.train_base_model_lbl)
        p3_inner_lyt.addWidget(info_card)

        # Training settings card
        train_card, train_lyt = self._make_card("Training Settings")
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 1000)
        self.epochs_spin.setValue(100)
        train_lyt.addLayout(self._form_row("Epochs:", self.epochs_spin))

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.001, 1.0)
        self.lr_spin.setSingleStep(0.01)
        self.lr_spin.setValue(0.1)
        train_lyt.addLayout(self._form_row("Learning rate:", self.lr_spin))

        self.model_name_input = QLineEdit()
        self.model_name_input.setText("custom_pollen_model")
        train_lyt.addLayout(self._form_row("Model name:", self.model_name_input))

        self.start_train_btn = QPushButton("🧬  Start Training")
        self.start_train_btn.setObjectName("PrimaryBtn")
        self.start_train_btn.setFixedHeight(46)
        self.start_train_btn.clicked.connect(self.start_training)
        train_lyt.addWidget(self.start_train_btn)
        p3_inner_lyt.addWidget(train_card)

        # Log card
        log_card, log_lyt = self._make_card("Training Log")
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setMinimumHeight(200)
        self.train_log.setObjectName("TrainLog")
        self.train_log.setFontFamily("Courier New, Courier, monospace")
        log_lyt.addWidget(self.train_log)
        p3_inner_lyt.addWidget(log_card)
        p3_inner_lyt.addStretch()

        p3_scroll.setWidget(p3_inner)
        p3_lyt.addWidget(p3_scroll)
        self._stack.addWidget(p3)

        # ─────────────────────────────────────────────────────
        # PAGE 4 — RESULTS & EXPORT
        # ─────────────────────────────────────────────────────
        p4 = QWidget()
        p4_lyt = QVBoxLayout(p4)
        p4_lyt.setContentsMargins(0, 0, 0, 0)
        p4_lyt.setSpacing(0)
        p4_lyt.addWidget(self._page_header("Results & Export", "Generate statistics, plots, and PDF report"))

        p4_scroll = QScrollArea()
        p4_scroll.setWidgetResizable(True)
        p4_inner = QWidget()
        p4_inner_lyt = QVBoxLayout(p4_inner)
        p4_inner_lyt.setContentsMargins(28, 24, 28, 28)
        p4_inner_lyt.setSpacing(14)

        export_card, export_lyt = self._make_card("Export Actions")
        self.analyze_btn = QPushButton("📊  Generate Statistics & Export Everything")
        self.analyze_btn.setObjectName("PrimaryBtn")
        self.analyze_btn.setFixedHeight(46)
        self.analyze_btn.setToolTip(
            "Counts validated images, generates statistics, saves CSV, box-plot (PNG + PDF), "
            "PDF report, masks, overlays and outlines to the output folder."
        )
        self.analyze_btn.clicked.connect(self.analyze_pollen)
        export_lyt.addWidget(self.analyze_btn)

        self.save_int_cb = QCheckBox("Save intermediate images (masks, overlays, outlines)")
        self.save_int_cb.setObjectName("OptionCheck")
        self.save_int_cb.setChecked(True)   # on by default so everything is saved
        export_lyt.addWidget(self.save_int_cb)
        p4_inner_lyt.addWidget(export_card)

        # Results summary text
        res_card, res_lyt = self._make_card("Analysis Output")
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(200)
        self.results_text.setObjectName("ResultsLog")
        self.results_text.setFontFamily("Courier New, Courier, monospace")
        res_lyt.addWidget(self.results_text)
        p4_inner_lyt.addWidget(res_card)

        jump_res_btn = QPushButton("📈  Open Results Dashboard →")
        jump_res_btn.setObjectName("ActionBtn")
        jump_res_btn.setFixedHeight(42)
        jump_res_btn.clicked.connect(lambda: self._switch_page(6))
        p4_inner_lyt.addWidget(jump_res_btn)
        p4_inner_lyt.addStretch()

        p4_scroll.setWidget(p4_inner)
        p4_lyt.addWidget(p4_scroll)
        self._stack.addWidget(p4)

        # ─────────────────────────────────────────────────────
        # PAGE 5 — VALIDATION EDITOR (full-screen image viewer)
        # ─────────────────────────────────────────────────────
        p5 = QWidget()
        p5_lyt = QVBoxLayout(p5)
        p5_lyt.setContentsMargins(0, 0, 0, 0)
        p5_lyt.setSpacing(0)
        p5_lyt.addWidget(self._page_header("Validation Editor", "Wheel: zoom  ·  Mid-click: pan  ·  Right-click drag: add mask  ·  Ctrl+click: remove mask"))

        # ── Compact toolbar ──────────────────────────────────
        toolbar = QFrame()
        toolbar.setObjectName("Card")
        toolbar.setMinimumHeight(46)
        tb_lyt = QHBoxLayout(toolbar)
        tb_lyt.setContentsMargins(12, 6, 12, 6)
        tb_lyt.setSpacing(8)

        # Prev / counter / Next
        self.btn_prev_v = QPushButton("◀  Prev")
        self.btn_prev_v.setObjectName("SecondaryBtn")
        self.btn_prev_v.setFixedHeight(38)
        self.btn_prev_v.clicked.connect(self.nav_prev)

        self.lbl_nav_v = QLabel("0 / 0")
        self.lbl_nav_v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_nav_v.setMinimumWidth(200)
        self.lbl_nav_v.setMaximumWidth(500)

        self.btn_next_v = QPushButton("Next  ▶")
        self.btn_next_v.setObjectName("SecondaryBtn")
        self.btn_next_v.setFixedHeight(38)
        self.btn_next_v.clicked.connect(self.nav_next)

        tb_lyt.addWidget(self.btn_prev_v)
        tb_lyt.addWidget(self.lbl_nav_v)
        tb_lyt.addWidget(self.btn_next_v)

        sep_v = QFrame()
        sep_v.setFrameShape(QFrame.Shape.VLine)
        sep_v.setObjectName("SidebarSep")
        sep_v.setFixedWidth(1)
        tb_lyt.addWidget(sep_v)

        # Draw mode
        draw_lbl_v = QLabel("Draw:")
        draw_lbl_v.setObjectName("FormLabel")
        self.draw_mode_combo_v = QComboBox()
        self.draw_mode_combo_v.addItems(["Ellipse", "Freehand Polygon"])
        self.draw_mode_combo_v.setFixedWidth(155)
        tb_lyt.addWidget(draw_lbl_v)
        tb_lyt.addWidget(self.draw_mode_combo_v)

        sep_v2 = QFrame()
        sep_v2.setFrameShape(QFrame.Shape.VLine)
        sep_v2.setObjectName("SidebarSep")
        sep_v2.setFixedWidth(1)
        tb_lyt.addWidget(sep_v2)

        # Opacity
        op_lbl_v = QLabel("Opacity:")
        op_lbl_v.setObjectName("FormLabel")
        self.opacity_slider_v = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider_v.setRange(0, 100)
        self.opacity_slider_v.setValue(50)
        self.opacity_slider_v.setFixedWidth(90)
        tb_lyt.addWidget(op_lbl_v)
        tb_lyt.addWidget(self.opacity_slider_v)

        sep_v3 = QFrame()
        sep_v3.setFrameShape(QFrame.Shape.VLine)
        sep_v3.setObjectName("SidebarSep")
        sep_v3.setFixedWidth(1)
        tb_lyt.addWidget(sep_v3)

        # Contrast & Brightness
        ct_lbl_v = QLabel("Contrast:")
        ct_lbl_v.setObjectName("FormLabel")
        self.contrast_slider_v = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider_v.setRange(10, 300)
        self.contrast_slider_v.setValue(100)
        self.contrast_slider_v.setFixedWidth(85)

        br_lbl_v = QLabel("Brightness:")
        br_lbl_v.setObjectName("FormLabel")
        self.brightness_slider_v = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider_v.setRange(-100, 100)
        self.brightness_slider_v.setValue(0)
        self.brightness_slider_v.setFixedWidth(85)

        tb_lyt.addWidget(ct_lbl_v)
        tb_lyt.addWidget(self.contrast_slider_v)
        tb_lyt.addWidget(br_lbl_v)
        tb_lyt.addWidget(self.brightness_slider_v)

        tb_lyt.addStretch()

        count_btn_v = QPushButton("📊  Count & Report")
        count_btn_v.setObjectName("PrimaryBtn")
        count_btn_v.setFixedHeight(36)
        count_btn_v.setToolTip("Count all validated images and generate report — no re-segmentation.")
        count_btn_v.clicked.connect(self._count_validated)
        tb_lyt.addWidget(count_btn_v)

        back_btn = QPushButton("←  Workflow")
        back_btn.setObjectName("ActionBtn")
        back_btn.setFixedHeight(36)
        back_btn.clicked.connect(lambda: self._switch_page(2))
        tb_lyt.addWidget(back_btn)

        p5_lyt.addWidget(toolbar)
        # ── end toolbar ──────────────────────────────────────

        views_splitter = QSplitter(Qt.Orientation.Horizontal)
        views_splitter.setObjectName("RootSplitter")

        self.view_orig = ZoomGraphicsView()
        views_splitter.addWidget(self.view_orig)

        self.view_overlay = InteractiveGraphicsView()
        self.view_overlay.masks_updated.connect(self.on_masks_updated)
        self.view_overlay.action_logged.connect(self.status_label.setText)
        views_splitter.addWidget(self.view_overlay)

        views_splitter.setSizes([600, 600])
        p5_lyt.addWidget(views_splitter, 1)
        self._stack.addWidget(p5)

        # ─────────────────────────────────────────────────────
        # PAGE 6 — RESULTS DASHBOARD
        # ─────────────────────────────────────────────────────
        p6 = QWidget()
        p6_lyt = QVBoxLayout(p6)
        p6_lyt.setContentsMargins(0, 0, 0, 0)
        p6_lyt.setSpacing(0)
        p6_lyt.addWidget(self._page_header("Results Dashboard", "Summary statistics and publication plot"))

        # ── Results Dashboard uses a horizontal splitter: controls left, plot right ──
        p6_splitter = QSplitter(Qt.Orientation.Horizontal)
        p6_splitter.setObjectName("RootSplitter")

        # ── Left panel: table + rename/reorder + plot options ──────────────────
        p6_left = QWidget()
        p6_left_lyt = QVBoxLayout(p6_left)
        p6_left_lyt.setContentsMargins(16, 16, 16, 16)
        p6_left_lyt.setSpacing(10)

        # Stats table
        tbl_lbl = QLabel("SUMMARY STATISTICS")
        tbl_lbl.setObjectName("CardTitle")
        p6_left_lyt.addWidget(tbl_lbl)
        self.res_table = QTableWidget()
        self.res_table.setColumnCount(5)
        self.res_table.setHorizontalHeaderLabels(["Sample", "n", "Mean", "Median", "SD"])
        self.res_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.res_table.setAlternatingRowColors(True)
        self.res_table.setFixedHeight(160)
        p6_left_lyt.addWidget(self.res_table)

        # Rename / reorder card
        reorder_lbl = QLabel("SAMPLE ORDER & LABELS")
        reorder_lbl.setObjectName("CardTitle")
        p6_left_lyt.addWidget(reorder_lbl)

        hint_reorder = QLabel("Drag to reorder  ·  Double-click to rename")
        hint_reorder.setObjectName("HintLabel")
        p6_left_lyt.addWidget(hint_reorder)

        self.plot_order_list = QListWidget()
        self.plot_order_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.plot_order_list.setMinimumHeight(200)
        self.plot_order_list.itemDoubleClicked.connect(self._rename_sample_item)
        p6_left_lyt.addWidget(self.plot_order_list)

        # Plot options card
        opt_lbl = QLabel("PLOT OPTIONS")
        opt_lbl.setObjectName("CardTitle")
        p6_left_lyt.addWidget(opt_lbl)

        self.plot_title_edit = QLineEdit()
        self.plot_title_edit.setPlaceholderText("Plot title (leave blank for default)")
        p6_left_lyt.addWidget(self.plot_title_edit)

        self.plot_ylabel_edit = QLineEdit()
        self.plot_ylabel_edit.setPlaceholderText("Y-axis label (leave blank for default)")
        p6_left_lyt.addWidget(self.plot_ylabel_edit)

        pt_row = QHBoxLayout()
        pt_lbl = QLabel("Point size:")
        pt_lbl.setObjectName("FormLabel")
        pt_lbl.setFixedWidth(80)
        self.plot_pt_spin = QSpinBox()
        self.plot_pt_spin.setRange(4, 80)
        self.plot_pt_spin.setValue(28)
        pt_row.addWidget(pt_lbl)
        pt_row.addWidget(self.plot_pt_spin)
        pt_row.addStretch()
        p6_left_lyt.addLayout(pt_row)

        self.plot_jitter_cb = QCheckBox("Show jittered data points")
        self.plot_jitter_cb.setChecked(True)
        p6_left_lyt.addWidget(self.plot_jitter_cb)

        self.plot_mean_cb = QCheckBox("Show mean ± SD overlay")
        self.plot_mean_cb.setChecked(True)
        p6_left_lyt.addWidget(self.plot_mean_cb)

        redraw_btn = QPushButton("🎨  Redraw Plot")
        redraw_btn.setObjectName("PrimaryBtn")
        redraw_btn.setFixedHeight(40)
        redraw_btn.clicked.connect(self._redraw_plot)
        p6_left_lyt.addWidget(redraw_btn)

        p6_left_lyt.addStretch()
        p6_splitter.addWidget(p6_left)

        # ── Right panel: matplotlib canvas ─────────────────────────────────────
        self.figure_canvas = FigureCanvas(plt.Figure(figsize=(6, 4), facecolor="white"))
        p6_splitter.addWidget(self.figure_canvas)
        self.figure_splitter = p6_splitter   # keep reference for canvas hot-swap

        p6_splitter.setSizes([280, 9999])
        p6_lyt.addWidget(p6_splitter, 1)
        self._stack.addWidget(p6)

        # ── backward-compat proxy objects — retained for safety but no longer
        # used internally. All navigation goes through self._switch_page() directly.
        class _ToolboxProxy:
            def __init__(self_, stack, nav_btns, switch_fn):
                self_._switch = switch_fn
            def setCurrentIndex(self_, idx):
                self_._switch(idx)
        self.toolbox = _ToolboxProxy(self._stack, self._nav_buttons, self._switch_page)

        class _RightTabsProxy:
            def __init__(self_, switch_fn):
                self_._switch = switch_fn
            def setCurrentIndex(self_, idx):
                self_._switch(5 + idx)   # 0→5 (Validation Editor), 1→6 (Results Dashboard)
        self.right_tabs = _RightTabsProxy(self._switch_page)

        # ── Sync scrollbars and zoom ──────────────────────────────────────────
        self.view_orig.verticalScrollBar().valueChanged.connect(self.view_overlay.verticalScrollBar().setValue)
        self.view_overlay.verticalScrollBar().valueChanged.connect(self.view_orig.verticalScrollBar().setValue)
        self.view_orig.horizontalScrollBar().valueChanged.connect(self.view_overlay.horizontalScrollBar().setValue)
        self.view_overlay.horizontalScrollBar().valueChanged.connect(self.view_orig.horizontalScrollBar().setValue)
        self.view_orig.zoom_applied.connect(self.view_overlay.apply_zoom)
        self.view_overlay.zoom_applied.connect(self.view_orig.apply_zoom)

        # Connect internal handlers
        self.opacity_slider.valueChanged.connect(self.view_overlay.set_mask_opacity)
        self.draw_mode_combo.currentTextChanged.connect(self.view_overlay.set_draw_mode)

        # ── Sync toolbar controls (page 5) with page 2 controls ──────────────
        # Use blockSignals to prevent feedback loops: A→B must not re-trigger A

        def _sync_opacity(val, source):
            target = self.opacity_slider if source == "v" else self.opacity_slider_v
            target.blockSignals(True)
            target.setValue(val)
            target.blockSignals(False)
            self.view_overlay.set_mask_opacity(val)

        self.opacity_slider_v.valueChanged.connect(lambda v: _sync_opacity(v, "v"))
        self.opacity_slider.valueChanged.connect(lambda v: _sync_opacity(v, "p2"))

        def _sync_contrast(val, source):
            target = self.contrast_slider if source == "v" else self.contrast_slider_v
            target.blockSignals(True)
            target.setValue(val)
            target.blockSignals(False)
            self.display_image()

        self.contrast_slider_v.valueChanged.connect(lambda v: _sync_contrast(v, "v"))
        self.contrast_slider.valueChanged.connect(lambda v: _sync_contrast(v, "p2"))

        def _sync_brightness(val, source):
            target = self.brightness_slider if source == "v" else self.brightness_slider_v
            target.blockSignals(True)
            target.setValue(val)
            target.blockSignals(False)
            self.display_image()

        self.brightness_slider_v.valueChanged.connect(lambda v: _sync_brightness(v, "v"))
        self.brightness_slider.valueChanged.connect(lambda v: _sync_brightness(v, "p2"))

        def _sync_draw_mode(text, source):
            target = self.draw_mode_combo if source == "v" else self.draw_mode_combo_v
            target.blockSignals(True)
            target.setCurrentText(text)
            target.blockSignals(False)
            self.view_overlay.set_draw_mode(text)

        self.draw_mode_combo_v.currentTextChanged.connect(lambda t: _sync_draw_mode(t, "v"))
        self.draw_mode_combo.currentTextChanged.connect(lambda t: _sync_draw_mode(t, "p2"))

        # Apply initial settings
        self.view_overlay.set_mask_opacity(self.opacity_slider.value())
        self.view_overlay.set_draw_mode(self.draw_mode_combo.currentText())

        # Initialize with cellpose model (or cellpose-SAM if available)
        self.change_model(self.model_combo.currentText())
        self._update_train_base_label()

        # Default page
        self._switch_page(0)

    def _on_theme_changed(self, theme_name):
        """Switch colour theme live without restarting."""
        self._theme = theme_name
        self.setStyleSheet(self._build_stylesheet(self._font_scale))

    def toggle_advanced_settings(self, checked):
        self.adv_btn.setText("▼  Hide Advanced Settings" if checked else "▶  Show Advanced Settings")
        self.adv_widget.setVisible(checked)

    def set_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_dir = folder
            self.lbl_output_dir.setText(f"Output: {folder}")

    def reset_enhancements(self):
        # blockSignals to prevent double display_image calls during reset
        for sl, val in [(self.opacity_slider, 50), (self.contrast_slider, 100), (self.brightness_slider, 0)]:
            sl.blockSignals(True)
            sl.setValue(val)
            sl.blockSignals(False)
        if hasattr(self, 'opacity_slider_v'):
            for sl, val in [(self.opacity_slider_v, 50), (self.contrast_slider_v, 100), (self.brightness_slider_v, 0)]:
                sl.blockSignals(True)
                sl.setValue(val)
                sl.blockSignals(False)
        self.view_overlay.set_mask_opacity(50)
        self.display_image()

    def on_masks_updated(self, new_masks):
        self.mask = new_masks
        if self.validation_entries and 0 <= self.current_idx < len(self.validation_entries):
            self.validation_entries[self.current_idx]['mask'] = new_masks
            self._refresh_validated_label()

    def stop_processing(self):
        self.stop_requested = True
        self.status_label.setText("Status: Stop requested — waiting for current image to finish…")
        for attr in ('_seg_thread', '_seg_batch_thread', '_analysis_thread'):
            thread = getattr(self, attr, None)
            if thread is not None and thread.isRunning():
                thread.quit()

    def _update_train_base_label(self):
        """Refresh the Training page indicator to show which model will be trained."""
        if not hasattr(self, 'train_base_model_lbl'):
            return
        name = getattr(self, '_active_model_display_name', 'cellpose (default)')
        self.train_base_model_lbl.setText(f"⚙  Base model for next training run:  {name}")

    def change_model(self, model_name):
        if model_name == "cellpose":
            self.current_model = model_cellpose
            self._active_model_display_name = "cellpose (built-in)"
        elif model_name == "cellpose-sam":
            self.current_model = model_cpsam if model_cpsam is not None else model_cellpose
            self._active_model_display_name = (
                "cellpose-SAM" if model_cpsam is not None
                else "cellpose (SAM unavailable, using built-in)"
            )
        elif model_name == "custom" and model_custom is not None:
            self.current_model = model_custom
            self._active_model_display_name = f"custom: {model_custom_name}"
        else:
            self.current_model = model_cellpose
            self._active_model_display_name = "cellpose (built-in)"
        self._update_train_base_label()

    def load_custom_model(self):
        global model_custom, model_custom_name
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Custom Model", "",
            "All Files (*);;Model Files (*.pt *.npy);;PyTorch (*.pt);;NumPy (*.npy)"
        )
        if file_path:
            try:
                model_custom = models.CellposeModel(
                    gpu=torch.cuda.is_available(), pretrained_model=file_path
                )
                model_custom_name = os.path.basename(file_path)
                self.custom_model_label.setText(f"Loaded: {model_custom_name}")
                self._active_model_display_name = f"custom: {model_custom_name}"

                # Insert into combo box if not present
                if "custom" not in [self.model_combo.itemText(i)
                                     for i in range(self.model_combo.count())]:
                    self.model_combo.addItem("custom")
                self.model_combo.setCurrentText("custom")
                self.current_model = model_custom
                self._update_train_base_label()
                self.status_label.setText(f"Status: Custom model loaded — {model_custom_name}")
            except Exception as e:
                self.status_label.setText(f"Status: Error loading custom model: {str(e)}")

    def load_batch_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Batch Folder")
        if folder_path:
            self.batch_folder_path = folder_path
            self.batch_files = []
            self.batch_list.clear()
            self.image = None
            self.image_display = None
            self.mask = None
            
            samples = set()

            # Find all image files in subfolders
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, folder_path)
                        self.batch_files.append((full_path, rel_path))
                        self.batch_list.addItem(rel_path)
                        
                        sname = os.path.basename(root)
                        if sname == "" or root == folder_path:
                            sname = "Sample"
                        samples.add(sname)

            self.sample_order_list.clear()
            def natural_key(s):
                return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]
            for s in sorted(list(samples), key=natural_key):
                self.sample_order_list.addItem(s)

            self.status_label.setText(f"Status: Loaded {len(self.batch_files)} images from {len(samples)} sample(s)")
            
            try:
                self.batch_list.itemDoubleClicked.disconnect()
            except Exception:
                pass
            self.batch_list.itemDoubleClicked.connect(self.load_batch_item)

    def load_batch_item(self, item):
        rel_path = item.text()
        for fp, rp in self.batch_files:
            if rp == rel_path:
                image = cv2.imread(fp)
                if image is not None:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.image = image_rgb
                    self.image_display = image_rgb
                    self.mask = None
                    self.validation_entries = [{'file': fp, 'image': image_rgb, 'mask': None}]
                    self.current_idx = 0
                    self.load_validation_image()
                    self._refresh_validated_label()
                    self._switch_page(2)
                    self.status_label.setText(f"Status: Loaded {rp} — click Segment Current Image.")
                else:
                    self.status_label.setText(f"Status: Failed to load image: {rp}")
                break

    def run_batch_analysis(self, autorun=False):
        if not self.batch_files:
            self.status_label.setText("Status: No batch files loaded")
            return

        self.stop_requested = False
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(self.batch_files))
        self.progress_bar.setValue(0)
        self.status_label.setText("Status: Loading images for batch segmentation...")
        QApplication.processEvents()

        self.validation_entries = []
        self._batch_full_paths = []

        resize_val    = self.resize_spin.value()
        channel_func  = CHANNEL_OPTIONS[self.channel_combo.currentText()]
        diameter      = 0 if self.auto_diameter.isChecked() else self.diameter_spin.value()
        flow_thresh   = self.flow_spin.value()
        cellprob      = self.cellprob_spin.value()
        max_iter      = self.iter_spin.value()
        norm_low      = self.norm_low_spin.value()
        norm_up       = self.norm_up_spin.value()

        params_list = []
        for full_path, rel_path in self.batch_files:
            image = cv2.imread(full_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self._batch_full_paths.append(full_path)
            params_list.append(dict(
                image=image,
                resize_val=resize_val,
                channel_func=channel_func,
                diameter=diameter,
                flow_thresh=flow_thresh,
                cellprob=cellprob,
                max_iter=max_iter,
                norm_low=norm_low,
                norm_up=norm_up,
            ))

        self._batch_autorun = autorun
        if hasattr(self, '_seg_batch_thread') and self._seg_batch_thread.isRunning():
            self._seg_batch_thread.quit()
            self._seg_batch_thread.wait(2000)
        self._seg_batch_thread = SegmentationThread(self.current_model, params_list)
        self._seg_batch_thread.progress.connect(
            lambda cur, tot: (self.progress_bar.setValue(cur),
                              self.status_label.setText(f"Status: Segmenting image {cur} / {tot}…"))
        )
        self._seg_batch_thread.finished.connect(self._on_batch_done)
        self._seg_batch_thread.error.connect(self._on_segmentation_error)
        self._seg_batch_thread.start()

    def _on_batch_done(self, results, _):
        self.progress_bar.setVisible(False)
        for idx, (masks_filtered, img_resized) in enumerate(results):
            full_path = self._batch_full_paths[idx] if idx < len(self._batch_full_paths) else "Unknown"
            self.validation_entries.append({
                'file': full_path,
                'image': img_resized,
                'mask': masks_filtered,
            })

        self._refresh_validated_label()

        if self._batch_autorun:
            self._switch_page(4)
            self.analyze_pollen()
        else:
            n = len(self.validation_entries)
            self.status_label.setText(
                f"Status: Segmented {n} image(s). Review masks in the Validation Editor, "
                f"then use 'Count & Generate Report' when done."
            )
            if self.validation_entries:
                self.current_idx = 0
                self.load_validation_image()
                self._switch_page(5)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff);;All Files (*)"
        )
        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.image = image
                self.image_display = image
                self.mask = None
                self.validation_entries = [{
                    'file': file_path,
                    'image': image,
                    'mask': None
                }]
                self._refresh_validated_label()
                self.current_idx = 0
                self.load_validation_image()
                self._switch_page(2)   # Go to Segment & Validate
                self.status_label.setText(f"Status: Image loaded — {os.path.basename(file_path)}")
            else:
                self.status_label.setText("Status: Error loading image — file may be corrupt or unsupported.")
        else:
            self.status_label.setText("Status: Image load cancelled")

    def load_validation_image(self):
        if self.current_idx < 0 or self.current_idx >= len(self.validation_entries):
            return
        entry = self.validation_entries[self.current_idx]
        self.image = entry['image']
        self.image_display = entry['image']
        self.mask = entry['mask']
        self.display_image()

    def nav_prev(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_validation_image()
            if self._stack.currentIndex() != 5:
                self._switch_page(5)

    def nav_next(self):
        if self.current_idx < len(self.validation_entries) - 1:
            self.current_idx += 1
            self.load_validation_image()
            if self._stack.currentIndex() != 5:
                self._switch_page(5)

    def display_image(self):
        """Schedule a debounced redraw (safe to call on every slider tick)."""
        self._display_timer.start()  # resets the timer if already running

    def _do_display_image(self):
        """Actual render — runs at most every 40 ms."""
        disp_img = getattr(self, 'image_display', None)
        if disp_img is None:
            disp_img = self.image
            
        if disp_img is None:
            return

        # Apply contrast/brightness
        alpha_cont = getattr(self, 'contrast_slider', None)
        beta_bright = getattr(self, 'brightness_slider', None)
        if alpha_cont and beta_bright:
            c_val = alpha_cont.value() / 100.0
            b_val = beta_bright.value()
            if c_val != 1.0 or b_val != 0:
                disp_img = cv2.convertScaleAbs(disp_img, alpha=c_val, beta=b_val)

        # Update Original Image Panel
        self.view_orig.load_image(disp_img)

        # Update Overlay Image Panel with masks
        mask_to_load = self.mask
        if mask_to_load is not None and disp_img.shape[:2] != mask_to_load.shape:
            mask_to_load = cv2.resize(mask_to_load, (disp_img.shape[1], disp_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        self.view_overlay.load_image_and_masks(disp_img, mask_to_load)

        if self.validation_entries and 0 <= self.current_idx < len(self.validation_entries):
            entry = self.validation_entries[self.current_idx]
            img_file = entry.get('file', 'Unknown')
            fname = os.path.basename(img_file)
            dirname = os.path.dirname(img_file)
            
            if getattr(self, 'batch_folder_path', None) and dirname == self.batch_folder_path:
                sname = "Sample"
            else:
                sname = os.path.basename(dirname) if dirname else "Sample"
                
            mask = entry.get('mask')
            pollen_count = int(mask.max()) if mask is not None else 0
            count_str = f"  ·  {pollen_count} pollen" if mask is not None else ""
            nav_text_full = f"Sample: {sname}  |  File: {fname}  |  Image {self.current_idx + 1} / {len(self.validation_entries)}{count_str}"
            nav_text_short = f"{sname}/{fname}  [{self.current_idx + 1}/{len(self.validation_entries)}]{count_str}"
            self.lbl_nav.setText(nav_text_full)
            if hasattr(self, "lbl_nav_v"):
                self.lbl_nav_v.setText(nav_text_short)
        else:
            self.lbl_nav.setText("Image 0 / 0")
            if hasattr(self, "lbl_nav_v"):
                self.lbl_nav_v.setText("0 / 0")

    def run_segmentation(self):
        if self.image is None:
            self.status_label.setText("Status: Load an image first")
            return

        self.status_label.setText("Status: Running segmentation (background)...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)   # indeterminate spinner

        params = dict(
            image=self.image,
            resize_val=self.resize_spin.value(),
            channel_func=CHANNEL_OPTIONS[self.channel_combo.currentText()],
            diameter=0 if self.auto_diameter.isChecked() else self.diameter_spin.value(),
            flow_thresh=self.flow_spin.value(),
            cellprob=self.cellprob_spin.value(),
            max_iter=self.iter_spin.value(),
            norm_low=self.norm_low_spin.value(),
            norm_up=self.norm_up_spin.value(),
        )
        if hasattr(self, '_seg_thread') and self._seg_thread.isRunning():
            self._seg_thread.quit()
            self._seg_thread.wait(2000)
        self._seg_thread = SegmentationThread(self.current_model, [params])
        self._seg_thread.finished.connect(self._on_segmentation_done)
        self._seg_thread.error.connect(self._on_segmentation_error)
        self._seg_thread.start()

    def _on_segmentation_done(self, results, _):
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.progress_bar.setVisible(False)
        if not results:
            self.status_label.setText("Status: Segmentation returned no results.")
            return
        masks_filtered, img_resized = results[0]
        n_valid = int(masks_filtered.max())
        self.image = img_resized
        self.image_display = img_resized
        self.mask = masks_filtered
        # Update the current validation entry if one exists
        if self.validation_entries and 0 <= self.current_idx < len(self.validation_entries):
            self.validation_entries[self.current_idx]['mask'] = masks_filtered
            self.validation_entries[self.current_idx]['image'] = img_resized
        self.display_image()
        self._refresh_validated_label()
        self.status_label.setText(f"Status: Segmentation complete. Found {n_valid} viable grains.")

    def _on_segmentation_error(self, msg):
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Status: Segmentation error — {msg}")

    def start_training(self):
        if not self.validation_entries:
            self.status_label.setText("Status: No images available for training.")
            return

        train_data = []
        train_labels = []
        channel_func = CHANNEL_OPTIONS[self.channel_combo.currentText()]

        for entry in self.validation_entries:
            img = entry.get('image')
            mask = entry.get('mask')
            if img is not None and mask is not None and np.max(mask) > 0:
                train_data.append(channel_func(img))
                train_labels.append(mask)

        if len(train_data) == 0:
            self.status_label.setText("Status: No masks drawn/found. Cannot train.")
            return

        if getattr(self, 'output_dir', None):
            base_dir = self.output_dir
        elif getattr(self, 'batch_folder_path', None):
            base_dir = self.batch_folder_path
        elif self.validation_entries and 'file' in self.validation_entries[0]:
            base_dir = os.path.dirname(self.validation_entries[0]['file'])
        else:
            base_dir = os.getcwd()

        save_dir = os.path.join(base_dir, "custom_models")
        os.makedirs(save_dir, exist_ok=True)

        # Default model name: if already on a custom model, append _v2, _v3, etc.
        # so each iteration is saved separately and nothing is overwritten.
        user_name = self.model_name_input.text().strip()
        if not user_name:
            user_name = "pollen_model"
        model_name = user_name

        self.start_train_btn.setEnabled(False)
        self.train_log.clear()
        base_display = getattr(self, '_active_model_display_name', 'cellpose (default)')
        self.train_log.append(
            f"Base model:  {base_display}\n"
            f"Training on: {len(train_data)} image(s)\n"
            f"Save name:   {model_name}\n"
            f"Epochs:      {self.epochs_spin.value()}  |  LR: {self.lr_spin.value()}\n"
            f"{'-'*48}\n"
        )
        self.status_label.setText(f"Status: Training {base_display}...")

        base_model = self.current_model

        # Clean up any previous thread
        if hasattr(self, 'training_thread') and self.training_thread.isRunning():
            self.training_thread.quit()
            self.training_thread.wait(3000)
        self.training_thread = TrainingThread(
            base_model, train_data, train_labels, save_dir, model_name,
            self.epochs_spin.value(), self.lr_spin.value()
        )
        self.training_thread.log_signal.connect(self.update_train_log)
        self.training_thread.finished_signal.connect(self.training_finished)
        self.training_thread.start()
        
    def update_train_log(self, text):
        self.train_log.insertPlainText(text)
        self.train_log.verticalScrollBar().setValue(self.train_log.verticalScrollBar().maximum())

    def training_finished(self, result_path):
        self.start_train_btn.setEnabled(True)
        if result_path.startswith("ERROR:"):
            self.status_label.setText("Status: Training failed.")
            self.train_log.append(f"\n{result_path}")
        else:
            self.status_label.setText("Status: Training complete!")
            self.train_log.append(f"\nModel saved to: {result_path}")

            # Automatically load the newly trained model and make it the active model.
            # This means the next training run will continue fine-tuning THIS model,
            # not fall back to the base cellpose/SAM model.
            global model_custom, model_custom_name
            try:
                model_custom = models.CellposeModel(
                    gpu=torch.cuda.is_available(), pretrained_model=result_path
                )
                model_custom_name = os.path.basename(result_path)
                self.custom_model_label.setText(f"Loaded: {model_custom_name}")
                self._active_model_display_name = f"custom: {model_custom_name}"
                if "custom" not in [self.model_combo.itemText(i)
                                     for i in range(self.model_combo.count())]:
                    self.model_combo.addItem("custom")
                # Block signals so change_model doesn't re-run _update_train_base_label
                # a second time — we do it explicitly below with the right name already set
                self.model_combo.blockSignals(True)
                self.model_combo.setCurrentText("custom")
                self.model_combo.blockSignals(False)
                self.current_model = model_custom
                self._update_train_base_label()
                self.status_label.setText(
                    "Status: New model loaded and set as active. "
                    "You can segment with it or train further."
                )
            except Exception as e:
                self.train_log.append(f"\nFailed to load the trained model: {str(e)}")

    def _rename_sample_item(self, item):
        """Inline rename of a sample label in the dashboard list."""
        old_text = item.text()
        new_text, ok = QInputDialog.getText(self, "Rename Sample", "New display name:", text=old_text)
        if ok and new_text.strip():
            item.setText(new_text.strip())

    def _redraw_plot(self):
        """Redraw the plot using current order/labels/options — no re-segmentation."""
        if not hasattr(self, '_last_df_counts') or self._last_df_counts is None:
            self.status_label.setText("Status: No analysis data yet — run Count first.")
            return

        df = self._last_df_counts.copy()

        # Build order and apply renames from plot_order_list
        new_order = []
        rename_map = {}
        for i in range(self.plot_order_list.count()):
            item = self.plot_order_list.item(i)
            orig = item.data(Qt.ItemDataRole.UserRole)
            label = item.text()
            new_order.append(orig)
            if label != orig:
                rename_map[orig] = label

        if rename_map:
            df['Sample'] = df['Sample'].map(lambda x: rename_map.get(x, x))
            new_order = [rename_map.get(s, s) for s in new_order]

        # Custom title/ylabel
        custom_title = self.plot_title_edit.text().strip() or None
        custom_ylabel = self.plot_ylabel_edit.text().strip() or None
        pt_size = self.plot_pt_spin.value()
        show_jitter = self.plot_jitter_cb.isChecked()
        show_mean = self.plot_mean_cb.isChecked()

        plot_path, _, fig = plot_publication_figure(
            df, self._last_sig_pairs, self._last_stat_method, "pyqt6",
            self._last_output_dir, sample_order=new_order,
            custom_title=custom_title, custom_ylabel=custom_ylabel,
            pt_size=pt_size, show_jitter=show_jitter, show_mean=show_mean
        )

        # Replace canvas in-place — close old figure first to free matplotlib memory
        splitter = self.figure_splitter
        splitter_sizes = splitter.sizes()
        old_fig = self.figure_canvas.figure
        new_canvas = FigureCanvas(fig)
        splitter.insertWidget(1, new_canvas)
        self.figure_canvas.setParent(None)
        self.figure_canvas.deleteLater()
        self.figure_canvas = new_canvas
        splitter.setSizes(splitter_sizes)
        plt.close(old_fig)
        self.status_label.setText(f"Status: Plot redrawn and saved to {plot_path}")

    def _refresh_validated_label(self):
        """Update the Step 2 counter label and enable/disable the count button."""
        n = len(self.validation_entries)
        if n == 0:
            self.validated_count_lbl.setText("No validated images yet.")
            self.count_validated_btn.setEnabled(False)
        else:
            with_masks = sum(1 for e in self.validation_entries if e.get('mask') is not None)
            self.validated_count_lbl.setText(
                f"{n} image(s) segmented  ·  {with_masks} with masks ready to count."
            )
            self.count_validated_btn.setEnabled(with_masks > 0)

    def _count_validated(self):
        """Go straight to counting using the already-validated masks — no re-segmentation."""
        if not self.validation_entries:
            self.status_label.setText("Status: No validated images to count.")
            return
        self.analyze_pollen()

    def analyze_pollen(self):
        if not self.validation_entries:
            self.status_label.setText("Status: No validated images to analyze")
            return

        self._switch_page(4)   # Results & Export page

        if getattr(self, 'output_dir', None):
            base_dir = self.output_dir
        elif getattr(self, 'batch_folder_path', None):
            base_dir = self.batch_folder_path
        elif self.validation_entries and 'file' in self.validation_entries[0]:
            base_dir = os.path.dirname(self.validation_entries[0]['file'])
        else:
            base_dir = os.getcwd()

        output_dir = os.path.join(base_dir, "pollen_analysis_output")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        _plot_order = []
        if hasattr(self, 'sample_order_list'):
            for idx in range(self.sample_order_list.count()):
                _plot_order.append(self.sample_order_list.item(idx).text())

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(self.validation_entries))
        self.progress_bar.setValue(0)
        self.status_label.setText("Status: Analysing in background…")

        if hasattr(self, '_analysis_thread') and self._analysis_thread.isRunning():
            self._analysis_thread.quit()
            self._analysis_thread.wait(2000)
        self._analysis_thread = AnalysisThread(
            self.validation_entries, output_dir, timestamp,
            self.save_int_cb.isChecked(), _plot_order,
            batch_folder_path=getattr(self, 'batch_folder_path', None),
        )
        self._analysis_thread.progress.connect(self._on_analysis_progress)
        self._analysis_thread.finished.connect(
            lambda r, _od=output_dir, _po=_plot_order: self._on_analysis_done(r, _od, _po)
        )
        self._analysis_thread.error.connect(self._on_analysis_error)
        self._analysis_thread.start()

    def _on_analysis_progress(self, current, total, msg):
        self.progress_bar.setValue(current)
        self.status_label.setText(f"Status: {msg}")

    def _on_analysis_error(self, msg):
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Status: Analysis error — {msg}")

    def _on_analysis_done(self, result, output_dir, plot_order):
        self.progress_bar.setVisible(False)

        if result.get("empty"):
            self.results_text.setText("No pollen detected to analyze.")
            self.status_label.setText("Status: No data")
            return

        df_counts   = result["df_counts"]
        csv_path    = result["csv_path"]
        stat_report = result["stat_report"]
        sig_pairs   = result["sig_pairs"]
        stat_method = result["stat_method"]
        plot_path   = result["plot_path"]
        pdf_path    = result["pdf_path"]
        fig         = result["fig"]

        # Populate stats table
        self.res_table.setRowCount(0)
        for sname in df_counts["Sample"].unique():
            sg = df_counts[df_counts["Sample"] == sname]
            v  = sg["Count"].values
            row_pos = self.res_table.rowCount()
            self.res_table.insertRow(row_pos)
            self.res_table.setItem(row_pos, 0, QTableWidgetItem(sname))
            self.res_table.setItem(row_pos, 1, QTableWidgetItem(str(len(sg))))
            self.res_table.setItem(row_pos, 2, QTableWidgetItem(f"{v.mean():.2f}"))
            self.res_table.setItem(row_pos, 3, QTableWidgetItem(f"{float(np.median(v)):.2f}"))
            self.res_table.setItem(row_pos, 4, QTableWidgetItem(f"{v.std():.2f}"))

        # Populate dashboard reorder list
        if hasattr(self, 'plot_order_list'):
            existing_labels = {
                self.plot_order_list.item(i).data(Qt.ItemDataRole.UserRole):
                    self.plot_order_list.item(i).text()
                for i in range(self.plot_order_list.count())
            }
            self.plot_order_list.clear()
            ordered_snames = plot_order if plot_order else list(df_counts["Sample"].unique())
            for sname in ordered_snames:
                label = existing_labels.get(sname, sname)
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, sname)
                self.plot_order_list.addItem(item)

        # Store for redraw
        self._last_df_counts  = df_counts
        self._last_sig_pairs  = sig_pairs
        self._last_stat_method = stat_method
        self._last_output_dir  = output_dir

        # Swap canvas — close old figure first to free matplotlib memory
        splitter       = self.figure_splitter
        splitter_sizes = splitter.sizes()
        old_fig = self.figure_canvas.figure
        new_canvas = FigureCanvas(fig)
        splitter.insertWidget(1, new_canvas)
        self.figure_canvas.setParent(None)
        self.figure_canvas.deleteLater()
        self.figure_canvas = new_canvas
        splitter.setSizes(splitter_sizes)
        plt.close(old_fig)

        # Text summary
        final_text  = "=== ANALYSIS COMPLETE ===\n\n"
        final_text += f"Total Images Processed: {len(self.validation_entries)}\n"
        final_text += f"Total Pollen Counted: {df_counts['Count'].sum()}\n\n"
        final_text += f"Saved Outputs to:\n{output_dir}\n"
        final_text += f"- Data: {os.path.basename(csv_path)}\n"
        final_text += f"- Plot: {os.path.basename(plot_path)}\n"
        final_text += f"- Report: {os.path.basename(pdf_path)}\n\n"
        final_text += "--- STATISTICAL REPORT ---\n"
        final_text += stat_report
        self.results_text.setText(final_text)
        self.status_label.setText(
            f"Status: Analysis complete. Reports saved to {output_dir}")
        self._switch_page(6)   # Jump to Results Dashboard to show the plot

    def save_results(self):
        if self.mask is None:
            self.status_label.setText("Status: No results to save")
            return

        # Ask user for a base save path (no extension — we'll write several files)
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Current Results (base name)", "",
            "PNG Files (*.png);;All Files (*)"
        )
        if not file_path:
            return

        # Ensure the path ends with .png for the mask; strip if user typed it
        base_path = file_path
        if base_path.lower().endswith('.png'):
            base_path = base_path[:-4]

        saved = []
        errors = []

        # ── 1. Save mask as uint16 PNG ──────────────────────────────────────────
        try:
            mask_path = base_path + "_mask.png"
            # cv2.imwrite supports 16-bit PNG natively via uint16 array
            cv2.imwrite(mask_path, self.mask.astype(np.uint16))
            saved.append(os.path.basename(mask_path))
        except Exception as e:
            errors.append(f"mask: {e}")

        # ── 2. Save overlay image (coloured cells on original) ──
        try:
            if self.image_display is not None:
                overlay_img = plot_overlay(self.image_display, self.mask, alpha=0.5)
                overlay_path = base_path + "_overlay.png"
                overlay_img.save(overlay_path)
                saved.append(os.path.basename(overlay_path))
        except Exception as e:
            errors.append(f"overlay: {e}")

        # ── 3. Save original resized image ──
        try:
            if self.image_display is not None:
                orig_path = base_path + "_image.png"
                img_to_save = self.image_display
                if img_to_save.ndim == 3:
                    img_bgr = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = img_to_save
                cv2.imwrite(orig_path, img_bgr)
                saved.append(os.path.basename(orig_path))
        except Exception as e:
            errors.append(f"image: {e}")

        # ── 4. Save per-image CSV with label counts ──
        try:
            n_cells = int(self.mask.max())
            ids = np.arange(1, n_cells + 1)
            areas = np.array([int(np.sum(self.mask == i)) for i in ids])
            diameters = 2.0 * np.sqrt(areas / np.pi)
            df_mask = pd.DataFrame({
                "label": ids,
                "area_px": areas,
                "diameter_px": np.round(diameters, 2),
            })
            # Summary row
            summary = pd.DataFrame([{
                "label": "TOTAL",
                "area_px": int(areas.sum()),
                "diameter_px": round(float(np.mean(diameters)), 2),
            }])
            df_out = pd.concat([df_mask, summary], ignore_index=True)
            csv_path = base_path + "_counts.csv"
            df_out.to_csv(csv_path, index=False)
            saved.append(os.path.basename(csv_path))
        except Exception as e:
            errors.append(f"CSV: {e}")

        # ── Status message ──
        msg = f"Status: Saved — {', '.join(saved)}"
        if errors:
            msg += f"  |  Errors: {'; '.join(errors)}"
        self.status_label.setText(msg)

if __name__ == "__main__":
    # Required for ProcessPoolExecutor with spawn context on Windows/macOS
    multiprocessing.freeze_support()

    # ── USER-TUNABLE FONT SCALE ──────────────────────────────────────────────
    # Increase this number to make ALL text in the app larger.
    # 1.0 = default  |  1.2 = 20% larger  |  1.5 = 50% larger  |  0.9 = 10% smaller
    FONT_SCALE = 1.0
    # ────────────────────────────────────────────────────────────────────────

    # Enable high-DPI scaling
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass

    app = QApplication(sys.argv)

    # Auto-scale based on screen DPI (96 dpi → base, 192 dpi → 2× base)
    _screen = app.primaryScreen()
    _dpi    = _screen.logicalDotsPerInch() if _screen is not None else 96.0
    _dpi_scale   = _dpi / 96.0
    _total_scale = FONT_SCALE * _dpi_scale

    # Set the QApplication base font — affects widgets not covered by stylesheet
    _base_pt = max(9, int(round(10 * _total_scale)))
    _app_font = app.font()
    _app_font.setPointSize(_base_pt)
    app.setFont(_app_font)

    window = PollenAnalysisApp(font_scale=_total_scale)
    window.show()
    exit_code = app.exec()

    # Graceful shutdown — stop all background threads before exit
    for attr in ('training_thread', '_seg_thread', '_seg_batch_thread', '_analysis_thread'):
        thread = getattr(window, attr, None)
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait(3000)

    sys.exit(exit_code)