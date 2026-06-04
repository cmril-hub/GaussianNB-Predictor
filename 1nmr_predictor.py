"""
╔══════════════════════════════════════════════════════════════╗
║  NucleicML — Prediction App                                 ║
║  Loads saved .pkl and predicts Nucleic_acid_intermediate    ║
╚══════════════════════════════════════════════════════════════╝
Run: streamlit run prediction_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import joblib
import io
import os
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NucleicML · Prediction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── STYLES ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }

.pred-banner {
    border-radius: 14px;
    padding: 1.8rem 2rem;
    text-align: center;
    margin: 1rem 0;
    border: 1px solid;
}
.pred-class  { font-family: 'DM Serif Display'; font-size: 2.2rem; margin-bottom: 0.3rem; }
.pred-sub    { font-family: 'DM Mono'; font-size: 0.85rem; opacity: 0.8; }
.prob-bar-wrap { margin: 0.3rem 0; }
.section-header {
    background: linear-gradient(90deg, #1f2937, transparent);
    border-left: 3px solid #7ee787;
    padding: 0.6rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1.5rem 0 1rem 0;
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: #e6edf3;
}
.csv-snippet {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-family: 'DM Mono';
    font-size: 0.78rem;
    color: #79c0ff;
    overflow-x: auto;
    white-space: pre;
}
.stButton > button {
    background: linear-gradient(135deg, #1a7a3a, #2ea043);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans';
    font-weight: 500;
}
.info-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─── CLASS META ───────────────────────────────────────────────────────────────
CLASS_META = {
    "DNA_rare_intermediate": {
        "color": "#f78166", "bg": "#2d1a18",
        "icon": "🧬",
        "desc": "Rare DNA intermediate (G-Quadruplex / D-Loop / Triplex / Promoter G4)",
        "structures": [
            "G-Quadruplex (telomeric)",
            "G-Quadruplex (c-MYC promoter)",
            "G-Quadruplex (bcl-2 promoter)",
            "DNA Triplex (TFO)",
            "D-Loop (RecA/Rad51 strand invasion)",
        ],
    },
    "DNA_common_intermediate": {
        "color": "#58a6ff", "bg": "#141d2e",
        "icon": "🔵",
        "desc": "Common DNA intermediate (Hairpin / Abasic Site Duplex)",
        "structures": [
            "DNA Hairpin (stem-loop) 16-nt",
            "Abasic Site Duplex (14-mer / THF)",
        ],
    },
    "RNA_intermediates": {
        "color": "#3fb950", "bg": "#172018",
        "icon": "🟢",
        "desc": "RNA intermediate (A-form Duplex / siRNA 21-bp)",
        "structures": [
            "RNA A-form Duplex (10-mer)",
            "siRNA (21-bp dsRNA antisense+sense)",
        ],
    },
    "RNA_oligonucleotides": {
        "color": "#d2a8ff", "bg": "#20182e",
        "icon": "🟣",
        "desc": "Short RNA oligonucleotide (Hairpin / Tetramer / Hexamer)",
        "structures": [
            "RNA 23-nt Pentaloop Hairpin (16S rRNA analog)",
            "RNA Tetramer (4-nt ssRNA)",
            "RNA Hexamer d(rCGCGCG)-like hairpin",
        ],
    },
}

# ─── SCHEMATIC DRAWERS ────────────────────────────────────────────────────────
def setup_dark_ax(ax, title=""):
    ax.set_facecolor("#0d1117")
    ax.figure.set_facecolor("#0d1117")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    if title:
        ax.set_title(title, color="#e6edf3", fontsize=15, fontfamily="monospace", pad=8)


def draw_g_quadruplex(ax):
    """G-Quadruplex hybrid topology schematic (4 G-quartets + K+ ions)"""
    setup_dark_ax(ax, "G-Quadruplex · Hybrid Topology")

    gold = "#c9a84c"
    blue_g = "#3a6fa8"
    kplus = "#3fb950"

    # Draw 4 G-quartet planes (stacked rectangles, tilted)
    y_levels = [0.18, 0.38, 0.58, 0.78]
    for i, y in enumerate(y_levels):
        # Outer rhombus quad
        r = 0x1a + i * 8
        g = 0x33 + i * 6
        b = 0xa8 + i * 4
        face_hex = f"#{r:02x}{g:02x}{b:02x}"
        sq = plt.Polygon([[0.22, y], [0.5, y-0.07], [0.78, y], [0.5, y+0.07]],
                         closed=True, facecolor=face_hex,
                         edgecolor=blue_g, linewidth=1.2, alpha=0.85, zorder=2)
        ax.add_patch(sq)
        # Label
        ax.text(0.5, y, f"G-quartet {i+1}", color="#79c0ff", ha="center", va="center",
                fontsize=13, fontfamily="monospace", zorder=3)

    # K+ ions between planes
    ion_y = [(y_levels[i]+y_levels[i+1])/2 for i in range(3)]
    for iy in ion_y:
        circ = plt.Circle((0.5, iy), 0.035, color=kplus, zorder=4)
        ax.add_patch(circ)
        ax.text(0.5, iy, "K⁺", color="#0d1117", ha="center", va="center",
                fontsize=12, fontweight="bold", zorder=5)

    # G-tract arrows (4 columns)
    for xpos, label, dy in [(0.18, "G1\nG5\nG9\nG13", 1), (0.38, "G3\nG7\nG11\nG15", -1),
                              (0.62, "G17\nG19", 1),  (0.78, "G21\nG3", -1)]:
        ax.annotate("", xy=(xpos, 0.85 if dy==1 else 0.12),
                    xytext=(xpos, 0.12 if dy==1 else 0.85),
                    arrowprops=dict(arrowstyle="->", color=gold, lw=1.8), zorder=1)

    # Loops
    ax.annotate("", xy=(0.22, 0.82), xytext=(0.38, 0.82),
                arrowprops=dict(arrowstyle="-", color="#8b949e",
                                connectionstyle="arc3,rad=-0.5", lw=1.2))
    ax.text(0.30, 0.93, "A2 loop", color="#8b949e", ha="center", fontsize=12)

    ax.text(0.5, 0.03, "d[AGGG(TTAGGG)₃] · Tel22  |  Hybrid (3+1) · K⁺ solution",
            color="#8b949e", ha="center", fontsize=12, fontfamily="monospace")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def draw_dna_hairpin(ax):
    """DNA stem-loop hairpin schematic"""
    setup_dark_ax(ax, "DNA Hairpin · Stem-Loop")
    blue = "#58a6ff"
    grey = "#8b949e"

    stem_pairs = 6
    stem_x_l, stem_x_r = 0.30, 0.70
    stem_y_start, stem_y_step = 0.20, 0.09

    # Draw stem base pairs
    for i in range(stem_pairs):
        y = stem_y_start + i * stem_y_step
        ax.plot([stem_x_l, stem_x_r], [y, y], color=blue, lw=2, zorder=2)
        ax.plot(stem_x_l, y, "o", color="#f78166", ms=8, zorder=3)
        ax.plot(stem_x_r, y, "o", color="#f78166", ms=8, zorder=3)
        bases_l = "ATCCTA"
        bases_r = "TAGGAT"[::-1]
        ax.text(stem_x_l-0.05, y, bases_l[i], color="#e6edf3", ha="center",
                va="center", fontsize=13, fontfamily="monospace")
        ax.text(stem_x_r+0.05, y, bases_r[i], color="#e6edf3", ha="center",
                va="center", fontsize=13, fontfamily="monospace")

    # Loop at top
    top_y = stem_y_start + stem_pairs * stem_y_step
    theta = np.linspace(0, np.pi, 40)
    loop_x = 0.5 + 0.20 * np.cos(theta)
    loop_y = top_y + 0.13 * np.sin(theta)
    ax.plot(loop_x, loop_y, color=grey, lw=2, zorder=2)
    ax.text(0.5, top_y+0.16, "Loop\n(4 nt)", color=grey, ha="center", fontsize=13)

    # Single strand tails
    ax.annotate("", xy=(stem_x_l, stem_y_start-0.08), xytext=(stem_x_l, stem_y_start-0.01),
                arrowprops=dict(arrowstyle="-", color=grey, lw=1.5))
    ax.text(stem_x_l-0.05, stem_y_start-0.10, "5'", color=grey, fontsize=14, fontfamily="monospace")
    ax.annotate("", xy=(stem_x_r, stem_y_start-0.08), xytext=(stem_x_r, stem_y_start-0.01),
                arrowprops=dict(arrowstyle="-", color=grey, lw=1.5))
    ax.text(stem_x_r+0.02, stem_y_start-0.10, "3'", color=grey, fontsize=14, fontfamily="monospace")

    ax.text(0.5, 0.03, "d(ATCCTATTTATAGGAT) · 16-nt unimolecular hairpin",
            color="#8b949e", ha="center", fontsize=12, fontfamily="monospace")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def draw_rna_duplex(ax):
    """RNA A-form duplex / siRNA helix schematic"""
    setup_dark_ax(ax, "RNA Duplex · A-form / siRNA")
    green = "#3fb950"
    grey  = "#8b949e"

    cx = 0.5
    n_pairs = 10
    base_y = 0.12
    dy = 0.07
    amp = 0.16   # helix amplitude
    freq = 2.5   # helix frequency

    t = np.linspace(0, np.pi * freq, 200)
    strand1_x = cx + amp * np.cos(t)
    strand2_x = cx - amp * np.cos(t)
    strand_y  = base_y + (t / (np.pi * freq)) * (n_pairs * dy + 0.05)

    ax.plot(strand1_x, strand_y, color=green, lw=2, zorder=2)
    ax.plot(strand2_x, strand_y, color="#d2a8ff", lw=2, zorder=2)

    # Base pair rungs
    for i in range(n_pairs):
        y = base_y + i * dy + dy/2
        pair_t = np.linspace(0, np.pi * freq, n_pairs)[i] if i < n_pairs else 0
        x1 = cx + amp * np.cos((i / n_pairs) * np.pi * freq * 2)
        x2 = cx - amp * np.cos((i / n_pairs) * np.pi * freq * 2)
        ax.plot([x1, x2], [y, y], color=grey, lw=1, alpha=0.6, zorder=1)

    # Labels
    ax.text(0.12, base_y + n_pairs*dy/2, "5'→3'\nstrand 1",
            color=green, ha="center", va="center", fontsize=12, fontfamily="monospace")
    ax.text(0.88, base_y + n_pairs*dy/2, "3'→5'\nstrand 2",
            color="#d2a8ff", ha="center", va="center", fontsize=12, fontfamily="monospace")

    # siRNA overhangs
    ax.annotate("", xy=(cx + amp + 0.02, 0.10), xytext=(cx + amp + 0.08, 0.10),
                arrowprops=dict(arrowstyle="-", color=green, lw=1.5))
    ax.text(cx + amp + 0.10, 0.10, "2 nt\noverhang", color=grey, fontsize=11)

    ax.text(0.5, 0.03, "siRNA (21-bp dsRNA)  |  RNA A-form Duplex (10-mer)",
            color="#8b949e", ha="center", fontsize=12, fontfamily="monospace")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def draw_rna_hairpin(ax):
    """RNA hairpin / short oligonucleotide schematic"""
    setup_dark_ax(ax, "RNA Oligonucleotide · Hairpin / Short ssRNA")
    purple = "#d2a8ff"
    grey   = "#8b949e"
    gold   = "#c9a84c"

    # ── 23-nt pentaloop hairpin ──
    # Stem pairs
    stem_pairs = 7
    sl, sr = 0.28, 0.72
    sy = 0.15
    sdy = 0.08

    for i in range(stem_pairs):
        y = sy + i * sdy
        ax.plot([sl, sr], [y, y], color=purple, lw=1.8, zorder=2, alpha=0.9)
        ax.plot(sl, y, "o", color=gold, ms=6, zorder=3)
        ax.plot(sr, y, "o", color=gold, ms=6, zorder=3)

    top_y = sy + stem_pairs * sdy
    # 5-nt loop (pentaloop)
    theta = np.linspace(0, np.pi, 50)
    lx = 0.5 + 0.22 * np.cos(theta)
    ly = top_y + 0.15 * np.sin(theta)
    ax.plot(lx, ly, color=grey, lw=2)
    ax.text(0.5, top_y + 0.19, "Pentaloop\n(5 nt)", color=grey, ha="center", fontsize=12.5)

    # 5' and 3' tails
    ax.plot([sl, sl], [0.05, sy], color=purple, lw=2)
    ax.text(sl-0.05, 0.04, "5'", color=grey, fontsize=14, fontfamily="monospace")
    ax.plot([sr, sr], [0.05, sy], color=purple, lw=2)
    ax.text(sr+0.02, 0.04, "3'", color=grey, fontsize=14, fontfamily="monospace")

    # Mini tetramer inset (top right)
    ins_ax = ax.inset_axes([0.72, 0.62, 0.26, 0.32])
    ins_ax.set_facecolor("#161b22")
    ins_ax.set_xticks([]); ins_ax.set_yticks([])
    ins_ax.set_title("4-nt ssRNA", color=grey, fontsize=12)
    for sp in ins_ax.spines.values(): sp.set_edgecolor("#30363d")
    for ix, base in enumerate(["r","C","G","C","G"]):
        ins_ax.text(0.15 + ix*0.18, 0.5, base, color=purple,
                    fontsize=15, fontfamily="monospace", fontweight="bold")

    ax.text(0.5, 0.02, "RNA 23-nt Hairpin (16S rRNA)  |  4-nt / 6-nt ssRNA",
            color="#8b949e", ha="center", fontsize=11, fontfamily="monospace")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def draw_dna_abasic(ax):
    """Abasic site / D-Loop / Triplex schematic"""
    setup_dark_ax(ax, "DNA Abasic Duplex · D-Loop · Triplex")
    blue  = "#58a6ff"
    red   = "#f78166"
    grey  = "#8b949e"
    gold  = "#c9a84c"

    # Main duplex (14-mer abasic)
    n = 7
    x_left, x_right = 0.22, 0.56
    y0, dy = 0.18, 0.09
    for i in range(n):
        y = y0 + i*dy
        if i == 3:  # abasic site
            ax.plot([x_left, x_right], [y, y], color=red, lw=2, alpha=0.7, linestyle="--")
            ax.plot(x_left, y, "X", color=red, ms=9, zorder=4, markeredgewidth=2)
            ax.text(x_left - 0.10, y, "⊘ THF", color=red, fontsize=12.5, va="center")
        else:
            ax.plot([x_left, x_right], [y, y], color=blue, lw=1.8)
            ax.plot(x_left, y, "o", color=blue, ms=7, zorder=3)
            ax.plot(x_right, y, "o", color=blue, ms=7, zorder=3)

    # Backbone lines
    ax.plot([x_left]*2, [y0-0.04, y0+n*dy], color=blue, lw=2.5, zorder=1)
    ax.plot([x_right]*2, [y0-0.04, y0+n*dy], color=blue, lw=2.5, zorder=1)

    # 5'/3' labels
    ax.text(x_left-0.04, y0+n*dy+0.02, "5'", color=grey, fontsize=14)
    ax.text(x_left-0.04, y0-0.07,      "3'", color=grey, fontsize=14)
    ax.text(x_right+0.02, y0+n*dy+0.02,"3'", color=grey, fontsize=14)
    ax.text(x_right+0.02, y0-0.07,     "5'", color=grey, fontsize=14)

    # D-Loop inset (top right)
    ins = ax.inset_axes([0.65, 0.40, 0.33, 0.55])
    ins.set_facecolor("#0d1117")
    ins.set_xticks([]); ins.set_yticks([])
    ins.set_title("D-Loop", color=gold, fontsize=13)
    for sp in ins.spines.values(): sp.set_edgecolor("#30363d")
    # Simplified D-loop: displaced strand bubble
    theta = np.linspace(0, np.pi, 50)
    ins.plot(0.5 + 0.35*np.cos(theta), 0.5 + 0.35*np.sin(theta), color=gold, lw=2)
    ins.plot([0.15, 0.85], [0.5, 0.5], color=blue, lw=2)
    ins.text(0.5, 0.88, "Displaced", color=grey, ha="center", fontsize=11)
    ins.text(0.5, 0.32, "Invasion strand", color=gold, ha="center", fontsize=11)
    ins.set_xlim(0, 1); ins.set_ylim(0, 1)

    ax.text(0.5, 0.02, "Abasic Duplex (14-mer/THF)  |  D-Loop  |  Triplex (TFO)",
            color="#8b949e", ha="center", fontsize=11, fontfamily="monospace")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


SCHEMATIC_FUNCS = {
    "DNA_rare_intermediate":    draw_g_quadruplex,
    "DNA_common_intermediate":  draw_dna_hairpin,
    "RNA_intermediates":        draw_rna_duplex,
    "RNA_oligonucleotides":     draw_rna_hairpin,
}


def make_schematic(predicted_class):
    fig, ax = plt.subplots(figsize=(7, 6.5), dpi=350)
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor":   "#0d1117",
        "text.color":       "#e6edf3",
        "font.family":      "monospace",
        "font.size":        13,
        "figure.dpi":       350,
        "savefig.dpi":      350,
    })
    draw_fn = SCHEMATIC_FUNCS.get(predicted_class)
    if draw_fn:
        draw_fn(ax)
    plt.tight_layout()
    return fig


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 NucleicML")
    st.markdown("**Prediction App**")
    st.markdown("---")

    st.markdown("#### Load Trained Model")
    model_file = st.file_uploader("Upload .pkl model bundle", type=["pkl"])
    st.caption("Train & download your .pkl from the Training App, then upload it here.")
    # Local path fallback (only works when running locally, not on Streamlit Cloud)
    with st.expander("🖥️ Local path (advanced)", expanded=False):
        model_path_txt = st.text_input("File path", value="./models/best_nucleic_model.pkl")
        st.caption("This only works when running locally. On Streamlit Cloud, use the uploader above.")

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
bundle = None
if model_file is not None:
    try:
        bundle = joblib.load(model_file)
        st.sidebar.success(f"✅ Model loaded: **{bundle['model_name']}**")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
elif os.path.exists(model_path_txt):
    try:
        bundle = joblib.load(model_path_txt)
        st.sidebar.success(f"✅ Model loaded: **{bundle['model_name']}**")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("# 🔬 NucleicML — Prediction Engine")
st.markdown("*Predict nucleic acid intermediate class from NMR relaxation parameters*")

if bundle is None:
    st.warning("⬆️ Upload a trained `.pkl` model file via the sidebar (or run `training_app.py` first to generate one).")
    st.stop()

model      = bundle["model"]
scaler     = bundle.get("scaler")
le         = bundle["label_encoder"]
feat_cols  = bundle["feature_cols"]
cls_map    = bundle.get("class_to_structures", {})
needs_sc   = bundle.get("needs_scaler", False)

def predict_single(field, t1, t2):
    inp = np.array([[field, t1, t2]])
    if needs_sc and scaler:
        inp = scaler.transform(inp)
    raw      = model.predict(inp)[0]
    # Model was trained on encoded integer labels; decode back to string class name
    pred_cls = le.inverse_transform([raw])[0] if hasattr(le, "classes_") else str(raw)
    proba    = model.predict_proba(inp)[0] if hasattr(model, "predict_proba") else None
    return pred_cls, proba

# ═══════════════════════════════════════════════════════════════════════
#  TAB LAYOUT
# ═══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📋 Batch CSV Prediction", "ℹ️ Model Info"])

# ── TAB 1: SINGLE ─────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Enter NMR Parameters</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        field_mhz = st.number_input(
            "Field_MHz",
            min_value=100.0, max_value=1200.0, value=600.0, step=100.0,
            help="Magnetic field strength in MHz (e.g. 500, 600, 800)"
        )
        st.caption("🔵 Proton NMR field frequency")
    with col2:
        t1_s = st.number_input(
            "T1_s (Longitudinal relaxation)",
            min_value=0.001, max_value=10.0, value=1.03, step=0.01, format="%.4f",
            help="T1 relaxation time in seconds"
        )
        st.caption("📈 Spin-lattice relaxation time")
    with col3:
        t2_s = st.number_input(
            "T2_s (Transverse relaxation)",
            min_value=0.001, max_value=5.0, value=0.045, step=0.001, format="%.4f",
            help="T2 relaxation time in seconds"
        )
        st.caption("📉 Spin-spin relaxation time")

    predict_btn = st.button("🔮 Predict Class", width='stretch')

    if predict_btn:
        pred_cls, proba = predict_single(field_mhz, t1_s, t2_s)
        meta = CLASS_META.get(pred_cls, {})
        color = meta.get("color", "#58a6ff")
        bg    = meta.get("bg", "#161b22")
        icon  = meta.get("icon", "🧬")

        # Result banner
        st.markdown(f"""
        <div class="pred-banner" style="background:{bg}; border-color:{color};">
            <div style="font-size:2.5rem">{icon}</div>
            <div class="pred-class" style="color:{color}">{pred_cls.replace('_', ' ')}</div>
            <div class="pred-sub">{meta.get('desc','')}</div>
        </div>
        """, unsafe_allow_html=True)

        # Probabilities
        if proba is not None:
            st.markdown("#### Class Probabilities")
            prob_df = pd.DataFrame({
                "Class": le.classes_,
                "Probability": proba
            }).sort_values("Probability", ascending=False)

            fig_prob, ax_prob = plt.subplots(figsize=(9, 4.5), dpi=350)
            ax_prob.set_facecolor("#161b22")
            fig_prob.set_facecolor("#0d1117")
            colors_prob = [CLASS_META.get(c, {}).get("color", "#30363d") for c in prob_df["Class"]]
            bars = ax_prob.barh(
                prob_df["Class"].str.replace("_", " "),
                prob_df["Probability"],
                color=colors_prob, edgecolor="#21262d"
            )
            for bar, val in zip(bars, prob_df["Probability"]):
                ax_prob.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                             f"{val:.1%}", va="center", fontsize=13,
                             color="#e6edf3", fontfamily="monospace")
            ax_prob.set_xlim(0, 1.15)
            ax_prob.set_xlabel("Probability", color="#8b949e")
            ax_prob.tick_params(colors="#8b949e")
            ax_prob.axvline(0.5, color="#30363d", lw=1, ls="--")
            for sp in ax_prob.spines.values(): sp.set_edgecolor("#30363d")
            plt.tight_layout()
            st.pyplot(fig_prob, width='stretch')
            plt.close()

        # Layout: schematic + structure list
        col_s, col_i = st.columns([1, 1])

        with col_s:
            st.markdown("#### Structural Schematic")
            fig_sch = make_schematic(pred_cls)
            st.pyplot(fig_sch, width='stretch')
            plt.close()

        with col_i:
            st.markdown("#### Known Structures of this Class")
            structs = cls_map.get(pred_cls, meta.get("structures", []))
            for idx, s in enumerate(structs, 1):
                st.markdown(f"""
                <div style="
                    background: {bg};
                    border: 2px solid {color};
                    border-left: 6px solid {color};
                    border-radius: 10px;
                    padding: 0.9rem 1.2rem;
                    margin: 0.55rem 0;
                    display: flex;
                    align-items: flex-start;
                    gap: 0.75rem;
                ">
                    <span style="
                        color: {color};
                        font-size: 1.1rem;
                        font-weight: 700;
                        min-width: 1.6rem;
                        line-height: 1.5;
                    ">{idx}.</span>
                    <span style="
                        color: #e6edf3;
                        font-family: 'DM Mono', monospace;
                        font-size: 1.0rem;
                        font-weight: 500;
                        line-height: 1.6;
                        letter-spacing: 0.01em;
                        word-break: break-word;
                    ">{s}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### Input Summary")
            st.markdown(f"""
            <div style="
                background: #161b22;
                border: 2px solid #30363d;
                border-radius: 10px;
                padding: 1rem 1.3rem;
                margin: 0.5rem 0;
            ">
                <table style="width:100%; border-spacing: 0.4rem 0.6rem;">
                  <tr>
                    <td style="color:#8b949e; font-size:1.0rem; font-weight:600; white-space:nowrap;">Field_MHz</td>
                    <td style="color:#e6edf3; font-family:'DM Mono',monospace; font-size:1.1rem; font-weight:700;">{field_mhz}</td>
                  </tr>
                  <tr>
                    <td style="color:#8b949e; font-size:1.0rem; font-weight:600; white-space:nowrap;">T1_s</td>
                    <td style="color:#e6edf3; font-family:'DM Mono',monospace; font-size:1.1rem; font-weight:700;">{t1_s:.4f}</td>
                  </tr>
                  <tr>
                    <td style="color:#8b949e; font-size:1.0rem; font-weight:600; white-space:nowrap;">T2_s</td>
                    <td style="color:#e6edf3; font-family:'DM Mono',monospace; font-size:1.1rem; font-weight:700;">{t2_s:.4f}</td>
                  </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

            if proba is not None:
                conf = proba.max()
                conf_color = "#3fb950" if conf > 0.80 else ("#f0b429" if conf > 0.60 else "#f78166")
                conf_label = "High confidence" if conf > 0.8 else ("Moderate confidence" if conf > 0.6 else "Low confidence — review inputs")
                st.markdown(f"""
                <div style="
                    background: #161b22;
                    border: 2px solid {conf_color};
                    border-radius: 10px;
                    padding: 1rem 1.3rem;
                    margin: 0.5rem 0;
                ">
                    <div style="color:{conf_color}; font-family:'DM Mono',monospace; font-size:1.5rem; font-weight:700;">
                        {conf:.1%} confidence
                    </div>
                    <div style="color:#c9d1d9; font-size:1.0rem; font-weight:500; margin-top:0.35rem;">
                        {conf_label}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ── TAB 2: BATCH ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Batch CSV Prediction</div>', unsafe_allow_html=True)

    # Expected CSV format
    st.markdown("#### Expected CSV Format")
    st.markdown("Upload a CSV with exactly these columns (column names must match):")

    example_csv = (
        "Field_MHz,T1_s,T2_s\n"
        "600,0.632,0.014\n"
        "800,1.450,0.065\n"
        "500,0.980,0.033\n"
        "600,2.100,0.210\n"
        "800,0.750,0.028\n"
    )
    st.markdown(f'<div class="csv-snippet">{example_csv}</div>', unsafe_allow_html=True)

    col_dl, _ = st.columns([1, 3])
    with col_dl:
        st.download_button(
            "⬇️ Download Template CSV",
            data=example_csv,
            file_name="nucleic_prediction_template.csv",
            mime="text/csv"
        )

    st.markdown("---")
    batch_file = st.file_uploader("Upload batch CSV", type=["csv"], key="batch_upload")

    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file, encoding="latin1")
        except Exception:
            batch_df = pd.read_csv(batch_file, encoding="utf-8", errors="replace")

        # Validate
        missing_cols = [c for c in feat_cols if c not in batch_df.columns]
        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols}. Please check your CSV.")
        else:
            X_batch = batch_df[feat_cols].copy()
            X_arr   = scaler.transform(X_batch.values) if (needs_sc and scaler) else X_batch.values

            preds_raw = model.predict(X_arr)
            # Decode integer predictions → string class names
            preds  = le.inverse_transform(preds_raw) if hasattr(le, "classes_") else preds_raw
            probas = model.predict_proba(X_arr) if hasattr(model, "predict_proba") else None

            result_df = batch_df.copy()
            result_df["Predicted_Class"] = preds
            if probas is not None:
                for i, cls in enumerate(le.classes_):
                    result_df[f"Prob_{cls}"] = probas[:, i]
                result_df["Confidence"] = probas.max(axis=1)

            st.markdown("#### Prediction Results")
            # Summary
            counts = pd.Series(preds).value_counts()
            c1, c2, c3, c4 = st.columns(4)
            for col, (cls, cnt) in zip([c1,c2,c3,c4], counts.items()):
                meta = CLASS_META.get(cls, {})
                with col:
                    st.markdown(f"""
                    <div style="background:{meta.get('bg','#161b22')};
                                border:1px solid {meta.get('color','#30363d')};
                                border-radius:10px; padding:0.8rem; text-align:center;">
                        <div style="font-size:1.5rem">{meta.get('icon','🧬')}</div>
                        <div style="color:{meta.get('color','#58a6ff')}; font-family:'DM Mono'; font-size:1.4rem">{cnt}</div>
                        <div style="color:#8b949e; font-size:0.68rem">{cls.replace('_',' ')}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.dataframe(result_df, width='stretch', height=350)

            # Class distribution plot
            fig_b, ax_b = plt.subplots(figsize=(9, 4.5), dpi=350)
            ax_b.set_facecolor("#161b22")
            fig_b.set_facecolor("#0d1117")
            colors_b = [CLASS_META.get(c, {}).get("color", "#58a6ff") for c in counts.index]
            ax_b.bar(
                [c.replace("_", "\n") for c in counts.index],
                counts.values, color=colors_b, edgecolor="#21262d"
            )
            ax_b.set_ylabel("Count", color="#8b949e")
            ax_b.set_title("Batch Prediction Distribution", color="#e6edf3")
            ax_b.tick_params(colors="#8b949e")
            for sp in ax_b.spines.values(): sp.set_edgecolor("#30363d")
            for x, v in enumerate(counts.values):
                ax_b.text(x, v + 0.3, str(v), ha="center", color="#e6edf3", fontsize=13)
            plt.tight_layout()
            st.pyplot(fig_b, width='stretch')
            plt.close()

            # Schematic grid for predicted classes
            st.markdown("#### Schematics for Predicted Classes")
            unique_pred = list(pd.Series(preds).unique())
            ncols = min(len(unique_pred), 2)
            nrows = (len(unique_pred) + ncols - 1) // ncols
            fig_grid, axes_grid = plt.subplots(nrows, ncols,
                                               figsize=(7*ncols, 6.5*nrows),
                                               facecolor="#0d1117", dpi=350)
            if len(unique_pred) == 1:
                axes_grid = [[axes_grid]]
            elif nrows == 1:
                axes_grid = [axes_grid]
            for r in range(nrows):
                for c in range(ncols):
                    idx = r * ncols + c
                    ax_g = axes_grid[r][c] if isinstance(axes_grid[r], (list, np.ndarray)) else axes_grid[r]
                    if idx < len(unique_pred):
                        cls = unique_pred[idx]
                        draw_fn = SCHEMATIC_FUNCS.get(cls, lambda a: setup_dark_ax(a))
                        draw_fn(ax_g)
                        color_g = CLASS_META.get(cls, {}).get("color", "#58a6ff")
                        ax_g.set_xlabel(cls.replace("_", " "), color=color_g,
                                       fontfamily="monospace", fontsize=14)
                    else:
                        ax_g.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_grid, width='stretch')
            plt.close()

            # Download results
            csv_out = result_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Results CSV",
                data=csv_out,
                file_name="nucleic_predictions.csv",
                mime="text/csv",
                width='stretch'
            )

# ── TAB 3: MODEL INFO ─────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Model Information</div>', unsafe_allow_html=True)

    col_mi1, col_mi2 = st.columns(2)
    with col_mi1:
        st.markdown("#### Bundle Contents")
        st.markdown(f"""
        <div style="background:#161b22; border:2px solid #30363d; border-radius:10px; padding:1rem 1.3rem; margin:0.5rem 0;">
          <table style="width:100%; border-spacing:0.4rem 0.65rem;">
            <tr><td style="color:#8b949e; font-size:1.0rem; font-weight:600;">Algorithm</td>
                <td style="color:#e6edf3; font-family:'DM Mono',monospace; font-size:1.05rem; font-weight:700;">{bundle.get('model_name','—')}</td></tr>
            <tr><td style="color:#8b949e; font-size:1.0rem; font-weight:600;">Features</td>
                <td style="color:#e6edf3; font-family:'DM Mono',monospace; font-size:1.05rem; font-weight:700;">{', '.join(feat_cols)}</td></tr>
            <tr><td style="color:#8b949e; font-size:1.0rem; font-weight:600;">Classes</td>
                <td style="color:#e6edf3; font-family:'DM Mono',monospace; font-size:1.05rem; font-weight:700;">{len(le.classes_)}</td></tr>
            <tr><td style="color:#8b949e; font-size:1.0rem; font-weight:600;">Scaler</td>
                <td style="color:#e6edf3; font-family:'DM Mono',monospace; font-size:1.05rem; font-weight:700;">{'StandardScaler' if needs_sc else 'None'}</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    with col_mi2:
        st.markdown("#### Saved Metrics")
        saved_metrics = bundle.get("metrics", {})
        if saved_metrics:
            for k, v in saved_metrics.items():
                st.markdown(f"`{k}`: **{v:.4f}**" if isinstance(v, float) else f"`{k}`: **{v}**")

    st.markdown("#### Class → Structure Lookup")
    for cls, structs in cls_map.items():
        meta = CLASS_META.get(cls, {})
        color_cls = meta.get("color", "#58a6ff")
        with st.expander(f"{meta.get('icon','🧬')} {cls.replace('_',' ')} — {len(structs)} structures"):
            for idx, s in enumerate(structs, 1):
                st.markdown(f"""
                <div style="
                    display:flex; align-items:flex-start; gap:0.7rem;
                    padding:0.55rem 0.2rem; border-bottom:1px solid #21262d;
                ">
                    <span style="color:{color_cls}; font-size:1.0rem; font-weight:700; min-width:1.5rem;">{idx}.</span>
                    <span style="color:#e6edf3; font-family:'DM Mono',monospace; font-size:1.0rem;
                                 font-weight:500; line-height:1.55; word-break:break-word;">{s}</span>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("#### How Columns Were Used")
    st.markdown("""
    | Column | Name | Role in ML | Rationale |
    |--------|------|-----------|-----------|
    | 1 | `Field_MHz` | ✅ Feature | NMR field strength directly affects relaxation rates |
    | 2 | `T1_s` | ✅ Feature | Longitudinal relaxation — key discriminator |
    | 3 | `T2_s` | ✅ Feature | Transverse relaxation — key discriminator |
    | 4 | `Structure` | 🗂 Metadata | Full name stored as lookup; not a feature (too specific) |
    | 5 | `Class` | 🗂 Dropped | Sample-level ID (e.g. Tel122–Tel621); not generalisable as a feature |
    | 6 | `Nucleic_acid_intermediate` | 🎯 Target | Output class to predict |
    """)

st.markdown("---")
st.caption("NucleicML · Nucleic Acid Intermediate Classifier · Built with Streamlit & scikit-learn")
