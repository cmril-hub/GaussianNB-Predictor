"""
NMR Relaxation Classifier — Prediction App
Accepts a saved .pkl (or .pt) model and predicts enzyme binding state
from T1, T2, and Correlation Time inputs — single sample or batch CSV.
"""

import io, warnings, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Optional PyTorch ──────────────────────────────────────────────────────────
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_COLS  = ["T1_Relaxation_Time(s)", "T2_Relaxation_Time(s)", "Correlation_Time(ns)"]
FEAT_LABELS   = ["T₁ Relaxation Time (s)", "T₂ Relaxation Time (s)", "Correlation Time (ns)"]
FEAT_SHORT    = ["T₁ (s)", "T₂ (s)", "τc (ns)"]

# Feature physical ranges for sliders (broad, encompassing typical values)
FEAT_RANGES = {
    "T1_Relaxation_Time(s)":  (0.05, 3.0,  0.001),
    "T2_Relaxation_Time(s)":  (0.005, 0.5, 0.001),
    "Correlation_Time(ns)":   (1.0,  30.0, 0.01),
}
FEAT_DEFAULTS = {
    "T1_Relaxation_Time(s)":  0.95,
    "T2_Relaxation_Time(s)":  0.048,
    "Correlation_Time(ns)":   12.5,
}

# Class colours (consistent with the training app)
CLASS_COLOURS = {
    "Free_Enzyme":           "#16A34A",
    "Bound_to_Wild_type_DNA":"#2563EB",
    "Bound_to_Damage_DNA":   "#DC2626",
}
DEFAULT_COLOUR = "#7C3AED"

# ── Plotly global settings ────────────────────────────────────────────────────
PAPER  = "#0F1923"
PLOT   = "#131F2E"
GRID   = "rgba(255,255,255,0.07)"
FS_T   = 20
FS_A   = 15
FS_TK  = 13
FS_LEG = 14


def _colour(cls):
    return CLASS_COLOURS.get(cls, DEFAULT_COLOUR)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_pkl(file_bytes):
    payload = pickle.loads(file_bytes)
    return payload["pipe"], payload["label_encoder"], payload["classes"]


def load_pt(file_bytes):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not installed — cannot load .pt file.")
    buf = io.BytesIO(file_bytes)
    payload = torch.load(buf, map_location="cpu", weights_only=False)
    pipe = pickle.loads(payload["pipe_bytes"])["pipe"]
    le   = payload["label_encoder"]
    classes = payload["classes"]
    return pipe, le, classes


@st.cache_resource(show_spinner="Loading model…")
def load_model_cached(file_bytes: bytes, file_name: str):
    """Cached so the model is only deserialised once per uploaded file."""
    if file_name.endswith(".pkl"):
        return load_pkl(file_bytes)
    elif file_name.endswith(".pt"):
        return load_pt(file_bytes)
    else:
        raise ValueError("Unsupported file type. Upload a .pkl or .pt model file.")


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def predict_single(pipe, le, values: list):
    X = np.array([values])
    pred_num  = pipe.predict(X)[0]
    pred_lbl  = le.inverse_transform([pred_num])[0]
    try:
        probs = pipe.predict_proba(X)[0]
    except Exception:
        probs = None
    return pred_lbl, probs


def predict_batch(pipe, le, df: pd.DataFrame):
    X         = df[FEATURE_COLS].values
    pred_nums = pipe.predict(X)
    pred_lbls = le.inverse_transform(pred_nums)
    try:
        probs = pipe.predict_proba(X)
    except Exception:
        probs = None
    return pred_lbls, probs


# ══════════════════════════════════════════════════════════════════════════════
#  CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def prob_gauge_chart(probs, classes):
    """Horizontal probability bar chart."""
    clrs  = [_colour(c) for c in classes]
    pcts  = [p * 100 for p in probs]
    order = np.argsort(probs)[::-1]

    fig = go.Figure()
    for i in order:
        fig.add_trace(go.Bar(
            x=[pcts[i]], y=[classes[i]],
            orientation="h",
            marker=dict(color=clrs[i], line=dict(width=0)),
            text=[f"{pcts[i]:.1f}%"],
            textposition="auto",
            textfont=dict(size=FS_A, color="white", family="DM Mono, monospace"),
            name=classes[i],
            showlegend=False,
        ))

    fig.update_layout(
        paper_bgcolor=PAPER, plot_bgcolor=PLOT,
        height=180 + len(classes) * 36,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(
            range=[0, 105], title_text="Confidence (%)",
            title_font=dict(size=FS_A, color="#94A3B8"),
            tickfont=dict(size=FS_TK, color="#94A3B8"),
            gridcolor=GRID, ticksuffix="%",
        ),
        yaxis=dict(
            tickfont=dict(size=FS_A, color="#E2E8F0"),
            autorange="reversed",
        ),
        barmode="relative",
    )
    return fig


def batch_class_bar(pred_lbls, classes):
    """Class frequency bar for batch predictions."""
    vc = pd.Series(pred_lbls).value_counts().reindex(classes, fill_value=0).reset_index()
    vc.columns = ["Class", "Count"]
    vc["Pct"] = (vc["Count"] / len(pred_lbls) * 100).round(1)

    fig = go.Figure()
    for _, row in vc.iterrows():
        fig.add_trace(go.Bar(
            x=[row["Class"]], y=[row["Count"]],
            marker_color=_colour(row["Class"]),
            marker_line_width=0,
            text=[f"{row['Count']}<br>({row['Pct']:.1f}%)"],
            textposition="outside",
            textfont=dict(size=FS_TK + 1, color="#E2E8F0", family="DM Mono, monospace"),
            showlegend=False,
            name=row["Class"],
        ))

    fig.update_layout(
        paper_bgcolor=PAPER, plot_bgcolor=PLOT,
        height=420,
        margin=dict(l=40, r=40, t=50, b=80),
        xaxis=dict(
            title_text="Enzyme State",
            title_font=dict(size=FS_A, color="#94A3B8"),
            tickfont=dict(size=FS_TK + 1, color="#E2E8F0"),
            linecolor="#334155",
        ),
        yaxis=dict(
            title_text="Number of Samples",
            title_font=dict(size=FS_A, color="#94A3B8"),
            tickfont=dict(size=FS_TK, color="#94A3B8"),
            gridcolor=GRID,
        ),
        title=dict(
            text="Predicted Class Distribution",
            font=dict(size=FS_T, color="#F1F5F9"),
            x=0.5,
        ),
    )
    return fig


def prob_heatmap(probs, pred_lbls, classes, max_rows=50):
    """Probability heatmap for batch results."""
    n    = min(len(probs), max_rows)
    data = probs[:n]
    fig  = go.Figure(go.Heatmap(
        z=data,
        x=classes,
        y=[f"#{i+1}" for i in range(n)],
        colorscale=[[0, "#0F1923"], [0.5, "#2563EB"], [1, "#22D3EE"]],
        text=[[f"{v:.2f}" for v in row] for row in data],
        texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        hovertemplate="Sample %{y}<br>%{x}: %{z:.3f}<extra></extra>",
        showscale=True,
        colorbar=dict(
            title_text="Probability",
            title_font=dict(size=FS_TK, color="#94A3B8"),
            tickfont=dict(size=FS_TK - 1, color="#94A3B8"),
            bgcolor=PAPER,
        ),
        zmin=0, zmax=1,
    ))
    fig.update_layout(
        paper_bgcolor=PAPER, plot_bgcolor=PLOT,
        height=max(420, n * 18 + 120),
        margin=dict(l=60, r=60, t=50, b=60),
        xaxis=dict(
            title_text="Enzyme State",
            title_font=dict(size=FS_A, color="#94A3B8"),
            tickfont=dict(size=FS_TK + 1, color="#E2E8F0"),
        ),
        yaxis=dict(
            tickfont=dict(size=max(8, FS_TK - 2), color="#94A3B8"),
            autorange="reversed",
        ),
        title=dict(
            text=f"Prediction Probabilities (first {n} rows)" if n < len(probs) else "Prediction Probabilities — All Rows",
            font=dict(size=FS_T - 2, color="#F1F5F9"),
            x=0.5,
        ),
    )
    return fig


def feature_scatter_batch(df_results, classes):
    """Scatter of T1 vs Correlation_Time coloured by predicted class."""
    fig = px.scatter(
        df_results,
        x="T1_Relaxation_Time(s)", y="Correlation_Time(ns)",
        color="Predicted_Class",
        color_discrete_map=CLASS_COLOURS,
        size_max=10,
        opacity=0.8,
        labels={
            "T1_Relaxation_Time(s)": "T₁ Relaxation Time (s)",
            "Correlation_Time(ns)":  "Correlation Time (ns)",
            "Predicted_Class":        "Predicted Class",
        },
        hover_data=["T2_Relaxation_Time(s)", "Top_Confidence(%)"],
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="rgba(255,255,255,0.3)")))
    fig.update_layout(
        paper_bgcolor=PAPER, plot_bgcolor=PLOT,
        height=460,
        margin=dict(l=60, r=40, t=60, b=60),
        xaxis=dict(
            title_font=dict(size=FS_A, color="#94A3B8"),
            tickfont=dict(size=FS_TK, color="#94A3B8"),
            gridcolor=GRID, linecolor="#334155",
        ),
        yaxis=dict(
            title_font=dict(size=FS_A, color="#94A3B8"),
            tickfont=dict(size=FS_TK, color="#94A3B8"),
            gridcolor=GRID, linecolor="#334155",
        ),
        legend=dict(
            title_text="Predicted Class",
            title_font=dict(size=FS_TK, color="#94A3B8"),
            font=dict(size=FS_LEG, color="#E2E8F0"),
            bgcolor="rgba(15,25,35,0.8)",
            bordercolor="#334155", borderwidth=1,
        ),
        title=dict(
            text="Sample Distribution — T₁ vs Correlation Time",
            font=dict(size=FS_T - 2, color="#F1F5F9"),
            x=0.5,
        ),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  PAGES
# ══════════════════════════════════════════════════════════════════════════════

def page_single(pipe, le, classes):
    st.markdown("### 🔬 Single-Sample Prediction")
    st.markdown(
        "<p style='color:#94A3B8;font-size:0.95rem;margin-top:-8px;'>"
        "Adjust the sliders or type exact values, then click <b>Predict</b>.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col_inp, col_res = st.columns([1, 1], gap="large")

    with col_inp:
        st.markdown("#### Input Parameters")
        vals = {}
        for feat, label, short in zip(FEATURE_COLS, FEAT_LABELS, FEAT_SHORT):
            lo, hi, step = FEAT_RANGES[feat]
            default      = FEAT_DEFAULTS[feat]
            v = st.number_input(
                label,
                min_value=float(lo),
                max_value=float(hi),
                value=float(default),
                step=float(step),
                format="%.4f",
                key=f"single_{feat}",
            )
            vals[feat] = v

        # Input summary card
        st.markdown(
            f"""
            <div style="background:#131F2E;border:1px solid #1E3A5F;border-radius:12px;
                        padding:16px 20px;margin-top:12px;">
              <div style="color:#64748B;font-size:0.8rem;font-family:'DM Mono',monospace;
                          letter-spacing:0.08em;margin-bottom:10px;">INPUT VECTOR</div>
              <div style="display:flex;gap:12px;flex-wrap:wrap;">
                <div style="background:#0F1923;border-radius:8px;padding:8px 14px;
                            border:1px solid #1E3A5F;">
                  <div style="color:#64748B;font-size:0.72rem;">T₁</div>
                  <div style="color:#22D3EE;font-family:'DM Mono',monospace;font-size:1.05rem;font-weight:600;">
                    {vals["T1_Relaxation_Time(s)"]:.4f} s</div>
                </div>
                <div style="background:#0F1923;border-radius:8px;padding:8px 14px;
                            border:1px solid #1E3A5F;">
                  <div style="color:#64748B;font-size:0.72rem;">T₂</div>
                  <div style="color:#22D3EE;font-family:'DM Mono',monospace;font-size:1.05rem;font-weight:600;">
                    {vals["T2_Relaxation_Time(s)"]:.4f} s</div>
                </div>
                <div style="background:#0F1923;border-radius:8px;padding:8px 14px;
                            border:1px solid #1E3A5F;">
                  <div style="color:#64748B;font-size:0.72rem;">τc</div>
                  <div style="color:#22D3EE;font-family:'DM Mono',monospace;font-size:1.05rem;font-weight:600;">
                    {vals["Correlation_Time(ns)"]:.3f} ns</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")
        predict_btn = st.button("⚡ Predict", type="primary", width="stretch")

    with col_res:
        st.markdown("#### Prediction Result")
        if predict_btn or "single_result" in st.session_state:

            if predict_btn:
                feature_values = [vals[f] for f in FEATURE_COLS]
                pred_lbl, probs = predict_single(pipe, le, feature_values)
                st.session_state["single_result"] = (pred_lbl, probs, list(classes))

            pred_lbl, probs, cls_list = st.session_state["single_result"]
            clr = _colour(pred_lbl)

            # Predicted class badge
            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg,{clr}22,{clr}11);
                            border:2px solid {clr};border-radius:16px;
                            padding:24px 28px;text-align:center;margin-bottom:16px;">
                  <div style="color:{clr};font-size:0.78rem;letter-spacing:0.15em;
                              font-family:'DM Mono',monospace;margin-bottom:6px;">
                    PREDICTED CLASS
                  </div>
                  <div style="color:#F1F5F9;font-size:1.45rem;font-weight:800;
                              letter-spacing:0.02em;line-height:1.3;">
                    {pred_lbl.replace("_", " ")}
                  </div>
                  {"<div style='margin-top:10px;color:" + clr + ";font-family:DM Mono,monospace;font-size:1.15rem;font-weight:700;'>" + f"{max(probs)*100:.1f}% confident</div>" if probs is not None else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if probs is not None:
                st.markdown("**Class probabilities**")
                st.plotly_chart(prob_gauge_chart(probs, cls_list),
                                width="stretch")
        else:
            st.markdown(
                """
                <div style="border:1px dashed #1E3A5F;border-radius:16px;
                            padding:60px 28px;text-align:center;color:#475569;">
                  <div style="font-size:2.5rem;margin-bottom:12px;">🧬</div>
                  <div style="font-size:0.95rem;">
                    Set the input values on the left<br>and click <b>⚡ Predict</b>.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def page_batch(pipe, le, classes):
    st.markdown("### 📂 Batch Prediction from CSV")
    st.markdown(
        "<p style='color:#94A3B8;font-size:0.95rem;margin-top:-8px;'>"
        "Upload a CSV file containing multiple samples. "
        "The app will predict a class for every row.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # CSV format reminder
    with st.expander("📋 Required CSV format", expanded=False):
        st.markdown(
            """
            Your CSV must contain these **three columns** (names are case-sensitive):

            | Column | Unit | Example |
            |---|---|---|
            | `T1_Relaxation_Time(s)` | seconds | `0.9534` |
            | `T2_Relaxation_Time(s)` | seconds | `0.0480` |
            | `Correlation_Time(ns)` | nanoseconds | `12.50` |

            Any additional columns are ignored. Row order is preserved in the output.
            """,
            unsafe_allow_html=True,
        )
        # Download sample CSV
        sample_df = pd.DataFrame({
            "T1_Relaxation_Time(s)":  [0.9534, 0.4856, 1.0821],
            "T2_Relaxation_Time(s)":  [0.0480, 0.0519, 0.0461],
            "Correlation_Time(ns)":   [12.50,  13.80,  15.40],
        })
        csv_sample = sample_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download sample CSV template",
                           csv_sample, "sample_input.csv", "text/csv")

    # File upload
    uploaded = st.file_uploader("Upload input CSV", type=["csv"], key="batch_upload")

    if uploaded is None:
        st.markdown(
            """
            <div style="border:1px dashed #1E3A5F;border-radius:16px;
                        padding:60px 28px;text-align:center;color:#475569;margin-top:16px;">
              <div style="font-size:2.5rem;margin-bottom:12px;">📄</div>
              <div style="font-size:0.95rem;">
                Upload a CSV file above to begin batch prediction.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Load & validate
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    df_clean = df[FEATURE_COLS].dropna()
    n_total  = len(df)
    n_clean  = len(df_clean)

    if n_clean == 0:
        st.error("No valid rows found after removing rows with missing values.")
        return

    # Preview
    st.markdown(f"**Preview** — {n_total:,} rows loaded"
                + (f" ({n_total - n_clean:,} dropped for missing values)" if n_total != n_clean else ""))
    st.dataframe(df_clean.head(10), width="stretch")

    # Predict button
    if st.button("⚡ Run Batch Prediction", type="primary"):
        with st.spinner(f"Predicting {n_clean:,} samples…"):
            pred_lbls, probs = predict_batch(pipe, le, df_clean)

        # Assemble results dataframe
        df_results = df_clean.copy().reset_index(drop=True)
        df_results["Predicted_Class"] = pred_lbls

        if probs is not None:
            for i, cls in enumerate(classes):
                df_results[f"P({cls})"] = probs[:, i].round(4)
            df_results["Top_Confidence(%)"] = (probs.max(axis=1) * 100).round(1)
        else:
            df_results["Top_Confidence(%)"] = None

        st.session_state["batch_results"] = df_results
        st.session_state["batch_probs"]   = probs
        st.session_state["batch_classes"] = list(classes)
        st.success(f"✅ Predicted {n_clean:,} samples successfully!")

    # ── Show results ──────────────────────────────────────────────────────────
    if "batch_results" not in st.session_state:
        return

    df_results = st.session_state["batch_results"]
    probs_arr  = st.session_state["batch_probs"]
    cls_list   = st.session_state["batch_classes"]
    n = len(df_results)

    # ── Summary metric cards
    st.markdown("---")
    st.markdown("#### 📊 Summary")
    counts = df_results["Predicted_Class"].value_counts()
    metric_cols = st.columns(max(len(classes), 3))
    for i, cls in enumerate(cls_list):
        cnt = counts.get(cls, 0)
        pct = cnt / n * 100
        clr = _colour(cls)
        metric_cols[i].markdown(
            f"""
            <div style="background:linear-gradient(135deg,{clr}22,{clr}08);
                        border:1px solid {clr}55;border-radius:12px;padding:16px;
                        text-align:center;">
              <div style="color:{clr};font-size:0.72rem;letter-spacing:0.12em;
                          font-family:'DM Mono',monospace;">{cls.replace('_',' ')}</div>
              <div style="color:#F1F5F9;font-size:2rem;font-weight:800;margin:4px 0;">
                {cnt:,}</div>
              <div style="color:#94A3B8;font-size:0.85rem;">{pct:.1f}% of samples</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if "Top_Confidence(%)" in df_results.columns and df_results["Top_Confidence(%)"].notna().any():
        avg_conf = df_results["Top_Confidence(%)"].mean()
        st.markdown(
            f"""
            <div style="background:#131F2E;border:1px solid #1E3A5F;border-radius:10px;
                        padding:12px 20px;margin-top:12px;display:inline-block;">
              <span style="color:#64748B;font-size:0.82rem;">Average confidence: </span>
              <span style="color:#22D3EE;font-family:'DM Mono',monospace;font-size:1.05rem;
                           font-weight:700;">{avg_conf:.1f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Charts
    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(batch_class_bar(df_results["Predicted_Class"].values, cls_list),
                        width="stretch")
    with c2:
        st.plotly_chart(feature_scatter_batch(df_results, cls_list),
                        width="stretch")

    if probs_arr is not None:
        st.plotly_chart(prob_heatmap(probs_arr, df_results["Predicted_Class"].values, cls_list),
                        width="stretch")

    # ── Results table
    st.markdown("#### 📋 Full Results Table")
    # Colour-coded predicted class column
    def highlight_class(val):
        clr = _colour(val)
        return f"background-color:{clr}33;color:{clr};font-weight:600;"

    styled = df_results.style.map(highlight_class, subset=["Predicted_Class"])
    st.dataframe(styled, width="stretch", height=400)

    # ── Download
    st.markdown("#### ⬇️ Download Results")
    c_dl1, c_dl2 = st.columns(2)
    with c_dl1:
        csv_out = df_results.to_csv(index=False).encode()
        st.download_button(
            "📥 Download full results (.csv)",
            csv_out,
            file_name="nmr_predictions.csv",
            mime="text/csv",
            width="stretch",
        )
    with c_dl2:
        # Summary-only CSV
        summary_df = df_results[["T1_Relaxation_Time(s)", "T2_Relaxation_Time(s)",
                                  "Correlation_Time(ns)", "Predicted_Class",
                                  "Top_Confidence(%)"]].copy()
        summary_csv = summary_df.to_csv(index=False).encode()
        st.download_button(
            "📥 Download compact results (.csv)",
            summary_csv,
            file_name="nmr_predictions_compact.csv",
            mime="text/csv",
            width="stretch",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="NMR Predictor",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Global CSS — dark lab-instrument theme ────────────────────────────────
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

      /* Base */
      html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
        background-color: #0F1923;
        color: #E2E8F0;
      }
      .stApp { background-color: #0F1923; }

      /* Main container */
      .block-container { padding-top: 2rem; }

      /* Sidebar */
      section[data-testid="stSidebar"] {
        background: #0A1219;
        border-right: 1px solid #1E3A5F;
      }
      section[data-testid="stSidebar"] * { color: #CBD5E1; }
      section[data-testid="stSidebar"] h1 { color: #F1F5F9 !important; font-size: 1.25rem !important; }
      section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #94A3B8 !important; font-size: 0.85rem !important;
        text-transform: uppercase; letter-spacing: 0.1em;
      }

      /* Tabs */
      .stTabs [data-baseweb="tab-list"] {
        background: #0A1219;
        border-bottom: 1px solid #1E3A5F;
        border-radius: 0;
        gap: 4px;
      }
      .stTabs [data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: #64748B;
        padding: 12px 24px;
        border-radius: 8px 8px 0 0;
        border-bottom: 3px solid transparent;
        letter-spacing: 0.02em;
      }
      .stTabs [aria-selected="true"] {
        color: #22D3EE !important;
        border-bottom: 3px solid #22D3EE !important;
        background: #131F2E !important;
      }

      /* Inputs */
      [data-testid="stNumberInput"] input,
      [data-testid="stTextInput"] input {
        background: #131F2E !important;
        border: 1px solid #1E3A5F !important;
        border-radius: 8px !important;
        color: #E2E8F0 !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.95rem !important;
      }
      [data-testid="stNumberInput"] input:focus,
      [data-testid="stTextInput"] input:focus {
        border-color: #22D3EE !important;
        box-shadow: 0 0 0 2px rgba(34,211,238,0.15) !important;
      }

      /* Labels */
      [data-testid="stNumberInput"] label,
      [data-testid="stTextInput"] label {
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
      }

      /* Buttons */
      .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0891B2, #22D3EE) !important;
        color: #0F1923 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 28px !important;
        letter-spacing: 0.03em !important;
        transition: all 0.2s !important;
      }
      .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(34,211,238,0.35) !important;
      }
      .stButton > button[kind="secondary"],
      [data-testid="stDownloadButton"] > button {
        background: #131F2E !important;
        color: #94A3B8 !important;
        border: 1px solid #1E3A5F !important;
        border-radius: 10px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
      }
      [data-testid="stDownloadButton"] > button:hover {
        border-color: #22D3EE !important;
        color: #22D3EE !important;
      }

      /* Dataframe */
      [data-testid="stDataFrame"] { border: 1px solid #1E3A5F; border-radius: 10px; }

      /* Success / Error / Info banners */
      [data-testid="stAlert"] {
        border-radius: 10px !important;
        border: 1px solid #1E3A5F !important;
        background: #131F2E !important;
      }

      /* Headers */
      h1 { color: #F1F5F9 !important; font-size: 1.7rem !important; font-weight: 800 !important; }
      h2 { color: #E2E8F0 !important; font-size: 1.25rem !important; font-weight: 700 !important; }
      h3 { color: #94A3B8 !important; font-size: 1rem !important; font-weight: 600 !important; }
      h4 { color: #CBD5E1 !important; font-size: 0.95rem !important; }

      hr { border-color: #1E3A5F !important; margin: 16px 0 !important; }

      /* Expander */
      [data-testid="stExpander"] {
        background: #131F2E !important;
        border: 1px solid #1E3A5F !important;
        border-radius: 10px !important;
      }
      [data-testid="stExpander"] summary { color: #94A3B8 !important; font-weight: 600 !important; }

      /* Upload widget */
      [data-testid="stFileUploader"] {
        background: #131F2E !important;
        border: 1px dashed #1E3A5F !important;
        border-radius: 12px !important;
        padding: 8px !important;
      }
      [data-testid="stFileUploader"] label { color: #94A3B8 !important; }

      /* Metric */
      [data-testid="stMetricLabel"] { color: #64748B !important; font-size: 0.8rem !important; }
      [data-testid="stMetricValue"] { color: #22D3EE !important; font-family: 'DM Mono' !important; }

      /* Scrollbar */
      ::-webkit-scrollbar { width: 6px; height: 6px; }
      ::-webkit-scrollbar-track { background: #0F1923; }
      ::-webkit-scrollbar-thumb { background: #1E3A5F; border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        # Logo / title block
        st.markdown("""
        <div style="padding:20px 4px 8px 4px;">
          <div style="font-size:2rem;margin-bottom:4px;">🧬</div>
          <div style="font-size:1.3rem;font-weight:800;color:#F1F5F9;
                      letter-spacing:-0.01em;line-height:1.2;">
            NMR Predictor
          </div>
          <div style="font-size:0.78rem;color:#475569;margin-top:4px;
                      font-family:'DM Mono',monospace;">
            Enzyme State Classifier
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        # Model upload
        st.markdown("#### Model File")
        model_file = st.file_uploader(
            "Upload .pkl or .pt model",
            type=["pkl", "pt"],
            key="model_upload",
        )

        if model_file is None:
            st.markdown("""
            <div style="background:#131F2E;border:1px solid #1E3A5F;border-radius:10px;
                        padding:14px 16px;margin-top:8px;">
              <div style="color:#64748B;font-size:0.8rem;line-height:1.6;">
                Upload the <code style='color:#22D3EE;'>.pkl</code> or
                <code style='color:#22D3EE;'>.pt</code> model file exported
                from the NMR Classifier training app.
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        # Load model (cached — only deserialised once per file)
        try:
            raw = model_file.read()
            pipe, le, classes = load_model_cached(raw, model_file.name.lower())
            st.session_state["pipe"]    = pipe
            st.session_state["le"]      = le
            st.session_state["classes"] = list(classes)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

        # Model info card
        model_name = model_file.name.replace(".pkl","").replace(".pt","").replace("_"," ")
        st.markdown("---")
        st.markdown("#### Loaded Model")
        st.markdown(
            f"""
            <div style="background:#0A1219;border:1px solid #1E3A5F;border-radius:10px;
                        padding:14px 16px;">
              <div style="color:#22D3EE;font-family:'DM Mono',monospace;font-size:0.85rem;
                          font-weight:500;margin-bottom:10px;">{model_name}</div>
              <div style="color:#64748B;font-size:0.75rem;margin-bottom:4px;">CLASSES ({len(classes)})</div>
              {"".join(f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
                       f'<div style="width:10px;height:10px;border-radius:50%;'
                       f'background:{_colour(c)};flex-shrink:0;"></div>'
                       f'<div style="color:#CBD5E1;font-size:0.82rem;">{c.replace("_"," ")}</div>'
                       f'</div>' for c in classes)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("#### Features")
        for feat, label in zip(FEATURE_COLS, FEAT_LABELS):
            st.markdown(
                f'<div style="color:#475569;font-size:0.78rem;font-family:\'DM Mono\',monospace;'
                f'margin-bottom:4px;">• {label}</div>',
                unsafe_allow_html=True,
            )

    # Pull from session (already validated above)
    pipe    = st.session_state["pipe"]
    le      = st.session_state["le"]
    classes = st.session_state["classes"]

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:24px;">
      <h1 style="margin:0;">NMR Enzyme State Predictor</h1>
      <p style="color:#64748B;font-size:0.95rem;margin-top:4px;">
        Predict enzyme binding state from NMR relaxation parameters —
        single sample or bulk CSV upload.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["🔬  Single Sample", "📂  Batch CSV Prediction"])

    with tab1:
        page_single(pipe, le, classes)

    with tab2:
        page_batch(pipe, le, classes)


if __name__ == "__main__":
    main()
