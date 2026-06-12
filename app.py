"""
app.py
------
Interface Streamlit

Lancer avec : streamlit run app.py
"""

import glob
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
import io


# Normality check
import plotly.graph_objects as go
from scipy.stats import probplot

# MANOVA LDA ->  MANOVA cherche les combinaisons linéaires de tes variables qui maximisent la séparation entre salles.
#  Ces combinaisons s'appellent les fonctions discriminantes — c'est ce que fait une LDA (Linear Discriminant Analysis) et c'est mathématiquement lié.
# Ça va nous permettre de dire quels singaux portent l'effet et quelles salles se ressemblent ou se distinguent.

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pingouin as pg

import matplotlib.pyplot as plt

from umap_utils import (
    compute_umap,
    extract_window,
    get_event_label,
    load_subject,
    aggregate_subjects,   
    run_anova_repeated,   
    run_anova_sexe,       
    run_manova,         
    run_normality_checks,  
    run_normality_checks_per_subject,
    run_friedman,
    run_posthoc_dunn,
    heatmap_participant_salle,
    compute_effect_sizes,
    run_lda_profile,
    plot_radar_salles,
    PHYSIO_COLS,
    DISPLACEMENT_COLS,
)

data_dir = "./data/csv"

# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT & CALCUL (avec cache Streamlit)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Chargement du sujet…")
def cached_load(file_bytes: bytes, filename: str):
    """
    On passe les bytes bruts (hashables) plutôt que l'objet UploadedFile.
    filename sert uniquement à construire l'id du sujet.
    """
    return load_subject(io.BytesIO(file_bytes))  # BytesIO = fichier en mémoire


@st.cache_data(show_spinner="Calcul UMAP…")
def cached_umap(df_json, mode, window_sec, use_physio, use_physio_filtered, use_displacement,
                n_neighbors, min_dist):
    # ← rien ne change ici
    df = pd.read_json(df_json, convert_dates=False)
    df["timestamp"] = df["timestamp"].astype(float)
    df["t_rel"] = df["t_rel"].astype(float)
    df_windowed = extract_window(df, mode=mode, window_sec=window_sec)
    if df_windowed.empty:
        return None
    df_umap = compute_umap(
        df_windowed,
        use_physio=use_physio,
        use_physio_filtered=use_physio_filtered,
        use_displacement=use_displacement,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    df_umap["event_label"] = df_umap.apply(get_event_label, axis=1)
    return df_umap


@st.cache_data(show_spinner="Chargement de tous les sujets...")
def load_all_subjects(data_dir: str) -> dict:
    # ← mode local, rien ne change
    subjects_data = {}
    for filepath in sorted(Path(data_dir).glob("*.csv")):
        subject_name = filepath.stem
        try:
            df = load_subject(str(filepath))
            subjects_data[subject_name] = df
        except Exception as e:
            #st.warning(f"{subject_name} : erreur au chargement ({e})")
            print(f"{subject_name} : erreur au chargement ({e})")
    return subjects_data


def load_all_subjects_uploaded(uploaded_files) -> dict:
    """
    Version pour les fichiers uploadés.
    Pas de @st.cache_data ici — le cache est géré dans cached_load,
    au niveau de chaque fichier individuellement.
    """
    subjects_data = {}
    for f in uploaded_files:
        subject_name = Path(f.name).stem
        try:
            df = cached_load(f.read(), f.name)  # f.read() → bytes hashables
            subjects_data[subject_name] = df
        except Exception as e:
            st.warning(f"{subject_name} : erreur au chargement ({e})")
    return subjects_data


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG PAGE
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VR Ambiances",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Fond sombre scientifique */
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .block-container { padding-top: 1.5rem; }
    h1 { color: #7ec8e3; font-family: 'Courier New', monospace; letter-spacing: 2px; }
    h2, h3 { color: #a8d8ea; }
    .stSelectbox label, .stSlider label, .stRadio label { color: #b0b8c1 !important; }
    /* Badge événement */
    .event-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        margin: 1px;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Contrôles
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("UMAP Explorer")
    st.caption("Réalité virtuelle · Ambiances lumineuses · LIRIS/Lyon1")
    st.divider()

    DATA_DIR = "./data/csv"

    # ── Chargement des données ─────────────────────────────────────────────
    uploaded_files = st.file_uploader(      # pas st.sidebar. — on est déjà dans with st.sidebar
        "Charger les CSV participants",
        accept_multiple_files=True,
        type="csv"
    )

    if uploaded_files:
        subjects = load_all_subjects_uploaded(uploaded_files)
    elif Path(DATA_DIR).exists():
        subjects = load_all_subjects("./data/csv")
    else:
        st.info("Charge les fichiers CSV dans la sidebar pour continuer.")
        st.stop()

    if not subjects:
        st.warning("Aucun sujet chargé.")
        st.stop()

    # ── Sélection du sujet ─────────────────────────────────────────────────
    # subjects est un dict {nom: df} — on navigue dedans directement,
    # plus besoin de csv_files ni de selected_file
    subject_names = sorted(subjects.keys())
    selected_subject = st.selectbox(
        "👤 Sujet",
        options=subject_names,
    )
    
    df = subjects[selected_subject]  # ← le DataFrame, directement
    st.divider()
    # ... reste de ta sidebar (options UMAP, etc.)

    # ── Fenêtrage temporel ────────────────────────────────────────────────────
    st.subheader("⏱️ Fenêtrage")
    window_mode = st.radio(
        "Mode",
        options=["full", "before_watch", "before_sam"],
        format_func=lambda x: {
            "full": "🔵 Session complète",
            "before_watch": "🕰️ Avant montre trouvée",
            "before_sam": "📊 Avant SAM validé"
        }[x],
        help=(
            "**Session complète** : tous les timestamps.\n\n"
            "**Avant montre** : fenêtre fixe avant chaque `premiere_interaction_montre`.\n\n"
            "**Avant SAM** : fenêtre fixe avant chaque `SAM_Validated`.\n\n"
            "Les deux derniers modes permettent de comparer des fenêtres de même durée "
            "pour tous les sujets/salles."
        )
    )
    window_sec = 30.0
    if window_mode != "full":
        window_sec = st.slider(
            "Durée fenêtre (secondes)",
            min_value=5,
            max_value=120,
            value=30,
            step=5,
            help="Durée de la fenêtre temporelle capturée avant l'événement"
        )

    st.divider()

    # ── Features UMAP ─────────────────────────────────────────────────────────
    st.subheader("🔬 Features UMAP")
    use_physio = st.checkbox(
        "Physiologie (EDA, BVP, HR, Temp)",
        value=True,
        help="eda_uS, bvp, hr_bpm, temp_C"
    )
    use_physio_filtered = st.checkbox(
        "Physiologie with filter (EDA_filtered, HRV_RMSSD)",
        value=False,
        help="eda_uS filtré et la différence de battement cardiaque"
    )
    use_displacement = st.checkbox(
        "Déplacement (position + vitesse)",
        value=True,
        help="head_x, head_y, head_z, speed_mps (calculée)"
    )

    st.divider()

    # ── Hyperparamètres UMAP ──────────────────────────────────────────────────
    with st.expander("⚙️ Hyperparamètres UMAP"):
        n_neighbors = st.slider(
            "n_neighbors",
            min_value=5, max_value=100, value=15, step=5,
            help=(
                "Taille du voisinage local.\n"
                "• Petit (5-10) → structure locale fine, clusters petits\n"
                "• Grand (30-50) → structure globale, topologie d'ensemble"
            )
        )
        min_dist = st.slider(
            "min_dist",
            min_value=0.0, max_value=1.0, value=0.1, step=0.05,
            help=(
                "Distance minimale entre points dans l'espace 2D.\n"
                "• Petit (~0.0) → points très compressés en clusters\n"
                "• Grand (~0.8) → points dispersés, vue d'ensemble"
            )
        )

    st.divider()

    # ── Coloration ────────────────────────────────────────────────────────────
    st.subheader("🎨 Coloration")
    color_mode = st.radio(
        "Couleur des points",
        options=["timestamp", "salle", "hr_bpm", "eda_uS"],
        format_func=lambda x: {
            "timestamp": "⏰ Temps (jaune→rouge)",
            "salle": "🚪 Numéro de salle",
            "hr_bpm": "❤️ Rythme cardiaque",
            "eda_uS": "💧 Conductance cutanée (EDA)"
        }[x]
    )

    run_umap = st.button("▶ Calculer UMAP", type="primary", use_container_width=True)



# ══════════════════════════════════════════════════════════════════════════════
# FONCTION CHECK NORMALITY QQ PLOT
# ══════════════════════════════════════════════════════════════════════════════

def make_qqplot(df_agg: pd.DataFrame, col: str, label: str) -> go.Figure:
    """
    Q-Q plot avec Plotly.
    
    probplot() de scipy calcule les quantiles théoriques (loi normale)
    vs quantiles observés. Si les points suivent la droite → normalité.
    """
    #data = df_agg[col].dropna().values
    data = df_agg.groupby("subject")[col].mean().dropna().values

    # osm = quantiles théoriques, osr = quantiles observés (triés)
    (osm, osr), (slope, intercept, r) = probplot(data, dist="norm")
    
    fig = go.Figure()
    
    # Points observés
    fig.add_trace(go.Scatter(
        x=osm, y=osr,
        mode="markers",
        name="Données",
        marker=dict(color="#636EFA", size=6)
    ))
    
    # Droite théorique (si normal, les points la suivent)
    line_x = [min(osm), max(osm)]
    line_y = [slope * x + intercept for x in line_x]
    fig.add_trace(go.Scatter(
        x=line_x, y=line_y,
        mode="lines",
        name="Normale théorique",
        line=dict(color="red", dash="dash")
    ))
    
    fig.update_layout(
        title=f"Q-Q plot — {label}",
        xaxis_title="Quantiles théoriques",
        yaxis_title="Quantiles observés",
        height=350
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# CONTENU PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

st.title("VR · Ambiances Lumineuses")

# Chargement du sujet sélectionné
#df_raw = cached_load(df,subject_names)
df_raw = df

# ── Résumé rapide du sujet ────────────────────────────────────────────────────
col1, col2 = st.columns(2) #col3, col4 = st.columns(4)
duration_s = df_raw["t_rel"].max()
n_events = df_raw["c3d_event"].notna().sum()
n_salles = df_raw["ev_salle"].nunique()
n_sam = (df_raw["c3d_event"] == "SAM_Validated").sum()

col1.metric("⏱️ Durée session", f"{duration_s:.0f} s ({duration_s/60:.1f} min)")
col2.metric("📍 Timestamps", f"{len(df_raw):,}")
#col3.metric("🚪 Salles visitées", int(n_salles))
#col4.metric("📊 SAM remplis", int(n_sam))

st.divider()

# ── Calcul UMAP au clic ───────────────────────────────────────────────────────
if "df_umap" not in st.session_state:
    st.session_state.df_umap = None

if run_umap:
    with st.spinner("Calcul en cours…"):
        df_umap = cached_umap(
            df_raw,
            mode=window_mode,
            window_sec=window_sec,
            use_physio=use_physio,
            use_physio_filtered=use_physio_filtered,
            use_displacement=use_displacement,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )
    if df_umap is None:
        st.error(
            f"Aucun événement `{'premiere_interaction_montre' if window_mode == 'before_watch' else 'SAM_Validated'}` "
            f"trouvé dans ce sujet pour le mode sélectionné."
        )
    else:
        st.session_state.df_umap = df_umap
        st.success(f"UMAP calculé sur {len(df_umap):,} points.")

df_umap = st.session_state.df_umap

if df_umap is None:
    st.info("👈 Configure les options dans la sidebar, puis clique sur **▶ Calculer UMAP**.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION PRINCIPALE — Scatter UMAP 2D
# ══════════════════════════════════════════════════════════════════════════════

tab_umap, tab_timeseries, tab_data, tab_participants, tab_stats, tab_nonparam, tab_sam= st.tabs(["🗺️ UMAP 2D", "📈 Séries temporelles", "📋 Données brutes", "👤 Participants", "📊 Statistiques", "📊 Non-paramétrique", "☹️😀 SAM"])

with tab_umap:
    # ── Paramètres de couleur ─────────────────────────────────────────────────
    if color_mode == "timestamp":
        color_col = "t_rel"
        color_label = "Temps (s depuis début)"
        color_scale = "plasma"   # jaune → violet → rouge : gradient temporel
    elif color_mode == "salle":
        color_col = "ev_salle"
        color_label = "Numéro de salle"
        color_scale = "viridis"
    elif color_mode == "hr_bpm":
        color_col = "hr_bpm"
        color_label = "HR (bpm)"
        color_scale = "RdYlGn_r"  # vert=calme, rouge=élevé
    else:
        color_col = "eda_uS"
        color_label = "EDA (µS)"
        color_scale = "Blues"

    # ev_salle est déjà propagé (ffill+bfill) dans load_subject → pas de NaN ici

    # ── Marqueurs d'événements ────────────────────────────────────────────────
    # On sépare les événements du reste pour les afficher avec des marqueurs distincts
    events_mask = df_umap["c3d_event"].notna() & (df_umap["c3d_event"] != "")
    df_events = df_umap[events_mask].copy()
    df_normal = df_umap[~events_mask].copy()

    symbol_map = {
        "🕰️ Montre": "star",
        "📊 SAM": "diamond",
        "🚪 Nouvelle salle": "cross",
        "—": "circle",
    }

    # ── Figure Plotly ─────────────────────────────────────────────────────────
    fig = go.Figure()

    # Points normaux (sans événement)
    fig.add_trace(go.Scatter(
        x=df_normal["UMAP_1"],
        y=df_normal["UMAP_2"],
        mode="markers",
        marker=dict(
            color=df_normal[color_col],
            colorscale=color_scale,
            size=5,
            opacity=0.7,
            colorbar=dict(title=color_label, thickness=15),
            showscale=True,
        ),
        text=df_normal.apply(
            lambda r: (
                f"t={r['t_rel']:.1f}s<br>"
                f"HR={r.get('hr_bpm', 'N/A'):.0f} bpm<br>"
                f"EDA={r.get('eda_uS', 'N/A'):.4f} µS<br>"
                f"Salle={r.get('ev_salle', 'N/A')}"
            ), axis=1
        ),
        hovertemplate="%{text}<extra></extra>",
        name="Timestamps",
    ))

    # Points événements (avec marqueurs distincts par type)
    event_types = df_events["event_label"].unique()
    event_colors = {
        "🕰️ Montre": "#FFD700",      # or
        "📊 SAM": "#FF6B6B",          # rouge
        "🚪 Nouvelle salle": "#98FB98",  # vert clair
    }
    for ev_type in event_types:
        sub = df_events[df_events["event_label"] == ev_type]
        fig.add_trace(go.Scatter(
            x=sub["UMAP_1"],
            y=sub["UMAP_2"],
            mode="markers",
            marker=dict(
                symbol=symbol_map.get(ev_type, "circle"),
                size=12,
                color=event_colors.get(ev_type, "#FFFFFF"),
                line=dict(color="white", width=1),
            ),
            text=sub.apply(
                lambda r: (
                    f"<b>{r['event_label']}</b><br>"
                    f"t={r['t_rel']:.1f}s<br>"
                    f"Valence={r.get('ev_Valence', 'N/A')}<br>"
                    f"Arousal={r.get('ev_Arousal', 'N/A')}"
                ), axis=1
            ),
            hovertemplate="%{text}<extra></extra>",
            name=ev_type,
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        height=600,
        title=dict(
            text=f"UMAP — {selected_subject} · mode: {window_mode}",
            font=dict(family="Courier New", size=16, color="#7ec8e3")
        ),
        xaxis=dict(title="UMAP 1", gridcolor="#2d333b"),
        yaxis=dict(title="UMAP 2", gridcolor="#2d333b"),
        legend=dict(
            bgcolor="#161b22",
            bordercolor="#2d333b",
            borderwidth=1,
            x=1.10,        # >1 = à droite du graphe (1.0 = bord droit)
            y=0.9,         # aligné en haut
            xanchor="left",
        ),
        hoverlabel=dict(bgcolor="#161b22", font_size=12),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Légende des marqueurs ─────────────────────────────────────────────────
    st.caption(
        "**Marqueurs** : ⭐ Montre trouvée · ◆ SAM validé · ✚ Nouvelle salle  |  "
        f"**Couleur** : {color_label}"
    )

    # ── Info fenêtrage ────────────────────────────────────────────────────────
    if window_mode != "full":
        n_windows = df_umap["window_id"].nunique()
        st.info(
            f"Mode **{window_mode}** · {n_windows} fenêtre(s) de {window_sec}s détectée(s). "
            f"Total : {len(df_umap):,} timestamps."
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Séries temporelles brutes
# ══════════════════════════════════════════════════════════════════════════════
with tab_timeseries:
    st.subheader("Séries temporelles")

    signals = st.multiselect(
        "Signaux à afficher",
        options=["hr_bpm", "eda_uS", "bvp", "temp_C",
                 "UMAP_1", "UMAP_2", 
                 "eda_uS_filtered", "eda_phasic", "eda_tonic",
                 "hr_bpm_filtered", "hrv_rmssd",
                 "eda_uS_filtered_zscore", "hr_bpm_filtered_zscore", "hrv_rmssd_zscore"],
        default=["hr_bpm",  "hr_bpm_filtered", "hrv_rmssd", 
                 "eda_uS", "eda_uS_filtered", "eda_phasic", "eda_tonic"],
    )

    if signals:
        for sig in signals:
            if sig not in df_umap.columns:
                st.warning(f"Colonne `{sig}` absente.")
                continue
            fig_ts = px.line(
                df_umap, x="t_rel", y=sig,
                title=sig,
                template="plotly_dark",
                labels={"t_rel": "Temps (s)", sig: sig},
            )
            # Ajouter les événements comme lignes verticales
            for _, ev_row in df_events.iterrows():
                fig_ts.add_vline(
                    x=ev_row["t_rel"],
                    line_dash="dot",
                    line_color=event_colors.get(ev_row["event_label"], "white"),
                    opacity=0.6,
                    annotation_text=ev_row["event_label"],
                    annotation_position="top",
                    annotation_font_size=9,
                )
            fig_ts.update_layout(
                paper_bgcolor="#0e1117",
                plot_bgcolor="#161b22",
                height=250,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_ts, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Données brutes
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.subheader("Données (avec colonnes UMAP)")
    show_cols = st.multiselect(
        "Colonnes à afficher",
        options=list(df_umap.columns),
        default=["t_rel", "hr_bpm", "eda_uS",
                 "ev_salle", "c3d_event", "event_label",
                 "UMAP_1", "UMAP_2"],
    )
    st.dataframe(
        df_umap[show_cols].reset_index(drop=True),
        use_container_width=True,
        height=400,
    )

    # Export CSV avec colonnes UMAP
    csv_export = df_umap.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Télécharger CSV avec UMAP_1 / UMAP_2",
        data=csv_export,
        file_name=f"{selected_subject.replace('.csv', '')}_umap.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Participants
# ══════════════════════════════════════════════════════════════════════════════
with tab_participants:
    st.subheader("Participants data")

    dfNew = pd.DataFrame({
        "participant": [f"PARTICIPAN{i}" for i in range(1, 48)]
    })

    dfNew["SEXE"] = None
    dfNew["VR"] = None
    dfNew["LUNETTE"] = None
    dfNew["ANOMALIES"] = None
    dfNew["HEIGHT (C3D) (not accurate)"] = None
    dfNew["SESSION DURATION (TOTAL)"] = None
    dfNew = dfNew.set_index("participant")

    dfNew.loc["PARTICIPAN1", "SEXE"] =  "F"
    dfNew.loc["PARTICIPAN2", "SEXE"] =  "H"
    dfNew.loc["PARTICIPAN3", "SEXE"] =  "H"
    dfNew.loc["PARTICIPAN4", "SEXE"] =  "H"
    dfNew.loc["PARTICIPAN5", "SEXE"] =  "H"
    dfNew.loc["PARTICIPAN6", "SEXE"] =  "H"
    dfNew.loc["PARTICIPAN7", "SEXE"] =  "F"
    dfNew.loc["PARTICIPAN8", "SEXE"] =  "F"
    dfNew.loc["PARTICIPAN9", "SEXE"] =  "H"
    dfNew.loc["PARTICIPAN10", "SEXE"] = "H"
    dfNew.loc["PARTICIPAN11", "SEXE"] = "H"
    dfNew.loc["PARTICIPAN12", "SEXE"] = "H"
    dfNew.loc["PARTICIPAN13", "SEXE"] = "H"
    dfNew.loc["PARTICIPAN14", "SEXE"] = "F"
    dfNew.loc["PARTICIPAN15", "SEXE"] = "H"
    dfNew.loc["PARTICIPAN16", "SEXE"] = "H"
    dfNew.loc["PARTICIPAN17", "SEXE"] = "F"
    dfNew.loc["PARTICIPAN18", "SEXE"] = "H"
    dfNew.loc["PARTICIPAN19", "SEXE"] = "F"
    dfNew.loc["PARTICIPAN20", "SEXE"] = "F"
    dfNew.loc["PARTICIPAN21", "SEXE"] = "H"
    dfNew.loc["PARTICIPAN22", "SEXE"] = "H"

    dfNew.loc["PARTICIPAN23", "SEXE"] = "X"
    dfNew.loc["PARTICIPAN24", "SEXE"] = "X"
    dfNew.loc["PARTICIPAN25", "SEXE"] = "X"
    dfNew.loc["PARTICIPAN26", "SEXE"] = "X"

    dfNew.loc["PARTICIPAN27", "SEXE"] = "F"
    dfNew.loc["PARTICIPAN28", "SEXE"] = "H"
    dfNew.loc["PARTICIPAN29", "SEXE"] = "H"
    dfNew.loc["PARTICIPAN30", "SEXE"] = "H"
    dfNew.loc["PARTICIPAN31", "SEXE"] = "H"
    dfNew.loc["PARTICIPAN32", "SEXE"] = "H"
    dfNew.loc["PARTICIPAN33", "SEXE"] = "H"
    dfNew.loc["PARTICIPAN34", "SEXE"] = "F"
    dfNew.loc["PARTICIPAN35", "SEXE"] = "F"
    dfNew.loc["PARTICIPAN36", "SEXE"] = "H"
    dfNew.loc["PARTICIPAN37", "SEXE"] = "F"
    dfNew.loc["PARTICIPAN38", "SEXE"] = "F"
    dfNew.loc["PARTICIPAN39", "SEXE"] = "F"
    dfNew.loc["PARTICIPAN40", "SEXE"] = "F"
    dfNew.loc["PARTICIPAN41", "SEXE"] = "H"
    dfNew.loc["PARTICIPAN42", "SEXE"] = "H"
    dfNew.loc["PARTICIPAN43", "SEXE"] = "H"
    dfNew.loc["PARTICIPAN44", "SEXE"] = "H"
    dfNew.loc["PARTICIPAN45", "SEXE"] = "F"
    dfNew.loc["PARTICIPAN46", "SEXE"] = "F"
    dfNew.loc["PARTICIPAN47", "SEXE"] = "F"

    ##############################################################################
    ##################################### VR  #####################################
    ##############################################################################

    dfNew.loc["PARTICIPAN1", "VR"] = "MORE THAN ONCE"
    dfNew.loc["PARTICIPAN2", "VR"] = "MORE THAN ONCE"
    dfNew.loc["PARTICIPAN3", "VR"] = "FIRST TIME"
    dfNew.loc["PARTICIPAN4", "VR"] = "UN PEU DE VR, MULTIPLE TIMES"
    dfNew.loc["PARTICIPAN5", "VR"] = "MORE THAN ONE, WITHOUT MOVING BEFORE"
    dfNew.loc["PARTICIPAN6", "VR"] = "FIRST TIME VR"
    dfNew.loc["PARTICIPAN7", "VR"] = "PAS MAL DE VR, JEU TYPE ESCAPE GAME"
    dfNew.loc["PARTICIPAN8", "VR"] = "GRANDE UTILISATRICE DE VR"
    dfNew.loc["PARTICIPAN9", "VR"] = "MORE THAN ONCE"
    dfNew.loc["PARTICIPAN10", "VR"] = "1 OU 2, BUT NOT LONG"
    dfNew.loc["PARTICIPAN11", "VR"] = "FIRST TIME, BUT NOT SURE 100%"
    dfNew.loc["PARTICIPAN12", "VR"] = "FIRST TIME"
    dfNew.loc["PARTICIPAN13", "VR"] = "X"
    dfNew.loc["PARTICIPAN14", "VR"] = "A BIT OF VR BEFORE, BUT NOT SURE, ASK JP"
    dfNew.loc["PARTICIPAN15", "VR"] = "A BIT OF VR BEFORE"
    dfNew.loc["PARTICIPAN16", "VR"] = "EXPERIENCED PLAYER"
    dfNew.loc["PARTICIPAN17", "VR"] = "UN PEU DE VR BEFORE, EGYPTE - CONFLUENCE"
    dfNew.loc["PARTICIPAN18", "VR"] = "UN PEU DE VR BEFORE"
    dfNew.loc["PARTICIPAN19", "VR"] = "UN PEU DE VR BEFORE, VILLEURBANNE, CONFLUENCE - VAN GOGH"
    dfNew.loc["PARTICIPAN20", "VR"] = "UN PEU DE VR BEFORE"
    dfNew.loc["PARTICIPAN21", "VR"] = "2 OU 3 FOIS BEFORE, PAS AVEC LE VISION PRO"
    dfNew.loc["PARTICIPAN22", "VR"] = "DEJA FAIT DE LA VR PLUSIEURS FOIS"
    dfNew.loc["PARTICIPAN23", "VR"] = "PAS BCP, PAR CI PAR LA"
    dfNew.loc["PARTICIPAN24", "VR"] = "X"
    dfNew.loc["PARTICIPAN25", "VR"] = "EXPERIENCED USER, DEJA TESTER UNE FOIS LE VISION PRO"
    dfNew.loc["PARTICIPAN26", "VR"] = "EXPERIENCED USER, VISION PRO PAS ENCORE TESTER"
    dfNew.loc["PARTICIPAN27", "VR"] = "WORK WITH VR, VISION PRO NOT YET TRIED"
    dfNew.loc["PARTICIPAN28", "VR"] = "X"
    dfNew.loc["PARTICIPAN29", "VR"] = "DID SOME VR BEFORE, NO AR"
    dfNew.loc["PARTICIPAN30", "VR"] = "EXPERIENCED USER, WORK WITH VR, ALREADY USED VISION PRO BEFORE"
    dfNew.loc["PARTICIPAN31", "VR"] = "EXPERIENCED USER"
    dfNew.loc["PARTICIPAN32", "VR"] = "FIRST/SECOND TIME VR, OR NOT TOO MUCH"
    dfNew.loc["PARTICIPAN33", "VR"] = "FIRST/SECOND TIME VR"
    dfNew.loc["PARTICIPAN34", "VR"] = "ONCE BEFORE, ANOTHER EXP"
    dfNew.loc["PARTICIPAN35", "VR"] = "DID A BIT OF VR BEFORE"
    dfNew.loc["PARTICIPAN36", "VR"] = "EXPERIENCED USER, FIRST TIME VISION PRO"
    dfNew.loc["PARTICIPAN37", "VR"] = "EXPERIENCED USER TOO? NEVER BEFORE VISION PRO"
    dfNew.loc["PARTICIPAN38", "VR"] = "ALREADY DID VR BEFORE, NOT A LOT"
    dfNew.loc["PARTICIPAN39", "VR"] = "VR, ONCE OR NOT MUCH"
    dfNew.loc["PARTICIPAN40", "VR"] = "FIRST TIME IN VR"
    dfNew.loc["PARTICIPAN41", "VR"] = "A BIT OF VR LONG AGO"
    dfNew.loc["PARTICIPAN42", "VR"] = "YES ALREADY DID VR, FEW OCCASIONS IN VR ROOMS"
    dfNew.loc["PARTICIPAN43", "VR"] = "FIRST TIME"
    dfNew.loc["PARTICIPAN44", "VR"] = "ASK JP"
    dfNew.loc["PARTICIPAN45", "VR"] = "ONCE (MONTAGNE RUSSE)"
    dfNew.loc["PARTICIPAN46", "VR"] = "NOT SURE"
    dfNew.loc["PARTICIPAN47", "VR"] = "ASK JP"


    ##############################################################################
    ##############################################################################
    ##############################################################################
    dfNew.loc["PARTICIPAN1",  "LUNETTE"] =  "X"
    dfNew.loc["PARTICIPAN2",  "LUNETTE"] =  "X"
    dfNew.loc["PARTICIPAN3",  "LUNETTE"] =  "X"
    dfNew.loc["PARTICIPAN4",  "LUNETTE"] =  "X"
    dfNew.loc["PARTICIPAN5",  "LUNETTE"] =  "X"
    dfNew.loc["PARTICIPAN6",  "LUNETTE"] =  "X"
    dfNew.loc["PARTICIPAN7",  "LUNETTE"] =  "YES"
    dfNew.loc["PARTICIPAN8",  "LUNETTE"] =  "X"
    dfNew.loc["PARTICIPAN9",  "LUNETTE"] =  "X"
    dfNew.loc["PARTICIPAN10", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN11", "LUNETTE"] = "YES, KEPT HIS GLASSES, CALIBRAGE WORKED, THOUGHT IT WOULD BE BETTER WITHOUT THEM"
    dfNew.loc["PARTICIPAN12", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN13", "LUNETTE"] = "YES, REALLY REALLY BAD SIGHT WITHOUT THEM, HAD TO REMOVE THEM, CALIBRAGE DIDNT WORK"
    dfNew.loc["PARTICIPAN14", "LUNETTE"] = "YES, BUT IT WAS OKAY"
    dfNew.loc["PARTICIPAN15", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN16", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN17", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN18", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN19", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN20", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN21", "LUNETTE"] = "YES, DONT SEE WITHOUT THEM, CALIBRAGE DIDNT WORK?"
    dfNew.loc["PARTICIPAN22", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN23", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN24", "LUNETTE"] = "YES, NO PROBLEM I THINK"
    dfNew.loc["PARTICIPAN25", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN26", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN27", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN28", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN29", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN30", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN31", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN32", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN33", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN34", "LUNETTE"] = "YES, CALIBRAGE DIDNT WORK"
    dfNew.loc["PARTICIPAN35", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN36", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN37", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN38", "LUNETTE"] = "YES, CALIBRAGE DIDNT WORK"
    dfNew.loc["PARTICIPAN39", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN40", "LUNETTE"] = "YES, NO PROBLEM"
    dfNew.loc["PARTICIPAN41", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN42", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN43", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN44", "LUNETTE"] = "YES, REMOVED THEM"
    dfNew.loc["PARTICIPAN45", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN46", "LUNETTE"] = "X"
    dfNew.loc["PARTICIPAN47", "LUNETTE"] = "X"

    ####################################################################################
    ####################################################################################
    ####################################################################################

    dfNew.loc["PARTICIPAN1", "ANOMALIES"] =  "SMALL HEAD"
    dfNew.loc["PARTICIPAN2", "ANOMALIES"] =  "X"
    dfNew.loc["PARTICIPAN3", "ANOMALIES"] =  "DOESN'T KNOW HOW TO PINCH"
    dfNew.loc["PARTICIPAN4", "ANOMALIES"] =  "BIG HEAD"
    dfNew.loc["PARTICIPAN5", "ANOMALIES"] =  "X"
    dfNew.loc["PARTICIPAN6", "ANOMALIES"] =  "BIG HEAD, COULDN'T CALIBRATE"
    dfNew.loc["PARTICIPAN7", "ANOMALIES"] =  "X"
    dfNew.loc["PARTICIPAN8", "ANOMALIES"] =  "AUTISTE, HAD A HEADACHE EVEN BEFORE THE EXP"
    dfNew.loc["PARTICIPAN9", "ANOMALIES"] =  "TALKING IN THE BACKGROUND"
    dfNew.loc["PARTICIPAN10", "ANOMALIES"] = "X"
    dfNew.loc["PARTICIPAN11", "ANOMALIES"] = "X"
    dfNew.loc["PARTICIPAN12", "ANOMALIES"] = "X"
    dfNew.loc["PARTICIPAN13", "ANOMALIES"] = "REALLY BAD INSIGHT WITHOUT GLASSES, COULDN'T SEE A LOT OF THINGS INSIDE THE VIRTUAL ENV"
    dfNew.loc["PARTICIPAN14", "ANOMALIES"] = "X"
    dfNew.loc["PARTICIPAN15", "ANOMALIES"] = "MYOPE, NO COLUMN HR_BPM?"
    dfNew.loc["PARTICIPAN16", "ANOMALIES"] = "X"
    dfNew.loc["PARTICIPAN17", "ANOMALIES"] = "BUG RECORDING EDA"
    dfNew.loc["PARTICIPAN18", "ANOMALIES"] = "BUG EDUROAM"
    dfNew.loc["PARTICIPAN19", "ANOMALIES"] = "SMALL HEAD"
    dfNew.loc["PARTICIPAN20", "ANOMALIES"] = "X"
    dfNew.loc["PARTICIPAN21", "ANOMALIES"] = "DIDNT SEE WITHOUT GLASSES"
    dfNew.loc["PARTICIPAN22", "ANOMALIES"] = "X"
    dfNew.loc["PARTICIPAN23", "ANOMALIES"] = "X"
    dfNew.loc["PARTICIPAN24", "ANOMALIES"] = "X"
    dfNew.loc["PARTICIPAN25", "ANOMALIES"] = "X"
    dfNew.loc["PARTICIPAN26", "ANOMALIES"] = "X"
    dfNew.loc["PARTICIPAN27", "ANOMALIES"] = "CONFONDU LA POCKET WATCH AVEC UN OBJET DE LA PIECE, HAD TO PAUSE THE EXP"
    dfNew.loc["PARTICIPAN28", "ANOMALIES"] = "X"
    dfNew.loc["PARTICIPAN29", "ANOMALIES"] = "X"
    dfNew.loc["PARTICIPAN30", "ANOMALIES"] = "X"
    dfNew.loc["PARTICIPAN31", "ANOMALIES"] = "X"
    dfNew.loc["PARTICIPAN32", "ANOMALIES"] = "X"
    dfNew.loc["PARTICIPAN33", "ANOMALIES"] = "THOUGHT THE EXP WAS OVER BUT IT WAS NOT, MAYBE DIMINISHING PRESENCE"
    dfNew.loc["PARTICIPAN34", "ANOMALIES"] = "GLASSES, FOUGHT AGAINST THE SAM TO VALIDATE ANSWERS, TAKE DATA WITH A GRAIN OF SALT"
    dfNew.loc["PARTICIPAN35", "ANOMALIES"] = "SENSIBLE TO LUMINOUS LIGHTING(S), SAID THE VALENCE WAS OBVIOUSLY LESS HIGH IN LAST TWO ROOMS BECAUSE OF THIS"
    dfNew.loc["PARTICIPAN36", "ANOMALIES"] = "X"
    dfNew.loc["PARTICIPAN37", "ANOMALIES"] = "REALLY DIDNT LIKE DARK AMBIANCES"
    dfNew.loc["PARTICIPAN38", "ANOMALIES"] = "X"
    dfNew.loc["PARTICIPAN39", "ANOMALIES"] = "TOO FAST BETWEEN THE ROOMS ?"
    dfNew.loc["PARTICIPAN40", "ANOMALIES"] = "STARTED RAINING OUTSIDE (REALITY)"
    dfNew.loc["PARTICIPAN41", "ANOMALIES"] = "X"
    dfNew.loc["PARTICIPAN42", "ANOMALIES"] = "X"
    dfNew.loc["PARTICIPAN43", "ANOMALIES"] = "X"
    dfNew.loc["PARTICIPAN44", "ANOMALIES"] = "X"
    dfNew.loc["PARTICIPAN45", "ANOMALIES"] = "X"
    dfNew.loc["PARTICIPAN46", "ANOMALIES"] = "MODE VOYAGE ACTIVATED, HAD TO RE-START THE APP"
    dfNew.loc["PARTICIPAN47", "ANOMALIES"] = "X"

    ####################################################################################
    ####################################################################################
    ####################################################################################


    dfNew.loc["PARTICIPAN1", "HEIGHT (C3D) (not accurate)"] =  "166.54103088378906"
    dfNew.loc["PARTICIPAN2", "HEIGHT (C3D) (not accurate)"] =  "174.10739135742188"
    dfNew.loc["PARTICIPAN3", "HEIGHT (C3D) (not accurate)"] =  "169.87521362304688"
    dfNew.loc["PARTICIPAN4", "HEIGHT (C3D) (not accurate)"] =  "171.6669158935547"
    dfNew.loc["PARTICIPAN5", "HEIGHT (C3D) (not accurate)"] =  "171.5142059326172"
    dfNew.loc["PARTICIPAN6", "HEIGHT (C3D) (not accurate)"] =  "178.26039123535156"
    dfNew.loc["PARTICIPAN7", "HEIGHT (C3D) (not accurate)"] =  "158.0647735595703"
    dfNew.loc["PARTICIPAN8", "HEIGHT (C3D) (not accurate)"] =  "156.65115356445312"
    dfNew.loc["PARTICIPAN9", "HEIGHT (C3D) (not accurate)"] =  "168.141357421875"
    dfNew.loc["PARTICIPAN10", "HEIGHT (C3D) (not accurate)"] = "167.69366455078125"
    dfNew.loc["PARTICIPAN11", "HEIGHT (C3D) (not accurate)"] = "157.19534301757812"
    dfNew.loc["PARTICIPAN12", "HEIGHT (C3D) (not accurate)"] = "170.38894653320312"
    dfNew.loc["PARTICIPAN13", "HEIGHT (C3D) (not accurate)"] = "171.8319854736328"
    dfNew.loc["PARTICIPAN14", "HEIGHT (C3D) (not accurate)"] = "156.3813934326172"
    dfNew.loc["PARTICIPAN15", "HEIGHT (C3D) (not accurate)"] = "165.6721649169922"
    dfNew.loc["PARTICIPAN16", "HEIGHT (C3D) (not accurate)"] = "176.28900146484375"
    dfNew.loc["PARTICIPAN17", "HEIGHT (C3D) (not accurate)"] = "160.2375946044922"
    dfNew.loc["PARTICIPAN18", "HEIGHT (C3D) (not accurate)"] = "X"
    dfNew.loc["PARTICIPAN19", "HEIGHT (C3D) (not accurate)"] = "171.3843536376953"
    dfNew.loc["PARTICIPAN20", "HEIGHT (C3D) (not accurate)"] = "167.97024536132812"
    dfNew.loc["PARTICIPAN21", "HEIGHT (C3D) (not accurate)"] = "166.5"
    dfNew.loc["PARTICIPAN22", "HEIGHT (C3D) (not accurate)"] = "159.09544372558594"

    dfNew.loc["PARTICIPAN23", "HEIGHT (C3D) (not accurate)"] = "159.4922332763672"
    dfNew.loc["PARTICIPAN24", "HEIGHT (C3D) (not accurate)"] = "175.96971130371094"
    dfNew.loc["PARTICIPAN25", "HEIGHT (C3D) (not accurate)"] = "171.26138305664062"
    dfNew.loc["PARTICIPAN26", "HEIGHT (C3D) (not accurate)"] = "168.37408447265625"

    dfNew.loc["PARTICIPAN27", "HEIGHT (C3D) (not accurate)"] = "164.96087646484375"
    dfNew.loc["PARTICIPAN28", "HEIGHT (C3D) (not accurate)"] = "164.43429565429688"
    dfNew.loc["PARTICIPAN29", "HEIGHT (C3D) (not accurate)"] = "166.04739379882812"
    dfNew.loc["PARTICIPAN30", "HEIGHT (C3D) (not accurate)"] = "171.07354736328125"
    dfNew.loc["PARTICIPAN31", "HEIGHT (C3D) (not accurate)"] = "164.69656372070312"
    dfNew.loc["PARTICIPAN32", "HEIGHT (C3D) (not accurate)"] = "169.9633026123047"
    dfNew.loc["PARTICIPAN33", "HEIGHT (C3D) (not accurate)"] = "162.432373046875"
    dfNew.loc["PARTICIPAN34", "HEIGHT (C3D) (not accurate)"] = "167.07733154296875"
    dfNew.loc["PARTICIPAN35", "HEIGHT (C3D) (not accurate)"] = "163.89401245117188"
    dfNew.loc["PARTICIPAN36", "HEIGHT (C3D) (not accurate)"] = "175.5880126953125"
    dfNew.loc["PARTICIPAN37", "HEIGHT (C3D) (not accurate)"] = "165.30455017089844"
    dfNew.loc["PARTICIPAN38", "HEIGHT (C3D) (not accurate)"] = "162.1743927001953"
    dfNew.loc["PARTICIPAN39", "HEIGHT (C3D) (not accurate)"] = "155.13841247558594"
    dfNew.loc["PARTICIPAN40", "HEIGHT (C3D) (not accurate)"] = "155.31643676757812"
    dfNew.loc["PARTICIPAN41", "HEIGHT (C3D) (not accurate)"] = "178.81942749023438"
    dfNew.loc["PARTICIPAN42", "HEIGHT (C3D) (not accurate)"] = "173.3527069091797"
    dfNew.loc["PARTICIPAN43", "HEIGHT (C3D) (not accurate)"] = "170.80160522460938"
    dfNew.loc["PARTICIPAN44", "HEIGHT (C3D) (not accurate)"] = "172.2941436767578"
    dfNew.loc["PARTICIPAN45", "HEIGHT (C3D) (not accurate)"] = "157.71286010742188"
    dfNew.loc["PARTICIPAN46", "HEIGHT (C3D) (not accurate)"] = "156.52554321289062"
    dfNew.loc["PARTICIPAN47", "HEIGHT (C3D) (not accurate)"] = "150.8526153564453"

    ####################################################################################
    ####################################################################################
    ####################################################################################

    dfNew.loc["PARTICIPAN1", "SESSION DURATION (TOTAL)"] =  "730"
    dfNew.loc["PARTICIPAN2", "SESSION DURATION (TOTAL)"] =  "421"
    dfNew.loc["PARTICIPAN3", "SESSION DURATION (TOTAL)"] =  "1132"
    dfNew.loc["PARTICIPAN4", "SESSION DURATION (TOTAL)"] =  "983"
    dfNew.loc["PARTICIPAN5", "SESSION DURATION (TOTAL)"] =  "895"
    dfNew.loc["PARTICIPAN6", "SESSION DURATION (TOTAL)"] =  "1025"
    dfNew.loc["PARTICIPAN7", "SESSION DURATION (TOTAL)"] =  "1113"
    dfNew.loc["PARTICIPAN8", "SESSION DURATION (TOTAL)"] =  "575"
    dfNew.loc["PARTICIPAN9", "SESSION DURATION (TOTAL)"] =  "414"
    dfNew.loc["PARTICIPAN10", "SESSION DURATION (TOTAL)"] = "717"
    dfNew.loc["PARTICIPAN11", "SESSION DURATION (TOTAL)"] = "544"
    dfNew.loc["PARTICIPAN12", "SESSION DURATION (TOTAL)"] = "576"
    dfNew.loc["PARTICIPAN13", "SESSION DURATION (TOTAL)"] = "1145"
    dfNew.loc["PARTICIPAN14", "SESSION DURATION (TOTAL)"] = "590"
    dfNew.loc["PARTICIPAN15", "SESSION DURATION (TOTAL)"] = "807"
    dfNew.loc["PARTICIPAN16", "SESSION DURATION (TOTAL)"] = "532"
    dfNew.loc["PARTICIPAN17", "SESSION DURATION (TOTAL)"] = "307"
    dfNew.loc["PARTICIPAN18", "SESSION DURATION (TOTAL)"] = "594"
    dfNew.loc["PARTICIPAN19", "SESSION DURATION (TOTAL)"] = "730"
    dfNew.loc["PARTICIPAN20", "SESSION DURATION (TOTAL)"] = "864"
    dfNew.loc["PARTICIPAN21", "SESSION DURATION (TOTAL)"] = "710"
    dfNew.loc["PARTICIPAN22", "SESSION DURATION (TOTAL)"] = "699"
    dfNew.loc["PARTICIPAN23", "SESSION DURATION (TOTAL)"] = "506"
    dfNew.loc["PARTICIPAN24", "SESSION DURATION (TOTAL)"] = "448"
    dfNew.loc["PARTICIPAN25", "SESSION DURATION (TOTAL)"] = "919"
    dfNew.loc["PARTICIPAN26", "SESSION DURATION (TOTAL)"] = "407"
    dfNew.loc["PARTICIPAN27", "SESSION DURATION (TOTAL)"] = "587"
    dfNew.loc["PARTICIPAN28", "SESSION DURATION (TOTAL)"] = "451"
    dfNew.loc["PARTICIPAN29", "SESSION DURATION (TOTAL)"] = "573"
    dfNew.loc["PARTICIPAN30", "SESSION DURATION (TOTAL)"] = "569"
    dfNew.loc["PARTICIPAN31", "SESSION DURATION (TOTAL)"] = "431"
    dfNew.loc["PARTICIPAN32", "SESSION DURATION (TOTAL)"] = "474"
    dfNew.loc["PARTICIPAN33", "SESSION DURATION (TOTAL)"] = "1121"
    dfNew.loc["PARTICIPAN34", "SESSION DURATION (TOTAL)"] = "619"
    dfNew.loc["PARTICIPAN35", "SESSION DURATION (TOTAL)"] = "413"
    dfNew.loc["PARTICIPAN36", "SESSION DURATION (TOTAL)"] = "477"
    dfNew.loc["PARTICIPAN37", "SESSION DURATION (TOTAL)"] = "674"
    dfNew.loc["PARTICIPAN38", "SESSION DURATION (TOTAL)"] = "430"
    dfNew.loc["PARTICIPAN39", "SESSION DURATION (TOTAL)"] = "382"
    dfNew.loc["PARTICIPAN40", "SESSION DURATION (TOTAL)"] = "472"
    dfNew.loc["PARTICIPAN41", "SESSION DURATION (TOTAL)"] = "580"
    dfNew.loc["PARTICIPAN42", "SESSION DURATION (TOTAL)"] = "1078"
    dfNew.loc["PARTICIPAN43", "SESSION DURATION (TOTAL)"] = "571"
    dfNew.loc["PARTICIPAN44", "SESSION DURATION (TOTAL)"] = "645"
    dfNew.loc["PARTICIPAN45", "SESSION DURATION (TOTAL)"] = "573"
    dfNew.loc["PARTICIPAN46", "SESSION DURATION (TOTAL)"] = "530"
    dfNew.loc["PARTICIPAN47", "SESSION DURATION (TOTAL)"] = "452"


    dfDisplay = dfNew.replace("X", "—")
    st.dataframe(dfDisplay, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Statistiques
# ══════════════════════════════════════════════════════════════════════════════
with tab_stats:
    if uploaded_files:
        subjects_data = subjects 

    col1, col2, col3 = st.columns(3)
    with col1:
        agg_mode = st.selectbox(
            "Fenêtre d'agrégation",
            options=["full", "pre_sam"],
            format_func=lambda x: {
                "full":    "Toute la salle",
                "pre_sam": "30s avant SAM_Validated"
            }[x]
        )
    with col2:
        agg_window = st.number_input(
            "Durée fenêtre (secondes)",
            min_value=5, max_value=120,
            value=30, step=5,
            disabled=(agg_mode == "full")  # grisé si mode full
        )

    with col3:
        all_subject_ids = sorted(subjects_data.keys())
        selected_subjects = st.multiselect(
            label="Participants inclus dans l'analyse",
            options=all_subject_ids,
            default=all_subject_ids,  # tout sélectionné au départ
            help="Décoche un participant pour l'exclure de tous les calculs statistiques"
        )

        # 3. Garde-fou : on vérifie qu'il reste assez de sujets
        #    (Friedman a besoin d'au moins ~5 sujets pour être interprétable)
        if len(selected_subjects) < 5:
            st.warning("⚠️ Moins de 5 participants sélectionnés — les tests statistiques ne seront pas fiables.")

        # 4. Filtrer subjects_data selon la sélection
        #    dict comprehension : on ne garde que les clés sélectionnées
        subjects_filtered = {
            pid: subjects_data[pid]
            for pid in selected_subjects
        }
        
    st.divider()
    st.subheader("Analyse statistique")
    df_agg = aggregate_subjects(subjects_filtered, dfNew, mode=agg_mode, window_sec=agg_window)
    st.markdown("### Données agrégées (moyenne par sujet × salle)")
    st.dataframe(df_agg, use_container_width=True)
    
    signals_display = {
        "EDA RAW":      "eda_uS_mean",
        "EDA filtered": "eda_uS_filtered_mean",
        "EDA (zscore)": "eda_uS_filtered_zscore_mean",

        "EDA phasic":       "eda_phasic_mean",          
        "EDA phasic zscore":"eda_phasic_zscore_mean",   
        "EDA tonic":        "eda_tonic_mean",            
        "EDA tonic zscore": "eda_tonic_zscore_mean",    
        "EDA driver":        "eda_driver_mean",            
        "EDA driver zscore": "eda_driver_zscore_mean",  

        "SCR count" :"scr_count", 
        "SCR auc"   :"scr_auc",  
        "SDNN"      :"sdnn",
        "PNN50"     :"pnn50",
        "RMSSD2"    :"rmssd2",

        "Speed moyenne": "speed_mean",
        "Speed variabilité (écart type)": "speed_std",
        "Speed robustesse": "speed_median",
        "Distance totale cumulée": "path_length",
        "Proportion du temps immobile": "immobility_ratio",
        "Nombre de bouts immobilité": "immobility_bouts",

    

        "Vitesse angulaire moyenne": "angular_velocity_mean",
        "Vitesse angulaire écart type": "angular_velocity_std",
        
        "Variance totale de l'orientation - yaw": "head_yaw_range",  # amplitude gauche-droite
        "Variance totale de l'orientation - pitch": "head_pitch_range", # amplitude haut-bas

        "HR filtered":  "hr_bpm_filtered_mean",
        "HR (zscore)":  "hr_bpm_filtered_zscore_mean",
        "HRV (RMSSD)":  "hrv_rmssd_mean",
        "HRV (RMSSD zscore)":    "hrv_rmssd_zscore_mean"
    }


    # ── NORMALITY MEAN ────────────────────────────────────────────────────────
             
    st.markdown("### Vérification de la normalité (données filtrés mean - 44 sujets x 1 valeur moyenne)")
    st.caption("Shapiro-Wilk : p > 0.05 → on ne rejette pas H0 (normalité acceptable)")

    df_normality = run_normality_checks(df_agg, signals_display)
    st.dataframe(df_normality, use_container_width=True)

    # Résumé rapide
    n_non_normal = (df_normality["Normal ?"] == "❌").sum()
    n_total = len(df_normality)
    if n_non_normal == 0:
        st.success("Toutes les distributions sont normales → ANOVA paramétrique justifiée ✅")
    elif n_non_normal < n_total / 2:
        st.warning(f"{n_non_normal}/{n_total} distributions non-normales → interprète les résultats ANOVA avec prudence")
    else:
        st.error(f"{n_non_normal}/{n_total} distributions non-normales → envisage Friedman (RM) et Mann-Whitney (sexe)")

    
    # ── ANOVA à mesures répétées : effet salle ─────────────────────────
    st.divider()
    st.markdown("### ANOVA à mesures répétées — effet de la salle")
    st.caption("Chaque sujet passe par les 5 salles → mesures dépendantes → ANOVA RM")
    
    anova_rows = []
    for label, col in signals_display.items():
        res = run_anova_repeated(df_agg, col)
        if "error" in res:
            anova_rows.append({"Signal": label, "F": "—", "p": "—", "Note": res["error"]})
        else:
            anova_rows.append({
                "Signal": label,
                "F": f"{res['F']:.3f}",
                "p": f"{res['p']:.4f}",
                "Significatif": "✅" if res["p"] < 0.05 else "❌",
                "N sujets": res["n_subjects"]
            })
    
    df_anova = pd.DataFrame(anova_rows)
    st.dataframe(df_anova, use_container_width=True)
    
    # ── ANOVA entre-sujets : effet du sexe ────────────────────────────
    #MIS EN COMMENTAIRE
     #st.markdown("### ANOVA entre-sujets — effet du sexe (H vs F)")
    
     #sexe_rows = []
     #for label, col in signals_display.items():
       #  res = run_anova_sexe(df_agg, col)
         #if "error" in res:
           #  sexe_rows.append({"Signal": label, "F": "—", "p": "—"})
         #else:
           #  sexe_rows.append({
             #    "Signal": label,
               #  "F": f"{res['F']:.3f}",
             #    "p": f"{res['p']:.4f}",
              #   "Significatif": "✅" if res["p"] < 0.05 else "❌",
               #  "Moy. H": f"{res['mean_H']:.3f}",
                # "Moy. F": f"{res['mean_F']:.3f}",
               #  "Test": res["test_used"]
             #})
    
     #st.dataframe(pd.DataFrame(sexe_rows), use_container_width=True)
    
    # ── MANOVA ────────────────────────────────────────────────────────
    # st.markdown("### MANOVA — effet de la salle sur EDA + HR + HRV simultanément")
    # Construit le titre dynamiquement depuis les signaux utilisés
    signals_manova = ["eda_tonic_mean",
                      "hrv_rmssd_mean", 

                      "speed_std",
                      "speed_median",
                      "immobility_ratio",
                      "angular_velocity_mean", 
                      "angular_velocity_std", 
                      "head_yaw_range",
                      "sdnn",
                      "scr_count",
                      "scr_auc"
                      ]

    res_manova = run_manova(df_agg, signals_manova)

    signal_names = [
        s.replace("_mean", "")
        .replace("_zscore", " zscore")
        .replace("_", " ")
        for s in res_manova["signals"]
    ]
    
    if "error" not in res_manova:
        signal_names = [
            s.replace("_mean", "")
            .replace("_zscore", " zscore")
            .replace("_", " ")
            for s in res_manova["signals"]
        ]

        manova_title = "MANOVA — effet de la salle sur " + " + ".join(signal_names)
    else:
        manova_title = "MANOVA — effet de la salle"

    st.markdown(f"### {manova_title}")

    if "error" in res_manova:
        st.error(res_manova["error"])
    else:
        df_manova = pd.DataFrame(res_manova["rows"])
        st.dataframe(df_manova, use_container_width=True)
        
        # Résumé sur Pillai uniquement (le plus recommandé)
        pillai = df_manova[ (df_manova["Critère"] == "Pillai's trace") & (df_manova["Effet"] == "C(salle)") ]
        if not pillai.empty:
            p = pillai["p-value"].values[0]
            f = pillai["F"].values[0]
            if p < 0.05:
                st.success(f"Pillai's trace : F={f}, p={p} → effet significatif ✅")
            else:
                st.warning(f"Pillai's trace : F={f}, p={p} → pas d'effet significatif ❌")


    # ── MANOVA VERIFICATION ────────────────────────────────────────────────────────

    features = signals_manova

    X = df_agg[features].dropna()
    y = df_agg.loc[X.index, 'salle']

    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)

    # Variance expliquée par chaque axe discriminant
    print(lda.explained_variance_ratio_)
    # → "l'axe 1 explique 60% de la séparation entre salles"

    # Coefficients : quel signal contribue le plus à la séparation ?
    pd.DataFrame(lda.coef_, columns=features, index=lda.classes_)
    # → "EDA tonic pèse 0.8 sur l'axe qui sépare salle 3 des autres"

    # Pour chaque signal, proportion de variance expliquée par la salle
    # η² = SS_salle / (SS_salle + SS_résidu)
    # Interprétation : 0.01 petit, 0.06 moyen, 0.14 grand (Cohen 1988)
    for signal in signals_manova:
        aov = pg.rm_anova(data=df_agg, dv=signal, within='salle', subject='subject')
        st.write(f"{signal} → η²p = {aov['ng2'].values[0]:.3f}")




    # Explication
    #η²p → "L'EDA tonic a un grand effet (η²=0.18), HR un petit effet (η²=0.03) — l'ambiance lumineuse agit principalement sur l'arousal lent, pas sur le cœur"

    # ── Effect sizes ────────────────────────────────────────────────────────
    #st.subheader("Taille d'effet par signal (η² partiel)")
    
    #signals_for_eta = [
    #    'eda_tonic_mean',
    #    'hrv_rmssd_mean',
        #'scr_count',
        #'scr_amp_mean',
        #'scr_auc',
        #'sdnn',
        #'pnn50'
    #]
    #df_eta = compute_effect_sizes(df_agg, signals=signals_for_eta)
    #st.dataframe(df_eta)
    
    # ── SCR ────────────────────────────────────────────────────────
    #SCR → "La salle 3 déclenche 2× plus de réponses phasiques que la salle 1 — elle est plus stimulante au niveau sympathique"

    st.subheader("SCR — Réactivité phasique par salle")
    # df_agg contient déjà scr_count/scr_amp_mean/scr_auc si aggregate_subjects a été mis à jour
    scr_par_salle = df_agg.groupby('salle')[['scr_count', 'scr_amp_mean', 'scr_auc']].mean().round(3)
    st.dataframe(scr_par_salle)
    
    # --- LDA ---
    #LDA → "LD1 explique 70% de la séparation, porté surtout par EDA tonic — les salles se séparent principalement sur un axe d'activation tonique"

    st.subheader("LDA — Structure discriminante entre salles")
    
    features_lda = [
        'eda_tonic_mean',
        'hrv_rmssd_mean',
        'sdnn',
        'scr_count',
        'scr_auc',
        #'pnn50'
    ]
    lda_results = run_lda_profile(df_agg, features=features_lda)
    
    st.markdown("**Variance expliquée par axe discriminant**")
    st.dataframe(lda_results['variance_par_axe'].to_frame("variance expliquée"))
    ld1_var = lda_results['variance_par_axe'].iloc[0]
    if ld1_var > 0.90:
        st.info(f"LD1 capture {ld1_var:.1%} de la séparation → structure unidimensionnelle")

    #Ce que ça dit : LD1 capture 98.6% de toute la séparation entre salles. 
    # LD2 n'apporte presque rien. En pratique, un seul axe suffit pour discriminer les salles — la structure est unidimensionnelle.
    
    st.markdown("**Contribution des signaux à chaque axe**")
    st.dataframe(lda_results['coefficients'])

    #Ce que tu regardes : les valeurs absolues élevées = signal qui contribue le plus à distinguer cette salle. Ici eda_tonic_zscore à +2.1 pour salle 1 = "la salle 1 se distingue principalement par une EDA tonique élevée".
    #Un signe positif/négatif indique le sens : salle 1 tire vers le haut sur EDA tonic, salle 2 tire vers le bas.
    
    st.markdown("**Profil moyen par salle**")
    st.dataframe(lda_results['profil_salle'])

    #Ce que tu regardes : chaque ligne est la "signature physiologique" d'une salle. 
    # Une salle avec eda_tonic_zscore = +0.8 provoque une activation sympathique tonique supérieure à la baseline. Tu peux directement comparer les salles entre elles.

    #plot_radar_salles(lda_results['profil_salle'])






    # Scatter LD1 vs LD2 (si au moins 2 axes)
    #proj = lda_results['projection']
    #if 'LD2' in proj.columns:
    #    fig, ax = plt.subplots(figsize=(7, 5))
    #    for salle, grp in proj.groupby('salle'):
    #        ax.scatter(grp['LD1'], grp['LD2'], label=f'Salle {salle}', alpha=0.7)
    #    ax.set_xlabel('LD1')
    #    ax.set_ylabel('LD2')
    #    ax.legend()
    #    ax.set_title("Projection LDA — sujets dans l'espace discriminant")
    #    st.pyplot(fig)
    #    plt.close(fig)






    # ── Q-Qplots ────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Q-Q plots")

    signals_display = {
        "EDA RAW":      "eda_uS_mean",
        "EDA filtered": "eda_uS_filtered_mean",
        "EDA (zscore)": "eda_uS_filtered_zscore_mean",
        "EDA phasic":       "eda_phasic_mean",          
        "EDA phasic zscore":"eda_phasic_zscore_mean",   
        "EDA tonic":        "eda_tonic_mean",            
        "EDA tonic zscore": "eda_tonic_zscore_mean",    
        "HR filtered":  "hr_bpm_filtered_mean",
        "HR (zscore)":  "hr_bpm_filtered_zscore_mean",
        "HRV (RMSSD)":  "hrv_rmssd_mean",
        "HRV (RMSSD zscore)":    "hrv_rmssd_zscore_mean"
    }


    row1 = st.columns(3)
    row2 = st.columns(3)
    row3 = st.columns(3)
    row4 = st.columns(2)
    all_cols = row1 + row2 + row3 + row4
    for i, (label, col) in enumerate(signals_display.items()):
        if col in df_agg.columns and i < len(all_cols):
            with all_cols[i]:
                fig = make_qqplot(df_agg, col, label)
                st.plotly_chart(fig, use_container_width=True)

    # ── NORMALITY PAR PARTICIPANT ────────────────────────────────────────────────────────
    #'''   
    #MIS EN COMMENTAIRE

    #st.markdown("### Normalité par participant (données brutes - pleins de points)")
    #df_normality_subj = run_normality_checks_per_subject(subjects_data)


    # Filtre par signal pour ne pas afficher 6×47 lignes d'un coup
    #signal_filter = st.selectbox(
    #    "Signal à afficher",
    #    options=list(signals_display.keys())
    #)

    #col_filter = signals_display[signal_filter]
    # Reconstruit le label matching
    #label_filter = signal_filter

    #df_filtered = df_normality_subj[df_normality_subj["Signal"] == label_filter]
    #st.dataframe(df_filtered, use_container_width=True)

    # Compte rapide des non-normaux
    #n_non_normal = (df_filtered["Normal ?"] == "❌").sum()
    #st.caption(f"{n_non_normal}/{len(df_filtered)} sujets non-normaux pour ce signal")
    
    #'''

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Séries temporelles brutes
# ══════════════════════════════════════════════════════════════════════════════
with tab_nonparam:
    # ── Friedman ──────────────────────────────────────────────────────
    st.markdown("### Friedman — effet de la salle (non-paramétrique)")
    st.caption("Alternative à l'ANOVA RM quand la normalité n'est pas respectée")

    signals_nonparam = {
        "EDA RAW":           "eda_uS_mean",
        "EDA filtered":      "eda_uS_filtered_mean",
        "EDA (zscore)":      "eda_uS_filtered_zscore_mean",

        "EDA phasic":        "eda_phasic_mean",
        "EDA phasic (zscore)": "eda_phasic_zscore_mean",
        "EDA tonic":         "eda_tonic_mean",
        "EDA tonic (zscore)":"eda_tonic_zscore_mean",
        "EDA driver":        "eda_driver_mean",
        "EDA driver (zscore)":"eda_driver_zscore_mean",

        "SCR count" :        "scr_count", 
        "SCR auc"   :        "scr_auc",  
        "SDNN"      :        "sdnn",
        "PNN50"     :        "pnn50",
        "RMSSD2"    :        "rmssd2",

        "HR filtered":       "hr_bpm_filtered_mean",
        "HR (zscore)":       "hr_bpm_filtered_zscore_mean",
        "HRV (RMSSD)":       "hrv_rmssd_mean",
        "HRV zscore":        "hrv_rmssd_zscore_mean",
    }
    
    df_friedman = run_friedman(df_agg, signals_nonparam)
    st.dataframe(df_friedman, use_container_width=True)

    st.markdown("### Post-hoc Dunn — quelles salles diffèrent entre elles ?")
    st.caption("Correction Bonferroni — seulement pertinent pour les signaux significatifs en Friedman")

    signals_dunn_results = {
        "EDA RAW":           "eda_uS_mean",
        "EDA filtered":      "eda_uS_filtered_mean",
        "EDA (zscore)":      "eda_uS_filtered_zscore_mean",
        "EDA phasic":        "eda_phasic_mean",
        "EDA phasic (zscore)": "eda_phasic_zscore_mean",
        "EDA tonic":         "eda_tonic_mean",
        "EDA tonic zscore":  "eda_tonic_zscore_mean",
        "EDA driver":        "eda_driver_mean",
        "EDA driver zscore": "eda_driver_zscore_mean",
        "SCR count" :        "scr_count", 
        "SCR auc"   :        "scr_auc",  
        "PNN50"     :        "pnn50",
    }
    dunn_results = run_posthoc_dunn(df_agg, signals_dunn_results)

    # Sélecteur de signal
    signal_dunn = st.selectbox(
        "Signal",
        options=list(dunn_results.keys()),
        key="dunn_signal"
    )

    if signal_dunn in dunn_results:
        p_matrix = dunn_results[signal_dunn]
        
        # Colore les cellules : vert si p < 0.05, rouge sinon
        def color_pvalue(val):
            if isinstance(val, float):
                if val < 0.001:
                    return "background-color: #1a6b3c; color: white"  # vert foncé
                elif val < 0.01:
                    return "background-color: #2d9e5f; color: white"  # vert moyen
                elif val < 0.05:
                    return "background-color: #5cb88a; color: white"  # vert clair
                else:
                    return "background-color: #8b1a1a; color: white"  # rouge
            return ""
        
        # Renomme les colonnes/index pour plus de clarté
        p_matrix.columns = [f"Salle {int(c)}" for c in p_matrix.columns]
        p_matrix.index   = [f"Salle {int(i)}" for i in p_matrix.index]
        
        # Arrondi pour lisibilité
        p_matrix_display = p_matrix.round(4)
        
        st.dataframe(
            p_matrix_display.style.applymap(color_pvalue),
            use_container_width=True
        )
        
        # Résumé textuel des paires significatives
        st.markdown("**Paires significatives (p < 0.05) :**")
        paires = []
        salles = list(p_matrix.columns)
        for i in range(len(salles)):
            for j in range(i + 1, len(salles)):
                p = p_matrix.iloc[i, j]
                if p < 0.05:
                    paires.append(f"{salles[i]} vs {salles[j]} (p={p:.4f})")
        
        if paires:
            for p in paires:
                st.write(f"✅ {p}")
        else:
            st.write("❌ Aucune paire significative après correction Bonferroni")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 - SAM
# ══════════════════════════════════════════════════════════════════════════════

def extract_sam_scores(subjects_data: dict) -> pd.DataFrame:
    rows = []
    for subject, df in subjects_data.items():
        df_sam = df[df['c3d_event'] == 'SAM_Validated'].copy()
        
        # Dédoublonner : garder uniquement la 1ère ligne par salle
        # (l'événement SAM_Validated peut durer plusieurs frames)
        df_sam = df_sam.drop_duplicates(subset=['ev_salle'], keep='first')
        
        for _, row in df_sam.iterrows():
            rows.append({
                'subject': subject,
                'salle':   row['ev_salle'],
                'valence': row['ev_Valence'],
                'arousal': row['ev_Arousal']
            })
    
    return pd.DataFrame(rows)


with tab_sam:
    st.markdown("## SAM: Self-Assessment Manikin Résultats")
    st.caption("Réponse subjective à chaque salle sur Valence et Arousal")

    # Extraire les scores SAM depuis les données brutes
    df_sam = extract_sam_scores(subjects_filtered) 
    
    for variable in ['valence', 'arousal']:
        st.subheader(variable.capitalize())
        heatmap_participant_salle(
            df_sam,               
            variable=variable,
            subject_col='subject',
            titre=f"Heatmap {variable} — Participant × Salle",
        )