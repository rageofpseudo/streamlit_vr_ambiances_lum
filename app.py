"""
app.py
------
Interface Streamlit

Lancer avec : streamlit run app.py
"""

import os
# Doit être réglé AVANT le premier import qui charge OpenMP (numpy/scipy/umap via
# numba, xgboost...) — sinon le runtime OpenMP est déjà initialisé par numba quand
# xgboost essaie d'initialiser le sien, d'où le crash macOS "OMP: Error #179:
# Function pthread_mutex_init failed" → segfault.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
import umap

from scipy.stats import probplot
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

import pingouin as pg

from feature_glossary import describe_feature

# analyse_questionnaire.py vit un dossier au-dessus (.../analyse/), pas dans project/.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyse_questionnaire import df as df_questionnaire_raw  # noqa: E402 (import après sys.path.insert nécessaire)

from umap_utils import (
    compute_umap,
    extract_window,
    get_event_label,
    load_subject,
    aggregate_subjects,
    run_anova_repeated,
    run_manova_pca,
    run_normality_checks,
    run_levene_test,
    run_friedman,
    run_posthoc_dunn,
    run_key_color_factorial,
    compare_two_salles,
    run_xgboost_importance,
    heatmap_participant_salle,
    run_lda_profile,
    plot_feature_boxplot,
    top_features_par_cluster,
    get_feat_all,
    run_clustering_pipeline,
    afficher_resultats,
    afficher_resultats_participants,
    compare_clustering_to_known_groupings,
    suggest_dbscan_eps,
    extract_series_per_room,
    resample_series,
    normalize_series,
    encode_with_moment,
    build_umap_from_embeddings,
    evaluate_clustering,
    compute_time_to_find_watch,
    compute_signal_sam_correlation,
    extract_c3d_features,
    load_audio_sentiment,
    compute_audio_sentiment_sam_correlation,
)

data_dir = "./data/csv"


FEATURE_GROUPS = {
        "Position (head_x/y/z)":        ["head_x_", "head_y_", "head_z_"],
        "Orientation (pitch/yaw/roll)": ["pitch_", "yaw_", "roll_"],
        "Jerk":                         ["jerk_"],
        "Vitesse":                      ["speed_", "path_", "angular_"],
        "Statisme":                     ["immobility_"],
        "EDA":                          ["eda_tonic_"],
        "HR":                           ["hr_"],
        "HRV":                          ["hrv_"],
}

# ══════════════════════════════════════════════════════════════════════════════
# EXCLUSIONS — qualité des données
# Appliquées globalement (chargement + agrégation) pour que stats et clustering
# travaillent sur le même jeu de données nettoyé.
# ══════════════════════════════════════════════════════════════════════════════
EXCLUSIONS_SUJET = {
    "PARTICIPAN15",          # données suspectes + HR manquant + myopie
    "PARTICIPAN34",          # lunettes + "fought against SAM to validate answers" + score
                             # symptômes outlier (20/48, ~4 écarts-types au-dessus de la
                             # moyenne) — anomalie déjà notée "take data with a grain of salt"
}
EXCLUSIONS_SALLE = {
    ("PARTICIPAN27", 4.0),   # série trop courte
    ("PARTICIPAN32", 4.0),
    ("PARTICIPAN40", 4.0),
    ("PARTICIPAN11", 5.0),   # EDA=0.08µS, probable capteur décroché
}

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN EXPÉRIMENTAL — mapping salle → facteurs lumineux (Q3)
# Salle 1 = baseline diffuse (pas de key/grading) — biaisée par le temps
# d'exploration plus long (découverte de l'environnement).
# ══════════════════════════════════════════════════════════════════════════════
SALLE_KEY = {1: "baseline", 2: "low", 3: "low", 4: "high", 5: "high"}
SALLE_COLOR = {1: "baseline", 2: "red", 3: "blue", 4: "blue", 5: "red"}


def apply_exclusions_subjects(subjects_data: dict) -> dict:
    """Retire les sujets exclus pour cause de données capteur défaillantes."""
    return {k: v for k, v in subjects_data.items() if k not in EXCLUSIONS_SUJET}


def apply_exclusions_salle(df_agg: pd.DataFrame) -> pd.DataFrame:
    """Retire les couples (sujet, salle) exclus pour cause de série trop courte/capteur décroché."""
    mask = ~df_agg.apply(lambda r: (r["subject"], float(r["salle"])) in EXCLUSIONS_SALLE, axis=1)
    return df_agg[mask].reset_index(drop=True)

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


@st.cache_data
def cached_umap(_df, subject: str, source="raw", mode=None, window_sec=30,
                use_displacement=False, new_features=True,
                n_neighbors=15, min_dist=0.1):
    
    df = _df.copy()
    
    if source == "raw":
        # Traitement spécifique au signal brut
        df["timestamp"] = df["timestamp"].astype(float)
        df["t_rel"] = df["t_rel"].astype(float)
        df_windowed = extract_window(df, mode=mode, window_sec=window_sec)
        if df_windowed.empty:
            return None
    else:
        # df_agg : pas de timestamp, pas de fenêtrage — on utilise directement
        df_windowed = df

    df_umap = compute_umap(
        df_windowed,
        source=source,
        use_displacement=use_displacement,
        new_features=new_features,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
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
        subjects_data = load_all_subjects_uploaded(uploaded_files)
    elif Path(DATA_DIR).exists():
        subjects_data = load_all_subjects("./data/csv")
    else:
        st.info("Charge les fichiers CSV dans la sidebar pour continuer.")
        st.stop()

    if not subjects_data:
        st.warning("Aucun sujet chargé.")
        st.stop()

    # Détecte les CSV qui n'ont pas chargé du tout (silencieux sinon — load_all_subjects
    # n'écrit l'erreur que dans le terminal). On compare aux 47 participants attendus
    # (data/participants.csv) pour repérer un échec de chargement.
    participants_attendus = set(pd.read_csv("./data/participants.csv")["participant"])
    participants_non_charges = sorted(participants_attendus - set(subjects_data.keys()))
    if participants_non_charges:
        st.caption(f"⚠️ {len(participants_non_charges)} participant(s) absent(s) — échec de chargement CSV : {participants_non_charges}")

    n_before = len(subjects_data)
    subjects_data = apply_exclusions_subjects(subjects_data)
    if len(subjects_data) < n_before:
        st.caption(f"⚠️ {n_before - len(subjects_data)} sujet(s) exclu(s) (qualité des données) : {sorted(EXCLUSIONS_SUJET)}")

    # ── Sélection du sujet ─────────────────────────────────────────────────
    # subjects est un dict {nom: df} — on navigue dedans directement,
    # plus besoin de csv_files ni de selected_file
    subject_names = sorted(subjects_data.keys())
    selected_subject = st.selectbox(
        "👤 Sujet",
        options=subject_names,
    )

    # Pour changer le cache du participant.
    if st.session_state.get("last_subject") != selected_subject:
        st.session_state.last_subject = selected_subject
        st.session_state.df_umap = None  # invalide l'ancien UMAP
        st.toast(f"Participant changé → UMAP invalidé")  # ← debug visuel

    
    df = subjects_data[selected_subject]  # ← le DataFrame, directement    
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
   
    umap_source = st.radio(
        "Source des données UMAP",
        ["Signal brut (frame par frame)", "Features agrégées (1 point par participant × salle)"],
    )

    use_displacement = st.checkbox(
        "Déplacement (position + vitesse)",
        value=False,
        help="head_x, head_y, head_z"
    )
    new_features = st.checkbox(
        "Most useful features",
        value=True,
        help="head_z_spectral_centroid, head_y_zcr, ..."
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

    run_umap = st.button("▶ Calculer UMAP", type="primary", width="stretch")



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

# Table participants (sexe, expérience VR, anomalies...) — chargée une fois,
# utilisée à la fois pour l'agrégation (sexe) et l'onglet Participants.
df_participants = pd.read_csv("./data/participants.csv").set_index("participant")

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

# ── Calcul df_agg ───────────────────────────────────────────────────────
# Valeurs par défaut pour l'agrégation (peuvent être surchargées dans tab_stats)
if "agg_mode" not in st.session_state:
    st.session_state.agg_mode = "full"
if "agg_window" not in st.session_state:
    st.session_state.agg_window = 30
if "selected_subjects" not in st.session_state:
    st.session_state.selected_subjects = sorted(subjects_data.keys())

subjects_filtered = {
    pid: subjects_data[pid]
    for pid in st.session_state.selected_subjects
}

df_agg = aggregate_subjects(subjects_filtered, df_participants,
                             mode=st.session_state.agg_mode,
                             window_sec=st.session_state.agg_window)
df_agg = apply_exclusions_salle(df_agg)



# ── Calcul UMAP au clic ───────────────────────────────────────────────────────
if "df_umap" not in st.session_state:
    st.session_state.df_umap = None


df_source = df_agg if umap_source == "Features agrégées (1 point par participant × salle)" else df_raw

if run_umap:
    with st.spinner("Calcul en cours…"):
        df_umap = cached_umap(
            df_source,
            subject=selected_subject,  
            source="agg" if umap_source == "Features agrégées (1 point par participant × salle)" else "raw",
            mode=window_mode,
            window_sec=window_sec,
            use_displacement=use_displacement,
            new_features=new_features,
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

tab_umap, tab_timeseries, tab_clustering, tab_foundation, tab_performance, tab_questionnaires, tab_data, tab_participants, tab_stats, tab_nonparam, tab_sam, tab_box_plot, tab_evaluation, tab_audio = st.tabs(["🗺️ UMAP 2D", "📈 Séries temporelles", "Clustering", "🧠 Fondation model", "Performance", "📝 Questionnaires", "📋 Données brutes", "👤 Participants", "📊 Statistiques", "📊 Non-paramétrique", "☹️😀 SAM", "Boxplot", "Evaluation", "🎙️ Audio"])

with tab_umap:
    # ── Paramètres de couleur ─────────────────────────────────────────────────

    is_agg = "_umap_source" in df_umap.columns and df_umap["_umap_source"].iloc[0] == "agg"

    if is_agg:
        if color_mode == "salle":
            color_col = "salle"
            color_label = "Salle"
            color_scale = "viridis"
        elif color_mode == "hr_bpm":
            color_col = "hr_mean"   # dans df_agg c'est hr_mean, pas hr_bpm
            color_label = "HR moyen (bpm)"
            color_scale = "RdYlGn_r"
        else:
            color_col = "eda_tonic_mean"
            color_label = "EDA tonic (moyenne)"
            color_scale = "Blues"
    else:
        # Colonnes disponibles dans df_raw
        if color_mode == "timestamp":
            color_col = "t_rel"
            color_label = "Temps (s depuis début)"
            color_scale = "plasma"
        elif color_mode == "salle":
            color_col = "ev_salle"
            color_label = "Numéro de salle"
            color_scale = "viridis"
        elif color_mode == "hr_bpm":
            color_col = "hr_bpm"
            color_label = "HR (bpm)"
            color_scale = "RdYlGn_r"
        else:
            color_col = "eda_uS"
            color_label = "EDA (µS)"
            color_scale = "Blues"

    # Garde-fou : si la colonne choisie n'existe pas quand même, fallback
    if color_col not in df_umap.columns:
        color_col = df_umap.columns[0]  # première colonne disponible
        color_label = color_col
        
    # ev_salle est déjà propagé (ffill+bfill) dans load_subject → pas de NaN ici

    # ── Marqueurs d'événements ────────────────────────────────────────────────
    if not is_agg and "c3d_event" in df_umap.columns:
        events_mask = df_umap["c3d_event"].notna() & (df_umap["c3d_event"] != "")
    else:
        events_mask = pd.Series(False, index=df_umap.index)

    df_events = df_umap[events_mask].copy()
    df_normal = df_umap[~events_mask].copy()

    # S'assurer que event_label existe dans df_events même si vide
    if "event_label" not in df_events.columns:
        df_events["event_label"] = pd.Series(dtype=str)

    symbol_map = {
        "🕰️ Montre": "star",
        "📊 SAM": "diamond",
        "🚪 Nouvelle salle": "cross",
        "—": "circle",
    }


    # ── Tooltip selon la source ───────────────────────────────────────────────
    if is_agg:
        tooltip = df_normal.apply(
            lambda r: (
                f"subject={r.get('subject', '?')}<br>"
                f"salle={r.get('salle', '?')}<br>"
                f"HR={r.get('hr_mean', float('nan')):.1f}<br>"
                f"EDA={r.get('eda_tonic_mean', float('nan')):.3f}"
            ),
            axis=1
        )
    else:
        tooltip = df_normal.apply(
            lambda r: (
                f"t={r.get('t_rel', float('nan')):.1f}s<br>"
                f"salle={r.get('ev_salle', '?')}<br>"
                f"HR={r.get('hr_bpm', float('nan')):.1f}<br>"
                f"EDA={r.get('eda_uS', float('nan')):.3f}"
            ),
            axis=1
        )


    # Avant le fig.add_trace, selon le type de color_col :
    if df_normal[color_col].dtype == object:
        # Colonne catégorielle (strings) → encoder en entiers
        categories = df_normal[color_col].astype("category")
        color_values = categories.cat.codes          # 0, 1, 2, ...
        tickvals = list(range(len(categories.cat.categories)))
        ticktext = list(categories.cat.categories)   # noms réels
        colorbar_extra = dict(
            tickvals=tickvals,
            ticktext=ticktext,
        )
    else:
        # Colonne numérique → on passe directement
        color_values = df_normal[color_col]
        colorbar_extra = {}
        
    # ── Figure Plotly ─────────────────────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_normal["UMAP_1"],
        y=df_normal["UMAP_2"],
        mode="markers",
        marker=dict(
            color=color_values,
            colorscale=color_scale,
            size=5,
            opacity=0.7,
            colorbar=dict(
                title=color_label,
                thickness=15,
                **colorbar_extra         
            ),
            showscale=True,
        ),
        text=tooltip,  # ← ici, plus le lambda inline
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

    st.plotly_chart(fig, width="stretch")

    # ── Légende des marqueurs ─────────────────────────────────────────────────
    st.caption(
        "**Marqueurs** : ⭐ Montre trouvée · ◆ SAM validé · ✚ Nouvelle salle  |  "
        f"**Couleur** : {color_label}"
    )

    # Juste après le graphique UMAP, dans la même tab ou une tab dédiée
    if df_umap is not None and 'cluster' in df_umap.columns:
        st.subheader("Features les plus discriminantes entre clusters")
        
        # Récupérer les features numériques disponibles (hors colonnes techniques)
        exclude = {'cluster', 'umap_x', 'umap_y', '_umap_source', 
                'subject', 'room', 't_rel', 'window_id'}
        feature_cols = [c for c in df_umap.columns 
                        if c not in exclude and pd.api.types.is_numeric_dtype(df_umap[c])]
        
        if feature_cols:
            df_importance, fig_features = top_features_par_cluster(df_umap, feature_cols, top_n=20)
            st.pyplot(fig_features)  # ← adapter si tu retournes aussi fig
        else:
            st.warning("Aucune feature numérique trouvée dans df_umap.")

    # ── Info fenêtrage ────────────────────────────────────────────────────────
    if window_mode != "full":
        if "window_id" in df_umap.columns:  # ← guard défensif
            n_windows = df_umap["window_id"].nunique()
            st.info(
                f"Mode **{window_mode}** · {n_windows} fenêtre(s) de {window_sec}s détectée(s). "
                f"Total : {len(df_umap):,} timestamps."
            )
        else:
            st.warning(f"Mode **{window_mode}** sélectionné mais `window_id` absent du DataFrame — vérifie `cached_umap`.")

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
                 "eda_uS_filtered_zscore", "hr_bpm_filtered_zscore", "hrv_rmssd_zscore",
                 "head_x", "head_y", "head_z"],
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
            st.plotly_chart(fig_ts, width="stretch")

 
# TAB — CLUSTERING (PCA → UMAP → KMeans / Hiérarchique / DBSCAN)
# ══════════════════════════════════════════════════════════════════════════════
with tab_clustering:
    st.header("Clustering — Profils de mouvement")

    # ── Sélection des features ────────────────────────────────────────────────
    feat_head_all = get_feat_all(df_agg)

    def features_du_groupe(prefixes, all_features):
        return [f for f in all_features if any(f.startswith(p) for p in prefixes)]

    st.subheader("Sélection des features")
    selected_features = []
    cols = st.columns(len(FEATURE_GROUPS))
    for col, (nom, prefixes) in zip(cols, FEATURE_GROUPS.items()):
        feats = features_du_groupe(prefixes, feat_head_all)
        if col.checkbox(nom, value=True, key=f"selection_features_{nom}"):
            selected_features.extend(feats)

    # Prise en compte immédiate des cases cochées — pas de bouton "valider" qui
    # désynchronisait la sélection affichée du clustering réellement calculé.
    if len(selected_features) < 2:
        st.error("Sélectionne au moins 2 features.")
        st.stop()

    feat_head = selected_features
    st.caption(f"{len(feat_head)} features sélectionnées")

    st.divider()



    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Agrégation par salle
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Section 1 — Agrégation par salle")
    st.caption("1 point par (participant × salle) — résumé global de chaque salle")

    exclude_salle1 = st.checkbox(
        "Exclure la salle 1 (baseline diffuse)",
        value=True,
        help=(
            "La salle 1 n'a pas de key/color grading spécifique — c'est la baseline d'exploration. "
            "Les participants y restent souvent plus longtemps (découverte de l'environnement), "
            "ce qui peut écraser le signal lié au design lumineux (key/color, salles 2-5) "
            "dans le clustering. Cocher pour tester la structure uniquement sur 2-5."
        ),
        key="exclude_salle1_s1",
    )

    # Métriques silhouette pour choisir k
    df_s1 = df_agg[["subject", "salle"] + feat_head].copy()
    if exclude_salle1:
        df_s1 = df_s1[df_s1["salle"] != 1].reset_index(drop=True)
        st.caption(f"Salle 1 exclue — {len(df_s1)} lignes restantes (salles 2-5)")
    X_s1_raw = df_s1[feat_head].values
    X_s1 = SimpleImputer(strategy="median").fit_transform(X_s1_raw)
    X_s1_scaled = StandardScaler().fit_transform(X_s1)
    pca_s1 = PCA(n_components=0.95, random_state=42)
    X_s1_pca = pca_s1.fit_transform(X_s1_scaled)
    st.caption(f"{X_s1_pca.shape[1]} composantes PCA sur {len(feat_head)} features")

    col1, col2, col3 = st.columns(3)
    with col1:
        k_agg = st.slider("k (K-Means + Hiérarchique)", 2, 7, 2, key="k_agg")
    with col3:
        min_samples_agg = st.slider("DBSCAN min_samples", 2, 10, 3, key="ms_agg")
    with col2:
        eps_suggestion_agg = suggest_dbscan_eps(X_s1_pca, min_samples_agg)
        # max dynamique : la borne fixe à 5.0 était souvent trop basse pour des
        # données PCA peu denses (peu de points) — on s'assure que le slider
        # couvre au moins 3x le p90 suggéré.
        eps_max_agg = max(5.0, round(eps_suggestion_agg["eps_p90"] * 3, 1))
        eps_agg = st.slider(
            "DBSCAN eps", 0.1, eps_max_agg, round(eps_suggestion_agg["eps_p50"], 1),
            step=0.1, key="eps_agg"
        )
        st.caption(
            f"💡 Suggéré ≈ {eps_suggestion_agg['eps_p50']:.2f} (médiane) "
            f"à {eps_suggestion_agg['eps_p90']:.2f} (90e percentile)."
        )

    ks = range(2, 8)
    sil_km, sil_hc = [], []
    for k in ks:
        sil_km.append(silhouette_score(
            X_s1_pca,
            KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_s1_pca)
        ))
        Z_tmp = linkage(pdist(X_s1_pca, metric="euclidean"), method="ward")
        sil_hc.append(silhouette_score(X_s1_pca, fcluster(Z_tmp, t=k, criterion="maxclust")))

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(list(ks), sil_km, "go-", label="K-Means")
    ax.plot(list(ks), sil_hc, "bo-", label="Hiérarchique")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette")
    ax.legend()
    ax.set_title("Silhouette selon k")
    st.pyplot(fig)

    # ── La salle organise-t-elle l'espace des features, ou cherche-t-on un effet
    # qui n'existe pas sous cette forme ? Comparaison à des regroupements connus
    # plutôt qu'espérer retomber sur 5 clusters par hasard (cf. discussion Q1-Q3).
    st.markdown("##### La salle (ou le design lumineux) organise-t-elle l'espace non supervisé ?")
    n_salles_s1 = df_s1["salle"].nunique()
    groupings_s1 = {
        f"Salle ({n_salles_s1} groupes)":  df_s1["salle"].values,
        "Key (low/high)" if exclude_salle1 else "Key (baseline/low/high)":
            df_s1["salle"].map(lambda s: SALLE_KEY[int(s)]).values,
        "Color grading (red/blue)" if exclude_salle1 else "Color grading (baseline/red/blue)":
            df_s1["salle"].map(lambda s: SALLE_COLOR[int(s)]).values,
    }
    compare_clustering_to_known_groupings(X_s1_pca, ks, sil_km, sil_hc, groupings_s1)

    X_s1_pca, X_s1_umap, labels_s1 = run_clustering_pipeline(
        X_s1_raw, df_s1, k_agg, eps_agg, min_samples_agg, mode_residus=True
    )
    afficher_resultats(df_s1, X_s1_pca, X_s1_umap, labels_s1, id_col="subject")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Fenêtrage glissant
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Section 2 — Fenêtrage glissant")
    st.caption("1 point par fenêtre temporelle — capture la dynamique au sein des salles")

    col1, col2, col3 = st.columns(3)
    with col1:
        window_size = st.slider("Taille fenêtre (frames)", 30, 200, 90, step=10, key="win_size")
    with col2:
        overlap = st.slider("Overlap (frames)", 10, 150, 45, step=5, key="overlap")
    with col3:
        feature_mode = st.radio(
            "Features",
            ["Simplifiées (notebook)", "Complètes (extract_c3d_features)"],
            key="feat_mode"
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        k_win = st.slider("k (K-Means + Hiérarchique)", 2, 7, 2, key="k_win")
    with col2:
        eps_win = st.slider("DBSCAN eps", 0.1, 20.0, 0.5, step=0.1, key="eps_win")
    with col3:
        min_samples_win = st.slider("DBSCAN min_samples", 2, 10, 3, key="ms_win")

    if st.button("Calculer features fenêtrées", type="primary", key="calculer features fenetres"):

        SIGNAUX = ["head_x", "head_y", "head_z", "pitch", "roll", "yaw"]

        def extraire_features_simples(fenetre):
            """Features simplifiées — même logique que le notebook."""
            features = {}
            for signal in SIGNAUX:
                if signal not in fenetre.columns:
                    continue
                x = fenetre[signal].dropna().values
                if len(x) < 10:
                    continue
                features[f"{signal}_mean"]      = np.mean(x)
                features[f"{signal}_std"]       = np.std(x)
                features[f"{signal}_median"]    = np.median(x)
                features[f"{signal}_kurtosis"]  = float(stats.kurtosis(x))
                features[f"{signal}_skewness"]  = float(stats.skew(x))
                features[f"{signal}_iqr"]       = float(stats.iqr(x))
                features[f"{signal}_zcr"]       = int(np.sum(np.diff(np.sign(x)) != 0))
                features[f"{signal}_autocorr"]  = float(np.corrcoef(x[:-1], x[1:])[0, 1])
                fft_vals = np.abs(np.fft.rfft(x))
                freqs    = np.fft.rfftfreq(len(x), d=1/9)
                features[f"{signal}_spectral_centroid"] = float(
                    np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-9)
                )
            return features

        with st.spinner("Extraction des features fenêtrées en cours..."):
            tous_resultats = []

            for subj, df_subj in subjects_data.items():
                # On itère par salle pour garder la trace
                for salle in df_subj["ev_salle"].dropna().unique():
                    if (subj, float(salle)) in EXCLUSIONS_SALLE:
                        continue
                    df_salle = df_subj[df_subj["ev_salle"] == salle].reset_index(drop=True)
                    n = len(df_salle)

                    for debut in range(0, n - window_size, overlap):
                        fin = debut + window_size
                        fenetre = df_salle.iloc[debut:fin]

                        if feature_mode == "Simplifiées (notebook)":
                            feats = extraire_features_simples(fenetre)
                        else:
                            feats = extract_c3d_features(fenetre)

                        feats["subject"]      = subj
                        feats["salle"]        = salle
                        feats["fenetre_debut"] = debut
                        tous_resultats.append(feats)

            df_fenetres = pd.DataFrame(tous_resultats)
            st.session_state.df_fenetres = df_fenetres
            st.success(f"{len(df_fenetres)} fenêtres extraites sur {len(subjects_data)} sujets")

    # Affichage si les features fenêtrées ont été calculées
    if "df_fenetres" in st.session_state:
        df_fenetres = st.session_state.df_fenetres

        meta_cols = ["subject", "salle", "fenetre_debut"]
        feat_win_cols = [c for c in df_fenetres.columns if c not in meta_cols]

        # On filtre pour garder uniquement les features sélectionnées
        # qui existent aussi dans les features fenêtrées
        feat_win_filtered = [f for f in feat_head if f in feat_win_cols]
        if len(feat_win_filtered) < 2:
            # Si la sélection ne matche pas, on prend toutes les features fenêtrées
            feat_win_filtered = feat_win_cols
            st.caption("Les features sélectionnées en haut ne matchent pas — toutes les features fenêtrées sont utilisées.")
        else:
            st.caption(f"{len(feat_win_filtered)} features utilisées (intersection sélection × features fenêtrées)")

        X_win_raw = df_fenetres[feat_win_filtered].values
        X_win_pca, X_win_umap, labels_win = run_clustering_pipeline(
            X_win_raw, df_fenetres, k_win, eps_win, min_samples_win, mode_residus=True
        )
        afficher_resultats(df_fenetres, X_win_pca, X_win_umap, labels_win, id_col="subject")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Profils de participants
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Section 3 — Profils de participants")
    st.caption(
        "1 point par participant (moyenne sur ses 5 salles) — pas de question d'effet "
        "salle ici, on cherche des types de répondants (ex: liés au sexe, à l'expérience VR)."
    )

    df_part_agg = df_agg.groupby("subject")[feat_head].mean().reset_index()
    df_part_agg = df_part_agg.merge(
        df_participants[["SEXE", "VR"]], left_on="subject", right_index=True, how="left"
    )
    X_part_raw = df_part_agg[feat_head].values

    col1, col2, col3 = st.columns(3)
    with col1:
        k_part = st.slider("k (K-Means + Hiérarchique)", 2, 7, 2, key="k_part")
    with col3:
        min_samples_part = st.slider("DBSCAN min_samples", 2, 10, 3, key="ms_part")
    with col2:
        # Suggestion d'eps via k-distance plot (sinon DBSCAN classe souvent tout
        # en outliers avec une valeur par défaut arbitraire — pas un vrai résultat).
        X_part_pca_preview = PCA(n_components=0.95, random_state=42).fit_transform(
            StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X_part_raw))
        )
        eps_suggestion = suggest_dbscan_eps(X_part_pca_preview, min_samples_part)
        eps_max_part = max(5.0, round(eps_suggestion["eps_p90"] * 3, 1))
        eps_part = st.slider(
            "DBSCAN eps", 0.1, eps_max_part, round(eps_suggestion["eps_p50"], 1), step=0.1, key="eps_part"
        )
        st.caption(
            f"💡 Suggéré ≈ {eps_suggestion['eps_p50']:.2f} (médiane) "
            f"à {eps_suggestion['eps_p90']:.2f} (90e percentile) des distances "
            f"au {min_samples_part}e voisin."
        )

    # mode_residus=False : ici un sujet = un seul point, pas de moyenne par sujet à soustraire.
    X_part_pca, X_part_umap, labels_part = run_clustering_pipeline(
        X_part_raw, df_part_agg, k_part, eps_part, min_samples_part, mode_residus=False
    )
    afficher_resultats_participants(
        df_part_agg, X_part_pca, X_part_umap, labels_part, group_cols=["SEXE", "VR"]
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2.5 — FONDATION MODEL (MOMENT) — embeddings de séries temporelles
# ══════════════════════════════════════════════════════════════════════════════
with tab_foundation:
    st.header("Fondation model — embeddings de séries temporelles (MOMENT)")
    st.caption(
        "Au lieu de features faites à la main (mean/std/jerk/...), un modèle pré-entraîné "
        "(MOMENT-1-large, AutonLab) convertit chaque visite de salle en un vecteur dense "
        "(embedding, 1024 dimensions) qui résume toute la dynamique temporelle. On cluster "
        "ensuite ces vecteurs directement (pas leur projection UMAP 2D, pour éviter le piège "
        "de circularité qu'on a démasqué plus tôt) — potentiellement plus sensible à des "
        "patterns non-linéaires que les statistiques agrégées ne capturent pas."
    )
    st.warning(
        "⚠️ Premier calcul : télécharge le modèle (~400 Mo) si pas déjà en cache local, et "
        "encode chaque série — peut prendre plusieurs minutes sur CPU pour ~150-200 séries."
    )

    col1, col2 = st.columns(2)
    with col1:
        signals_fm = st.multiselect(
            "Signaux à encoder",
            options=["head_x", "head_y", "head_z", "pitch", "yaw", "roll",
                     "eda_uS_filtered", "hr_bpm_filtered"],
            default=["head_y", "head_z", "pitch", "yaw"],
            help="Le notebook d'origine trouvait une meilleure silhouette avec le mouvement seul que mélangé à la physio.",
        )
    with col2:
        exclude_salle1_fm = st.checkbox(
            "Exclure la salle 1 (baseline diffuse)", value=True, key="exclude_salle1_fm"
        )

    if st.button("Calculer les embeddings MOMENT", type="primary", key="btn_moment"):
        with st.spinner("Extraction des séries brutes..."):
            series_dict = extract_series_per_room(subjects_data, signals=signals_fm)
            series_dict = {
                (subj, salle): arr
                for (subj, salle), arr in series_dict.items()
                if subj not in EXCLUSIONS_SUJET
                and (subj, float(salle)) not in EXCLUSIONS_SALLE
                and not (exclude_salle1_fm and salle == 1.0)
            }
            st.write(f"{len(series_dict)} séries (participant × salle) conservées")

        with st.spinner("Resampling (512 frames) + normalisation par série..."):
            series_resampled = resample_series(series_dict, target_len=512)
            series_normalized = normalize_series(series_resampled)

        with st.spinner("Encodage MOMENT — peut être long au premier lancement..."):
            embeddings_matrix, keys = encode_with_moment(series_normalized)

        st.session_state.fm_embeddings = embeddings_matrix
        st.session_state.fm_keys = keys
        st.success(f"Embeddings calculés : {embeddings_matrix.shape}")

    if "fm_embeddings" in st.session_state:
        embeddings_matrix = st.session_state.fm_embeddings
        keys = st.session_state.fm_keys
        df_keys = pd.DataFrame(keys, columns=["subject", "salle"])

        # Clustering hiérarchique directement sur les embeddings (distance cosine,
        # standard pour des embeddings de transformer — la direction compte plus
        # que la magnitude), PAS sur leur projection UMAP 2D.
        dist_condensed = pdist(embeddings_matrix, metric="cosine")
        Z_fm = linkage(dist_condensed, method="ward")

        st.subheader("Choix du nombre de clusters")
        df_metrics_fm = evaluate_clustering(embeddings_matrix, Z_fm, k_range=range(2, 8))
        st.dataframe(df_metrics_fm, width="stretch")
        st.caption(
            "Silhouette (métrique cosine) ↑ mieux, Davies-Bouldin ↓ mieux. "
            "Si aucun k ne se détache nettement, c'est qu'il n'y a probablement pas de "
            "structure de cluster franche dans l'espace des embeddings non plus."
        )

        k_fm = st.slider("k (clustering hiérarchique sur les embeddings)", 2, 7, 2, key="k_fm")
        labels_fm = fcluster(Z_fm, t=k_fm, criterion="maxclust")
        df_keys["cluster"] = labels_fm

        df_umap_fm = build_umap_from_embeddings(embeddings_matrix, keys)
        df_umap_fm["cluster"] = labels_fm

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 6))
            scatter = ax.scatter(
                df_umap_fm["UMAP_1"], df_umap_fm["UMAP_2"],
                c=labels_fm, cmap="viridis", s=80, alpha=0.8
            )
            plt.colorbar(scatter, ax=ax, label="Cluster")
            ax.set_title("UMAP des embeddings MOMENT — couleur cluster")
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            fig2, ax2 = plt.subplots(figsize=(7, 6))
            salles_cat = pd.Categorical(df_keys["salle"])
            scatter2 = ax2.scatter(
                df_umap_fm["UMAP_1"], df_umap_fm["UMAP_2"],
                c=salles_cat.codes, cmap="tab10", s=80, alpha=0.8
            )
            handles, _ = scatter2.legend_elements()
            ax2.legend(handles, salles_cat.categories, title="Salle")
            ax2.set_title("UMAP des embeddings MOMENT — couleur salle")
            st.pyplot(fig2)
            plt.close(fig2)

        st.subheader("Validation — est-ce lié à la salle/key/color, ou autre chose ?")
        groupings_fm = {f"Salle ({df_keys['salle'].nunique()} groupes)": df_keys["salle"].values}
        if exclude_salle1_fm:
            groupings_fm["Key (low/high)"] = df_keys["salle"].map(lambda s: SALLE_KEY[int(s)]).values
            groupings_fm["Color (red/blue)"] = df_keys["salle"].map(lambda s: SALLE_COLOR[int(s)]).values

        ks_fm = list(range(2, 8))
        sil_km_fm = [
            silhouette_score(
                embeddings_matrix,
                KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(embeddings_matrix),
                metric="cosine",
            )
            for k in ks_fm
        ]
        sil_hc_fm = df_metrics_fm["silhouette"].tolist()
        st.caption(
            "⚠️ La comparaison ci-dessous calcule la silhouette avec la distance euclidienne "
            "par défaut (fonction partagée avec les autres onglets), alors que le clustering "
            "ci-dessus utilise la distance cosine (standard pour ces embeddings) — les chiffres "
            "absolus ne sont donc pas directement comparables d'un onglet à l'autre, mais la "
            "comparaison interne (salle vs key vs color, ici) reste valide."
        )
        compare_clustering_to_known_groupings(embeddings_matrix, ks_fm, sil_km_fm, sil_hc_fm, groupings_fm)

        st.subheader("Crosstab salle × cluster")
        st.dataframe(pd.crosstab(df_keys["salle"], df_keys["cluster"], margins=True), width="stretch")

        st.subheader("Sujets par cluster")
        for c in sorted(df_keys["cluster"].unique()):
            subs = sorted(df_keys.loc[df_keys["cluster"] == c, "subject"].unique())
            preview = ", ".join(subs[:30]) + (" ..." if len(subs) > 30 else "")
            st.write(f"**Cluster {c}** ({len(subs)} sujets distincts, {(df_keys['cluster'] == c).sum()} lignes) : {preview}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB — PERFORMANCE : temps pour trouver la montre
# ══════════════════════════════════════════════════════════════════════════════
with tab_performance:
    st.header("Performance — temps pour trouver la montre")
    st.caption(
        "Latence entre l'entrée dans la salle (ou le début de session pour la salle 1) "
        "et le premier événement `premiere_interaction_montre` — une mesure de "
        "performance/efficacité d'exploration, indépendante de la physio et du mouvement "
        "déjà testés."
    )

    df_watch = compute_time_to_find_watch(subjects_data)

    if df_watch.empty:
        st.warning("Aucun événement `premiere_interaction_montre` détecté dans les données chargées.")
        st.stop()

    st.subheader("Tableau participant × salle")
    pivot_watch = df_watch.pivot_table(index="subject", columns="salle", values="temps_trouver_montre")
    st.dataframe(pivot_watch.round(1), width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Par participant (à travers ses salles)")
        stats_subject = df_watch.groupby("subject")["temps_trouver_montre"].agg(["mean", "var", "std", "count"])
        stats_subject.columns = ["Moyenne (s)", "Variance", "Écart-type (s)", "N salles"]
        st.dataframe(stats_subject.round(2).sort_values("Moyenne (s)", ascending=False), width="stretch")
    with col2:
        st.subheader("Par salle (à travers les participants)")
        stats_salle = df_watch.groupby("salle")["temps_trouver_montre"].agg(["mean", "var", "std", "count"])
        stats_salle.columns = ["Moyenne (s)", "Variance", "Écart-type (s)", "N participants"]
        st.dataframe(stats_salle.round(2), width="stretch")

    st.subheader("Distribution par salle")
    plot_feature_boxplot(df_watch.rename(columns={"temps_trouver_montre": "temps_trouver_montre"}), "temps_trouver_montre", title="Temps pour trouver la montre par salle")

    st.divider()
    st.subheader("Effet de la salle sur ce temps — ANOVA à mesures répétées")
    res_watch_anova = run_anova_repeated(df_watch, "temps_trouver_montre")
    if "error" in res_watch_anova:
        st.error(res_watch_anova["error"])
    else:
        st.write(f"F={res_watch_anova['F']:.3f}, p={res_watch_anova['p']:.4f}, N sujets={res_watch_anova['n_subjects']}")
        if res_watch_anova["p"] < 0.05:
            st.success("Effet significatif de la salle sur le temps pour trouver la montre ✅")
        else:
            st.warning("Pas d'effet significatif de la salle sur ce temps ❌")

    st.subheader("Design Q3 — key × color sur ce temps")
    res_watch_factorial = run_key_color_factorial(df_watch, "temps_trouver_montre", SALLE_KEY, SALLE_COLOR, subject_col="subject")
    rows_watch = []
    for factor_name, r in res_watch_factorial.items():
        if "error" in r:
            rows_watch.append({"Facteur": factor_name, "Note": r["error"]})
        else:
            lvl_lo, lvl_hi = r["levels"]
            rows_watch.append({
                "Facteur": factor_name,
                f"Moyenne {lvl_lo}": round(r[f"mean_{lvl_lo}"], 2),
                f"Moyenne {lvl_hi}": round(r[f"mean_{lvl_hi}"], 2),
                "p (Wilcoxon)": round(r["p"], 4),
                "Significatif": "✅" if r["p"] < 0.05 else "❌",
                "N sujets": r["n_subjects"],
            })
    st.dataframe(pd.DataFrame(rows_watch), width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB — QUESTIONNAIRES (NASA-TLX, Symptômes, Ownership/Agency/Change, IPQ)
# ══════════════════════════════════════════════════════════════════════════════
with tab_questionnaires:
    st.header("Questionnaires — NASA-TLX, Symptômes, Ownership/Agency/Change, IPQ")
    st.caption(
        "Données issues de analyse_questionnaire.py — un score par participant pour "
        "l'ensemble de la session (pas par salle, contrairement au SAM). On vérifie ici "
        "si ces scores subjectifs sont liés aux signaux physio, comme dans l'article "
        "(Saha et al. 2025) qui corrèle CF1/CF2 avec une mesure combinée IPQ/SSQ par "
        "participant — mais à l'échelle de la session entière, faute de questionnaire par salle."
    )

    df_questionnaire = df_questionnaire_raw.copy()
    df_questionnaire["subject"] = df_questionnaire["Participant"].apply(
        lambda p: f"PARTICIPAN{int(str(p).lstrip('P'))}"
    )
    n_quest_before = len(df_questionnaire)
    df_questionnaire = df_questionnaire[~df_questionnaire["subject"].isin(EXCLUSIONS_SUJET)]
    if len(df_questionnaire) < n_quest_before:
        st.caption(f"⚠️ {n_quest_before - len(df_questionnaire)} sujet(s) exclu(s) (qualité des données) : {sorted(EXCLUSIONS_SUJET)}")

    score_cols_quest = [
        "Score_NASA_TLX", "Score_Symptomes", "Score_Ownership", "Score_Agency",
        "Score_Change", "Score_IPQ_GP", "Score_IPQ_SP", "Score_IPQ_INV", "Score_IPQ_REAL",
    ]
    score_cols_quest = [c for c in score_cols_quest if c in df_questionnaire.columns]

    st.subheader("Scores par participant")
    st.dataframe(
        df_questionnaire[["subject"] + score_cols_quest].set_index("subject").round(2),
        width="stretch",
    )

    st.subheader("Résumé descriptif — tous les questionnaires")
    st.caption(
        "Échelle théorique entre parenthèses. Si la moyenne est proche du maximum pour "
        "presque tout le monde, ça réduit la variance disponible pour qu'un signal physio "
        "puisse la 'suivre' statistiquement (range restriction) — mais si min/max couvrent "
        "une bonne partie de l'échelle, la variance reste exploitable malgré une moyenne haute."
    )
    echelles_quest = {
        "Score_IPQ_GP":     "1-7 (présence générale)",
        "Score_IPQ_SP":     "1-7 (présence spatiale)",
        "Score_IPQ_INV":    "1-7 (involvement)",
        "Score_IPQ_REAL":   "1-7 (réalisme)",
        "Score_NASA_TLX":   "0-9 (charge de travail)",
        "Score_Symptomes":  "0-48 (type SSQ — plus haut = plus de symptômes)",
        "Score_Ownership":  "1-7 (appartenance corporelle)",
        "Score_Agency":     "1-7 (contrôle perçu)",
        "Score_Change":     "1-7 (illusion de changement)",
    }
    cols_quest_dispo = [c for c in echelles_quest if c in df_questionnaire.columns]
    if cols_quest_dispo:
        stats_quest = df_questionnaire[cols_quest_dispo].agg(["mean", "std", "min", "max"]).T
        stats_quest.columns = ["Moyenne", "Écart-type", "Min", "Max"]
        stats_quest.insert(0, "Échelle", [echelles_quest[c] for c in cols_quest_dispo])
        st.dataframe(stats_quest.round(2), width="stretch")
        st.caption(
            "⚠️ Les items IPQ_SP3, IPQ_REAL1 et IPQ_REAL3 ont un libellé négatif dans le "
            "questionnaire (cf. commentaires dans analyse_questionnaire.py, ex: 'here 7 is not "
            "real at all') — vérifie qu'ils sont bien inversés avant la moyenne si ce n'est pas "
            "déjà fait, sinon le score peut être faussé pour ces sous-échelles."
        )

    if "Score_Symptomes" in df_questionnaire.columns:
        idx_max_symptomes = df_questionnaire["Score_Symptomes"].idxmax()
        row_max = df_questionnaire.loc[idx_max_symptomes]
        subj_max = row_max["subject"]
        anomalie = (
            df_participants.loc[subj_max, "ANOMALIES"]
            if subj_max in df_participants.index and "ANOMALIES" in df_participants.columns
            else "—"
        )
        st.info(
            f"🔍 Score de symptômes le plus élevé : **{subj_max}** "
            f"(Score_Symptomes = {row_max['Score_Symptomes']:.0f}/48). "
            f"Anomalie déjà documentée dans le tableau Participants : **{anomalie}**"
        )

    # Mesure combinée façon article (IPQ / SSQ) — IPQ_SP (présence spatiale) sur
    # Symptômes (+1 pour éviter une division par 0 quand un participant n'a aucun symptôme).
    if "Score_IPQ_SP" in df_questionnaire.columns and "Score_Symptomes" in df_questionnaire.columns:
        df_questionnaire["Feedback_combine_IPQ_SSQ"] = (
            df_questionnaire["Score_IPQ_SP"] / (df_questionnaire["Score_Symptomes"] + 1)
        )
        score_cols_quest = score_cols_quest + ["Feedback_combine_IPQ_SSQ"]

    st.divider()
    st.subheader("Corrélation avec les signaux physio (CF1/CF2 inclus)")
    st.caption(
        "Moyenne du signal physio sur toutes les salles, par participant, corrélée "
        "(Spearman) avec chaque score de questionnaire (session entière) à travers les "
        "~40 participants. CF1/CF2 = features de l'article (Saha et al. 2025)."
    )

    physio_signals_quest = {
        "EDA tonique (moyenne)":             "eda_tonic_mean",
        "EDA tonique (spectral centroid)":   "eda_tonic_spectral_centroid",
        "CF1 (présence, Saha et al.)":       "cf1_presence",
        "CF2 (présence, Saha et al.)":       "cf2_presence",
        "HR (moyenne)":                      "hr_mean",
        "HRV RMSSD (moyenne)":               "hrv_rmssd_mean",
        "HRV LF/HF":                         "hrv_lf_hf",
        "SCR rate":                          "scr_rate",
    }
    physio_signals_quest = {k: v for k, v in physio_signals_quest.items() if v in df_agg.columns}

    df_physio_participant = df_agg.groupby("subject")[list(physio_signals_quest.values())].mean().reset_index()
    df_merged_quest = df_questionnaire.merge(df_physio_participant, on="subject", how="inner")

    rows_quest_corr = []
    for score_col in score_cols_quest:
        for label, col in physio_signals_quest.items():
            d = df_merged_quest[[score_col, col]].dropna()
            if len(d) < 5 or d[col].std() < 1e-9 or d[score_col].std() < 1e-9:
                continue
            r, p = stats.spearmanr(d[score_col], d[col])
            rows_quest_corr.append({
                "Score questionnaire": score_col,
                "Signal physio":       label,
                "r":                   round(r, 3),
                "p-value":             round(p, 4),
                "Significatif":        "✅" if p < 0.05 else "❌",
                "N":                   len(d),
            })

    df_quest_corr = pd.DataFrame(rows_quest_corr)
    if not df_quest_corr.empty:
        n_tests_quest = len(df_quest_corr)
        seuil_quest = 0.05 / n_tests_quest
        df_quest_corr["Significatif (Bonferroni)"] = df_quest_corr["p-value"].apply(
            lambda p: "✅" if p < seuil_quest else "❌"
        )
        st.dataframe(df_quest_corr.sort_values("p-value"), width="stretch")
        st.caption(f"{n_tests_quest} tests → seuil Bonferroni ajusté = {seuil_quest:.5f}")
    else:
        st.info("Pas assez de données pour calculer ces corrélations.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Données brutes
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.subheader("Données (avec colonnes UMAP)")

    available_cols = df_umap.columns.tolist()

    # Colonnes qu'on aimerait afficher par défaut selon la source
    if is_agg:
        preferred_defaults = ["subject", "salle", "hr_mean", "eda_tonic_mean"]
    else:
        preferred_defaults = ["t_rel", "ev_salle", "hr_bpm", "eda_uS"]

    # On ne garde que ceux qui existent vraiment
    safe_defaults = [c for c in preferred_defaults if c in available_cols]

    show_cols = st.multiselect(
        "Colonnes à afficher",
        options=list(df_umap.columns),
        default=safe_defaults
    )
    st.dataframe(
        df_umap[show_cols].reset_index(drop=True),
        width="stretch",
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

    dfNew = df_participants

    dfDisplay = dfNew.replace("X", "—")
    st.dataframe(dfDisplay, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Statistiques
# ══════════════════════════════════════════════════════════════════════════════
with tab_stats:
    exclude_salle1_stats = st.checkbox(
        "Exclure la salle 1 (baseline diffuse) de la MANOVA/ANOVA",
        value=False,
        help=(
            "La salle 1 n'a pas de key/color grading et les participants y restent "
            "souvent plus longtemps (découverte de l'environnement) — elle peut écraser "
            "le signal lié au vrai design 2×2 (key×color, salles 2-5). Cocher pour tester "
            "l'effet salle uniquement sur 2-5."
        ),
        key="exclude_salle1_stats",
    )

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

        deselected = sorted(set(all_subject_ids) - set(selected_subjects))
        if deselected:
            st.caption(f"🔲 {len(deselected)} participant(s) décoché(s) dans ce multiselect : {deselected}")

        # 4. Filtrer subjects_data selon la sélection
        #    dict comprehension : on ne garde que les clés sélectionnées
        subjects_filtered = {
            pid: subjects_data[pid]
            for pid in selected_subjects
        }
        
      
    # Quand les paramètres changent → mettre à jour session_state
    if agg_mode != st.session_state.agg_mode or agg_window != st.session_state.agg_window:
        st.session_state.agg_mode = agg_mode
        st.session_state.agg_window = agg_window
        st.rerun()  # Streamlit recalcule tout depuis le début avec les nouvelles valeurs
    
    if selected_subjects != st.session_state.selected_subjects:
        st.session_state.selected_subjects = selected_subjects
        st.rerun()

    st.dataframe(df_agg, width="stretch")
    st.divider()
    st.subheader("Analyse statistique")

    df_agg_stats = df_agg
    if exclude_salle1_stats:
        df_agg_stats = df_agg_stats[df_agg_stats["salle"] != 1].reset_index(drop=True)
        st.caption(f"Salle 1 exclue — {len(df_agg_stats)} lignes restantes (salles 2-5)")

    # df_agg = aggregate_subjects(subjects_filtered, dfNew, mode=agg_mode, window_sec=agg_window)
    st.markdown("### Données agrégées (moyenne par sujet × salle)")
    st.dataframe(df_agg_stats, width="stretch")

    COLONNES_EXCLURE = {"participant", "salle", "fichier", "fenetre_debut"}

    colonnes_features = [
        c for c in df_agg_stats.columns
        if c not in COLONNES_EXCLURE
        and pd.api.types.is_numeric_dtype(df_agg_stats[c])  # ← filtre par type
    ]

    signals_display = {col: col for col in colonnes_features}

    colonnes_manquantes = [col for col in signals_display.values() if col not in df_agg_stats.columns]
    if colonnes_manquantes:
        st.warning(f"Colonnes absentes de df_agg_stats : {colonnes_manquantes}")
        st.stop()
    else:
         st.write("Colonnes disponibles :", sorted(df_agg_stats.columns.tolist()))
    

    # ── NORMALITY MEAN ────────────────────────────────────────────────────────
             
    st.markdown("### Vérification de la normalité (données filtrés mean - 44 sujets x 1 valeur moyenne)")
    st.caption("Shapiro-Wilk : p > 0.05 → on ne rejette pas H0 (normalité acceptable)")

    df_normality = run_normality_checks(df_agg_stats, signals_display)
    st.dataframe(df_normality, width="stretch")

    # Résumé rapide
    n_non_normal = (df_normality["Normal ?"] == "❌").sum()
    n_total = len(df_normality)
    if n_non_normal == 0:
        st.success("Toutes les distributions sont normales → ANOVA paramétrique justifiée ✅")
    elif n_non_normal < n_total / 2:
        st.warning(f"{n_non_normal}/{n_total} distributions non-normales → interprète les résultats ANOVA avec prudence")
    else:
        st.error(f"{n_non_normal}/{n_total} distributions non-normales → envisage Friedman (RM) et Mann-Whitney (sexe)")

    # ── HOMOGÉNÉITÉ DES VARIANCES (Levene) ───────────────────────────────────
    st.markdown("### Test d'homogénéité des variances — Levene")
    st.caption(
        "H0 = les variances sont égales entre les salles. p > 0.05 → variances "
        "homogènes (hypothèse de l'ANOVA respectée). p < 0.05 → variances "
        "hétérogènes → interpréter l'ANOVA/MANOVA avec prudence, ou préférer "
        "Friedman (non-paramétrique, pas cette hypothèse)."
    )

    df_levene = run_levene_test(df_agg_stats, signals_display)
    st.dataframe(df_levene, width="stretch")

    n_heterogene = (df_levene["Variances homogènes ?"] == "❌").sum()
    n_total_levene = len(df_levene)
    if n_total_levene == 0:
        st.info("Aucun signal testé.")
    elif n_heterogene == 0:
        st.success("Variances homogènes partout → hypothèse de l'ANOVA respectée ✅")
    elif n_heterogene < n_total_levene / 2:
        st.warning(f"{n_heterogene}/{n_total_levene} signaux avec variances hétérogènes → interprète l'ANOVA/MANOVA sur ceux-ci avec prudence")
    else:
        st.error(f"{n_heterogene}/{n_total_levene} signaux avec variances hétérogènes → préfère Friedman/tests non-paramétriques pour la majorité des signaux")

    # ── ANOVA à mesures répétées : effet salle ─────────────────────────
    st.divider()
    st.markdown("### ANOVA à mesures répétées — effet de la salle")
    st.caption("Chaque sujet passe par les 5 salles → mesures dépendantes → ANOVA RM")
    
    anova_rows = []
    for label, col in signals_display.items():
        res = run_anova_repeated(df_agg_stats, col)
        if "error" in res:
            anova_rows.append({"Signal": label, "F": "—", "p": "—", "Note": res["error"]})
        else:
            # η²p (partiel) à partir de F et des degrés de liberté — formule équivalente
            # à SS_effet / (SS_effet + SS_résidu) pour une ANOVA à 1 facteur.
            eta2 = (res["F"] * res["df_num"]) / (res["F"] * res["df_num"] + res["df_den"])
            anova_rows.append({
                "Signal": label,
                "F": f"{res['F']:.3f}",
                "p": f"{res['p']:.4f}",
                "η²p": round(eta2, 3),
                "Significatif": "✅" if res["p"] < 0.05 else "❌",
                "N sujets": res["n_subjects"]
            })

    df_anova = pd.DataFrame(anova_rows)

    filtre_signal = st.text_input(
        "🔎 Filtrer par nom de signal (ex: entropy, entry_, eda_tonic)",
        value="",
        key="filtre_anova_signal",
    )
    df_anova_display = (
        df_anova[df_anova["Signal"].str.contains(filtre_signal, case=False, na=False)]
        if filtre_signal else df_anova
    )
    st.dataframe(df_anova_display, width="stretch")

    # ── Synthèse : qu'est-ce qui ressort réellement en significatif ? ───────
    if "Significatif" in df_anova.columns:
        df_sig = df_anova[df_anova["Significatif"] == "✅"].copy()
        if not df_sig.empty:
            df_sig = df_sig.sort_values("η²p", ascending=False)
            st.markdown(f"**{len(df_sig)} signal(aux) significatif(s) sur {len(df_anova)} testés** (η²p décroissant) :")
            for _, row in df_sig.iterrows():
                st.write(f"- **{row['Signal']}** (p={row['p']}, η²p={row['η²p']}) — {describe_feature(row['Signal'])}")
            st.caption(
                f"⚠️ {len(df_anova)} tests effectués sans correction multiple ici — avec un "
                f"seuil non-corrigé à 0.05, environ {len(df_anova) * 0.05:.0f} faux positifs "
                "seraient attendus par hasard sur ce nombre de tests. Vérifie les signaux "
                "ci-dessus avec un test apparié dédié (comme `run_key_color_factorial`) avant "
                "de les considérer comme confirmés."
            )
        else:
            st.info("Aucun signal significatif sur l'ANOVA à mesures répétées.")


    # ── MANOVA ────────────────────────────────────────────────────────
    # st.markdown("### MANOVA — effet de la salle sur EDA + HR + HRV simultanément")
    # Construit le titre dynamiquement depuis les signaux utilisés
    signals_manova_too_many = ["eda_tonic_mean", "eda_tonic_spectral_centroid","eda_tonic_skewness","eda_tonic_median", "eda_phasic_spectral_centroid",
                      #"sdnn",
                      #"speed_std", "speed_median",
                      #"immobility_ratio", 
                      #"angular_velocity_mean", "angular_velocity_std", 
                      #"head_yaw_range", 
                      #"hr_autocorr", "hr_std", "hr_skewness", "hr_kurtosis", "hr_mean", "hr_median", "hr_rms", "hr_bpm_filtered_mean", 
                      #"hrv_rmssd_mean",  "hrv_lf_hf", "hrv_rmssd_std", 
                      #"ibi_cv", "ibi_kurtosis", "ibi_mean",
                      #"jerk_x_mean_abs", "jerk_y_mean_abs", "jerk_z_mean_abs", "immobility_ratio",
                      #"head_x_mean", "head_x_std", "head_x_max", "head_x_median", "head_x_skewness", "head_x_variance", "head_x_rms", "head_x_iqr", "head_x_peak2peak", "head_x_mad", "head_x_zcr", "head_x_wavelet_std",
                      #"head_y_mean", "head_y_max", "head_y_median", "head_y_rms", "head_y_mean_abs_diff", 
                      #"head_z_mean", "head_z_min", "head_z_max", "head_z_median", "head_z_skewness", "head_z_rms", "head_z_iqr", "head_z_peak2peak", "head_z_zcr", "head_z_wavelet_std", 
                      "pitch_mean", "pitch_std", "pitch_median", "pitch_kurtosis", "pitch_skewness", "pitch_variance", "pitch_rms", "pitch_iqr", "pitch_mad", "pitch_autocorr", "pitch_mean_abs_diff", "pitch_spectral_centroid", "pitch_wavelet_std",
                      "yaw_mean", "yaw_std", "yaw_min", "yaw_max", "yaw_median", "yaw_kurtosis", "yaw_variance", "yaw_rms", "yaw_peak2peak", "yaw_mad", "yaw_spectral_centroid", "yaw_wavelet_std",
                      "roll_std", "roll_max", "roll_skewness", "roll_rms", "roll_iqr", "roll_mad", "roll_mean_abs_diff", "roll_wavelet_std", 
                      #"jerk_x_iqr", "jerk_x_mad", "jerk_x_autocorr", "jerk_x_mean_abs_diff", "jerk_x_spectral_centroid", "jerk_x_wavelet_std",
                      #"jerk_y_iqr", "jerk_y_mad", "jerk_y_mean_abs_diff", "jerk_y_wavelet_std",
                      #"jerk_z_iqr", "jerk_z_autocorr", "jerk_z_mean_abs_diff", "jerk_z_spectral_centroid", 
                      ]
    

    # signals_manova = [
    #     "eda_tonic_mean",          # EDA niveau de base
    #     "sdnn",                
    #     "speed_std",          # HRV
    #     "hrv_rmssd_mean",    # HR
    #     "speed_mean",              # mouvement
    #     "immobility_ratio",        # statisme
    #     "angular_velocity_mean",   # exploration visuelle
    #     "head_yaw_range",          # amplitude tête
    #     "head_x_mean",
    #     "pitch_mean",
    #     "jerk_x_iqr",
   #  ]

    n_components_manova = st.slider(
        "Nombre de composantes PCA (PC1 + PC2 + ... + PCn)",
        min_value=2, max_value=15, value=7,
        help=(
            "Chaque composante (PC1, PC2, ...) est une **combinaison linéaire** des "
            "features d'origine (ex: PC1 = 0.3×eda_tonic_mean + 0.5×pitch_std - 0.1×yaw_mean + ...), "
            "choisie par la PCA pour capturer le maximum de variance possible avec le moins de "
            "dimensions. PC1 capture le plus de variance, PC2 le 2e plus, etc. — les composantes "
            "sont orthogonales (non corrélées) entre elles.\n\n"
            "⚠️ Plus tu gardes de composantes, plus la MANOVA a de variables dépendantes à tester "
            "par rapport à ton nombre d'observations — au-delà d'un certain nombre, le test devient "
            "instable (le résultat significatif/non significatif peut changer juste en ajoutant ou "
            "retirant une composante). Avec ~40 sujets, reste autour de 5-8 composantes."
        ),
        key="n_components_manova",
    )
    res_manova = run_manova_pca(df_agg_stats, signals_manova_too_many, n_components=n_components_manova)

    if "error" in res_manova:
        st.error(f"MANOVA a échoué : {res_manova['error']}")
        st.stop()

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
        st.dataframe(df_manova, width="stretch")

        with st.expander("ℹ️ Comment lire ce tableau — Intercept vs C(salle), et les 4 critères"):
            st.markdown(
                """
**Intercept vs C(salle)** : deux questions différentes posées par la même MANOVA.
- **Intercept** : "les composantes PCA sont-elles globalement différentes de zéro sur
  l'ensemble de l'expérience ?" — répond juste "oui, les gens réagissent physiologiquement/
  comportementalement, pas zéro" ; ce n'est pas la question qui t'intéresse.
- **C(salle)** : "est-ce que les composantes PCA varient significativement selon la salle ?"
  — **c'est cette ligne qui teste ton effet d'intérêt** (Q1-Q3).

**Les 4 critères** (Wilks' lambda, Pillai's trace, Hotelling-Lawley trace, Roy's greatest
root) sont 4 façons différentes de résumer la séparation entre groupes sur *toutes* les
composantes en même temps :
- **Wilks' lambda** : proche de **0** = bonne séparation, proche de **1** = aucune
  séparation. (C'est l'inverse des 3 autres : ici, plus petit = plus significatif.)
- **Pillai's trace** : proche de **1** = bonne séparation, proche de **0** = aucune.
  **Le plus robuste** quand les hypothèses (normalité multivariée, homogénéité des
  variances) ne sont pas parfaitement respectées — c'est pour ça qu'on le préfère ici.
- **Hotelling-Lawley trace** : similaire à Pillai mais plus sensible quand un seul axe
  porte toute la séparation (et donc plus fragile si cette hypothèse est fausse).
- **Roy's greatest root** : ne regarde que **le meilleur axe de séparation unique** —
  le plus optimiste des 4, donne souvent le p le plus petit, mais le moins fiable
  (il peut être "significatif" même si un seul axe sépare bien et que tous les autres
  ne séparent rien).

En pratique : si les 4 critères sont d'accord (tous ✅ ou tous ❌), tu peux avoir confiance.
S'ils se contredisent (ex: Roy's ✅ mais Pillai's ❌, comme on l'a vu avec
`n_components=10`), c'est un signe que le résultat est fragile — préfère alors **Pillai's
trace** comme arbitre.
                """
            )

        # Résumé sur Pillai uniquement (le plus recommandé)
        pillai = df_manova[ (df_manova["Critère"] == "Pillai's trace") & (df_manova["Effet"] == "C(salle)") ]
        if not pillai.empty:
            p = pillai["p-value"].values[0]
            f = pillai["F"].values[0]
            if p < 0.05:
                st.success(f"Pillai's trace : F={f}, p={p} → effet significatif ✅")
            else:
                st.warning(f"Pillai's trace : F={f}, p={p} → pas d'effet significatif ❌")
            st.caption(
                "**F** : ratio entre la variance expliquée par la salle et la variance "
                "résiduelle (intra-groupe) — plus F est grand, plus la salle explique de "
                "variance par rapport au bruit. **p** : probabilité d'observer un F au moins "
                "aussi grand si la salle n'avait *aucun* effet réel — p < 0.05 = on rejette "
                "cette hypothèse nulle."
            )

    st.divider()

    # ── XGBoost — quelles features prédisent le mieux la cible ? ────────────────
    st.markdown("### XGBoost — quelles features sont les plus importantes pour prédire la cible ?")
    st.caption(
        "Approche complémentaire à la MANOVA : un modèle d'apprentissage supervisé "
        "(arbres de décision boostés) essaie de prédire la salle/le key/le color à partir "
        "des features, puis on regarde lesquelles il a le plus utilisées. Avantage : capture "
        "des interactions et relations non-linéaires qu'une MANOVA linéaire peut rater. "
        "Limite : sur ~40-160 lignes, le modèle peut facilement surapprendre — c'est pour ça "
        "qu'on valide avec une accuracy en validation croisée **groupée par sujet** (un sujet "
        "entier est soit en train, soit en test, jamais les deux — sinon le modèle \"reconnaît\" "
        "le sujet plutôt que d'apprendre un vrai effet), à comparer à la baseline (deviner "
        "toujours la classe la plus fréquente)."
    )

    target_xgb = st.radio(
        "Cible à prédire",
        options=["salle", "key", "color"],
        format_func=lambda x: {"salle": "Salle (toutes)", "key": "Key (low/high)", "color": "Color (red/blue)"}[x],
        horizontal=True,
        key="target_xgb",
    )

    df_xgb = df_agg_stats.copy()
    df_xgb["key"] = df_xgb["salle"].map(lambda s: SALLE_KEY[int(s)])
    df_xgb["color"] = df_xgb["salle"].map(lambda s: SALLE_COLOR[int(s)])
    if target_xgb in ("key", "color") and not exclude_salle1_stats:
        df_xgb = df_xgb[df_xgb["salle"] != 1]
        st.caption("Salle 1 (baseline, sans key/color) exclue automatiquement pour cette cible.")

    res_xgb = run_xgboost_importance(df_xgb, signals_manova_too_many, target_col=target_xgb)

    if "error" in res_xgb:
        st.error(res_xgb["error"])
    else:
        col1, col2 = st.columns(2)
        col1.metric(
            f"Accuracy CV ({res_xgb['n_splits']} folds par sujet)",
            f"{res_xgb['cv_accuracy_mean']:.1%} ± {res_xgb['cv_accuracy_std']:.1%}",
        )
        col2.metric(
            "Baseline (classe majoritaire)",
            f"{res_xgb['baseline_accuracy']:.1%}",
            delta=f"{(res_xgb['cv_accuracy_mean'] - res_xgb['baseline_accuracy']):+.1%}",
        )
        if res_xgb["cv_accuracy_mean"] <= res_xgb["baseline_accuracy"] + 0.05:
            st.warning(
                "Le modèle ne fait pas mieux (ou presque) que deviner la classe majoritaire "
                "→ pas de signal prédictif robuste détecté pour cette cible."
            )
        else:
            st.success("Le modèle fait mieux que la baseline → signal prédictif détecté.")

        top_importances = res_xgb["importances"].head(15)
        fig_xgb, ax_xgb = plt.subplots(figsize=(8, 5))
        ax_xgb.barh(top_importances.index[::-1], top_importances.values[::-1], color="steelblue")
        ax_xgb.set_xlabel("Importance (gain XGBoost)")
        ax_xgb.set_title(f"Top 15 features — prédiction de '{target_xgb}'")
        st.pyplot(fig_xgb)
        plt.close(fig_xgb)

        with st.expander("Détail des features et de leur importance"):
            for feat, imp in top_importances.items():
                st.write(f"- **{feat}** ({imp:.3f}) — {describe_feature(feat)}")

        st.divider()
        st.subheader("Statistiques descriptives — top 15 features (toutes conditions confondues)")
        stats_top = df_xgb[top_importances.index].agg(["mean", "var", "std"]).T
        stats_top.columns = ["Moyenne", "Variance", "Écart-type"]
        stats_top.insert(0, "Importance XGBoost", top_importances.values)
        st.dataframe(stats_top.round(4), width="stretch")

        st.caption(f"Boxplot par groupe de '{target_xgb}' — même découpage que le test statistique ci-dessous.")
        groupes_top = sorted(df_xgb[target_xgb].dropna().unique())
        n_top = len(top_importances.index)
        n_cols_top = 3
        n_rows_top = -(-n_top // n_cols_top)  # division entière arrondie vers le haut
        fig_top_box, axes_top_box = plt.subplots(n_rows_top, n_cols_top, figsize=(4 * n_cols_top, 3.2 * n_rows_top))
        axes_top_box = np.array(axes_top_box).reshape(-1)
        for ax, feat in zip(axes_top_box, top_importances.index):
            data_par_groupe = [df_xgb.loc[df_xgb[target_xgb] == g, feat].dropna().values for g in groupes_top]
            ax.boxplot(data_par_groupe, tick_labels=[str(g) for g in groupes_top])
            ax.set_title(feat, fontsize=9)
            ax.tick_params(axis="x", labelsize=8)
        for ax in axes_top_box[n_top:]:
            ax.axis("off")
        fig_top_box.suptitle(f"Top 15 features — boxplot par '{target_xgb}'")
        fig_top_box.tight_layout()
        st.pyplot(fig_top_box)
        plt.close(fig_top_box)

        st.divider()
        st.subheader(f"Tests statistiques sur le top 15 (cible : '{target_xgb}')")
        st.caption(
            "Pas un t-test indépendant : chaque sujet contribue à plusieurs salles/conditions, "
            "ce ne sont pas des échantillons indépendants (mesures répétées). On utilise donc "
            "l'**ANOVA à mesures répétées** pour 'salle' (plus de 2 groupes), et le **test de "
            "Wilcoxon appairé** (non-paramétrique, l'équivalent du t-test appairé mais sans "
            "hypothèse de normalité — pertinent ici vu tous les signaux non-normaux détectés "
            "plus tôt) pour 'key'/'color' (2 groupes, même sujet dans les deux)."
        )

        rows_top_test = []
        for feat in top_importances.index:
            if target_xgb == "salle":
                res_test = run_anova_repeated(df_xgb, feat)
                if "error" in res_test:
                    rows_top_test.append({"Feature": feat, "Test": "ANOVA RM", "Note": res_test["error"]})
                else:
                    rows_top_test.append({
                        "Feature": feat, "Test": "ANOVA RM",
                        "F": round(res_test["F"], 3), "p": round(res_test["p"], 4),
                        "Significatif": "✅" if res_test["p"] < 0.05 else "❌",
                        "N sujets": res_test["n_subjects"],
                    })
            else:
                res_factorial_top = run_key_color_factorial(df_xgb, feat, SALLE_KEY, SALLE_COLOR)
                r = res_factorial_top.get(target_xgb, {"error": "non calculé"})
                if "error" in r:
                    rows_top_test.append({"Feature": feat, "Test": "Wilcoxon appairé", "Note": r["error"]})
                else:
                    lvl_lo, lvl_hi = r["levels"]
                    rows_top_test.append({
                        "Feature": feat, "Test": "Wilcoxon appairé",
                        f"Moyenne {lvl_lo}": round(r[f"mean_{lvl_lo}"], 4),
                        f"Moyenne {lvl_hi}": round(r[f"mean_{lvl_hi}"], 4),
                        "p": round(r["p"], 4),
                        "Significatif": "✅" if r["p"] < 0.05 else "❌",
                        "N sujets": r["n_subjects"],
                    })

        df_top_test = pd.DataFrame(rows_top_test)
        if "p" in df_top_test.columns:
            n_tests_top = int(df_top_test["p"].notna().sum())
            seuil_top = 0.05 / n_tests_top if n_tests_top else 0.05
            df_top_test["Significatif (Bonferroni)"] = df_top_test["p"].apply(
                lambda p: "✅" if pd.notna(p) and p < seuil_top else ("❌" if pd.notna(p) else "")
            )
            st.caption(f"Correction Bonferroni sur {n_tests_top} tests → seuil ajusté = {seuil_top:.5f}")
        st.dataframe(df_top_test, width="stretch")

        st.divider()
        st.subheader("Comparer deux salles précises")
        st.caption(
            "Même sujets dans les deux salles → test apparié (Wilcoxon signed-rank), "
            "pas un t-test indépendant."
        )
        salles_dispo = sorted(df_xgb["salle"].dropna().unique())
        col1, col2 = st.columns(2)
        with col1:
            salle_a_sel = st.selectbox("Salle A", salles_dispo, index=0, key="salle_a_pair")
        with col2:
            salle_b_sel = st.selectbox(
                "Salle B", salles_dispo,
                index=min(1, len(salles_dispo) - 1), key="salle_b_pair"
            )

        rows_pair = []
        for feat in top_importances.index:
            res_pair = compare_two_salles(df_xgb, feat, salle_a_sel, salle_b_sel)
            if "error" in res_pair:
                rows_pair.append({"Feature": feat, "Note": res_pair["error"]})
            else:
                rows_pair.append({
                    "Feature": feat,
                    f"Moyenne salle {salle_a_sel}": round(res_pair["mean_a"], 4),
                    f"Moyenne salle {salle_b_sel}": round(res_pair["mean_b"], 4),
                    "p (paired t-test)": round(res_pair["t_p"], 4),
                    "Sig. (t-test)": "✅" if res_pair["t_p"] < 0.05 else "❌",
                    "p (Wilcoxon)": round(res_pair["w_p"], 4),
                    "Sig. (Wilcoxon)": "✅" if res_pair["w_p"] < 0.05 else "❌",
                    "N sujets": res_pair["n_subjects"],
                })
        st.dataframe(pd.DataFrame(rows_pair), width="stretch")

        st.divider()
        st.subheader("Comparaison agrégée — key (low/high) et color (red/blue)")
        st.caption(
            "Plutôt que deux salles précises : moyenne sur les 2 salles low vs les 2 salles "
            "high (idem pour color), même sujets dans les deux groupes → test apparié. "
            "Salle 1 exclue (pas de key/color)."
        )

        rows_agg_kc = []
        for feat in top_importances.index:
            res_kc = run_key_color_factorial(df_xgb, feat, SALLE_KEY, SALLE_COLOR)
            for factor_name, r in res_kc.items():
                if "error" in r:
                    rows_agg_kc.append({"Feature": feat, "Facteur": factor_name, "Note": r["error"]})
                else:
                    lvl_lo, lvl_hi = r["levels"]
                    rows_agg_kc.append({
                        "Feature": feat,
                        "Facteur": factor_name,
                        f"Moyenne {lvl_lo}": round(r[f"mean_{lvl_lo}"], 4),
                        f"Moyenne {lvl_hi}": round(r[f"mean_{lvl_hi}"], 4),
                        "p (paired t-test)": round(r["t_p"], 4),
                        "Sig. (t-test)": "✅" if r["t_p"] < 0.05 else "❌",
                        "p (Wilcoxon)": round(r["p"], 4),
                        "Sig. (Wilcoxon)": "✅" if r["p"] < 0.05 else "❌",
                        "N sujets": r["n_subjects"],
                    })

        df_agg_kc = pd.DataFrame(rows_agg_kc)
        if "p (Wilcoxon)" in df_agg_kc.columns:
            n_tests_kc = int(df_agg_kc["p (Wilcoxon)"].notna().sum())
            seuil_kc = 0.05 / n_tests_kc if n_tests_kc else 0.05
            df_agg_kc["Sig. Wilcoxon (Bonferroni)"] = df_agg_kc["p (Wilcoxon)"].apply(
                lambda p: "✅" if pd.notna(p) and p < seuil_kc else ("❌" if pd.notna(p) else "")
            )
            st.caption(f"Correction Bonferroni sur {n_tests_kc} tests → seuil ajusté = {seuil_kc:.5f}")
        st.dataframe(df_agg_kc, width="stretch")

    # ── MANOVA VERIFICATION ────────────────────────────────────────────────────────

    features = signals_manova_too_many

    X = df_agg_stats[features].dropna()
    y = df_agg_stats.loc[X.index, 'salle']

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
    resultats_anova = []

    for signal in features:
        try:
            aov = pg.rm_anova(data=df_agg_stats, dv=signal, within='salle', subject='subject')
            if 'ng2' in aov.columns:
                eta2 = aov['ng2'].values[0]
                resultats_anova.append({"signal": signal, "eta2": eta2})
            else:
                resultats_anova.append({"signal": signal, "eta2": None})
        except Exception as e:
            resultats_anova.append({"signal": signal, "eta2": None})

    resultats_anova.sort(key=lambda x: x["eta2"] if x["eta2"] is not None else -1, reverse=True)

    for r in resultats_anova:
        eta2_str = f"{r['eta2']:.3f}" if r["eta2"] is not None else "N/A"
        with st.expander(f"{r['signal']} → η²p = {eta2_str}"):
            st.caption(describe_feature(r["signal"]))


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
    #df_eta = compute_effect_sizes(df_agg_stats, signals=signals_for_eta)
    #st.dataframe(df_eta)
    
    # ── SCR ────────────────────────────────────────────────────────
    #SCR → "La salle 3 déclenche 2× plus de réponses phasiques que la salle 1 — elle est plus stimulante au niveau sympathique"

    st.subheader("SCR — Réactivité phasique par salle")
    # df_agg_stats contient déjà scr_count/scr_amp_mean/scr_auc si aggregate_subjects a été mis à jour
    scr_par_salle = df_agg_stats.groupby('salle')[['scr_rate', 'scr_amplitude_mean', 'scr_auc']].mean().round(3)
    st.dataframe(scr_par_salle)
    
    # --- LDA ---
    #LDA → "LD1 explique 70% de la séparation, porté surtout par EDA tonic — les salles se séparent principalement sur un axe d'activation tonique"

    st.subheader("LDA — Structure discriminante entre salles")
    
    features_lda = [
        'eda_tonic_mean',
        'hrv_rmssd_mean',
        'sdnn',
        'scr_rate',
        'scr_auc',
        #'pnn50'
    ]
    lda_results = run_lda_profile(df_agg_stats, features=features_lda)
    
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
        if col in df_agg_stats.columns and i < len(all_cols):
            with all_cols[i]:
                fig = make_qqplot(df_agg_stats, col, label)
                st.plotly_chart(fig, width="stretch")

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
    #st.dataframe(df_filtered, width="stretch")

    # Compte rapide des non-normaux
    #n_non_normal = (df_filtered["Normal ?"] == "❌").sum()
    #st.caption(f"{n_non_normal}/{len(df_filtered)} sujets non-normaux pour ce signal")
    
    #'''

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Friedmann - non param
# ══════════════════════════════════════════════════════════════════════════════
with tab_nonparam:
    # ── Friedman ──────────────────────────────────────────────────────
    st.markdown("### Friedman — effet de la salle (non-paramétrique)")
    st.caption("Alternative à l'ANOVA RM quand la normalité n'est pas respectée")

    exclude_salle1_nonparam = st.checkbox(
        "Exclure la salle 1 (baseline diffuse)",
        value=False,
        help=(
            "La salle 1 n'a pas de key/color grading et les participants y restent "
            "souvent plus longtemps (découverte de l'environnement) — elle peut écraser "
            "le signal lié au vrai design 2×2 (key×color, salles 2-5)."
        ),
        key="exclude_salle1_nonparam",
    )

    df_agg_nonparam = df_agg
    if exclude_salle1_nonparam:
        df_agg_nonparam = df_agg_nonparam[df_agg_nonparam["salle"] != 1].reset_index(drop=True)
        st.caption(f"Salle 1 exclue — {len(df_agg_nonparam)} lignes restantes (salles 2-5)")

    COLONNES_EXCLURE = {"participant", "salle", "fichier", "fenetre_debut"}

    colonnes_features = [
        c for c in df_agg_nonparam.columns
        if c not in COLONNES_EXCLURE
        and pd.api.types.is_numeric_dtype(df_agg_nonparam[c])  # ← filtre par type
    ]

    signals_nonparam = {col: col for col in colonnes_features}

    df_friedman = run_friedman(df_agg_nonparam, signals_nonparam)
    st.dataframe(df_friedman, width="stretch")

    st.markdown("### Post-hoc Dunn — quelles salles diffèrent entre elles ?")
    st.caption("Correction Bonferroni — seulement pertinent pour les signaux significatifs en Friedman")

    dunn_results = run_posthoc_dunn(df_agg_nonparam, signals_nonparam)

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
            p_matrix_display.style.map(color_pvalue),
            width="stretch"
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

    # ── Design factoriel Q3 : key (low/high) × color (red/blue) ─────────────
    st.divider()
    st.markdown("### Test direct du design Q3 — key (low/high) × color grading (red/blue)")
    st.caption(
        "Plutôt que comparer les 5 salles entre elles, on teste directement le facteur "
        "lumineux : pour chaque sujet, moyenne sur les salles low vs high"
        ", puis moyenne sur les salles red vs blue. La salle 1 (baseline, sans "
        "key/color) est exclue de ce test — elle n'a pas sa place dans un design 2×2."
    )

    st.caption(
        "⚠️ `jerk_x_mean` / `jerk_y_mean` / `jerk_z_mean` sont des moyennes **signées** "
        "(dérive directionnelle nette, valeurs proches de 0) — pas l'intensité de la "
        "brusquerie. Utiliser `jerk_x_mean_abs` / `jerk_y_mean_abs` / `jerk_z_mean_abs` "
        "pour l'intensité (toujours positive)."
    )

    top_features_jerk_speed = [
        "jerk_y_wavelet_std", "speed_std",
        "jerk_x_mean_abs", "jerk_y_mean_abs", "jerk_z_mean_abs",
        "jerk_y_mean", "jerk_z_mean", "jerk_x_mean",
        "jerk_y_mad", "jerk_y_mean_abs_diff",
        # Candidats restants après nettoyage MANOVA (salle 1 exclue, jerk corrigé) :
        # EDA tonique (arousal de fond) et variabilité de l'inclinaison de tête (pitch).
        "eda_tonic_spectral_centroid",
        "pitch_wavelet_std", "pitch_variance", "pitch_std", "pitch_rms",
        "pitch_mad", "pitch_iqr", "pitch_autocorr", "pitch_mean_abs_diff", "pitch_median",
        "head_z_min", "head_z_iqr", "head_z_wavelet_std",
        "yaw_mean", "yaw_min", "yaw_spectral_centroid", "yaw_kurtosis",
        "hrv_lf_hf",
        # Entropie d'exploration du regard + réaction immédiate à l'entrée en salle
        "pitch_entropy", "yaw_entropy",
        "entry_speed_peak", "entry_speed_mean", "entry_jerk_peak", "entry_angular_velocity_peak",
    ]
    top_features_jerk_speed = [f for f in top_features_jerk_speed if f in df_agg.columns]

    rows_factorial = []
    for feat in top_features_jerk_speed:
        res = run_key_color_factorial(df_agg, feat, SALLE_KEY, SALLE_COLOR)
        for factor_name, r in res.items():
            if "error" in r:
                rows_factorial.append({"Feature": feat, "Facteur": factor_name, "Note": r["error"]})
            else:
                lvl_lo, lvl_hi = r["levels"]
                rows_factorial.append({
                    "Feature":            feat,
                    "Facteur":            factor_name,
                    f"Moyenne {lvl_lo}":  round(r[f"mean_{lvl_lo}"], 4),
                    f"Moyenne {lvl_hi}":  round(r[f"mean_{lvl_hi}"], 4),
                    "p (Wilcoxon)":       round(r["p"], 4),
                    "p brut":             r["p"],
                    "Significatif (p<0.05, non corrigé)": "✅" if r["p"] < 0.05 else "❌",
                    "N sujets":           r["n_subjects"],
                })

    df_factorial = pd.DataFrame(rows_factorial)

    # Correction Bonferroni : seuil ajusté = 0.05 / nombre de tests valides effectués
    n_tests = df_factorial["p brut"].notna().sum() if "p brut" in df_factorial.columns else 0
    if n_tests > 0:
        seuil_bonferroni = 0.05 / n_tests
        df_factorial["Significatif (Bonferroni)"] = df_factorial["p brut"].apply(
            lambda p: "✅" if pd.notna(p) and p < seuil_bonferroni else ("❌" if pd.notna(p) else "")
        )
        df_factorial = df_factorial.drop(columns=["p brut"])
        st.caption(f"Correction Bonferroni sur {n_tests} tests → seuil ajusté = {seuil_bonferroni:.5f}")

    st.dataframe(df_factorial, width="stretch")
    st.caption(
        "describe_feature() (onglet glossaire) pour le détail de chaque feature — "
        "rappel : ce sont des features de mouvement de tête (jerk = brusque, "
        "speed = vitesse de déplacement), pas des mesures émotionnelles directes."
    )


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

    st.subheader("Statistiques descriptives — Valence & Arousal par salle")
    stats_sam_salle = df_sam.groupby("salle")[["valence", "arousal"]].agg(["mean", "var", "std"])
    stats_sam_salle.columns = [f"{var}_{stat}" for var, stat in stats_sam_salle.columns]
    st.dataframe(stats_sam_salle.round(3), width="stretch")

    col_lang_sam, col_mode_sam = st.columns(2)
    with col_lang_sam:
        lang_sam_stats = st.radio("Langue du graphique", ["FR", "EN"], horizontal=True, key="lang_sam_stats")
    with col_mode_sam:
        mode_sam_stats = st.radio("Type de graphique", ["Barres (moyenne ± SD)", "Boxplot"], horizontal=True, key="mode_sam_stats")
    labels_sam_stats = {
        "FR": {"valence": "Valence", "arousal": "Arousal", "x": "Salle", "y": "Score SAM"},
        "EN": {"valence": "Valence", "arousal": "Arousal", "x": "Room", "y": "SAM score"},
    }[lang_sam_stats]

    fig_sam_stats = go.Figure()
    if mode_sam_stats == "Boxplot":
        col_color_valence, col_color_arousal = st.columns(2)
        with col_color_valence:
            color_box_valence = st.color_picker("Couleur boxplot — Valence", "#FFD700", key="color_box_valence")
        with col_color_arousal:
            color_box_arousal = st.color_picker("Couleur boxplot — Arousal", "#1E90FF", key="color_box_arousal")
        colors_box_sam = {"valence": color_box_valence, "arousal": color_box_arousal}

        # Plotly ne permet pas de colorer la médiane différemment du contour du Box.
        # On positionne donc les boîtes nous-mêmes (x numérique) pour pouvoir superposer
        # un trait noir exactement sur la médiane de chacune (au lieu de se fier à
        # l'alignement automatique offsetgroup Box/Scatter, qui ne matche pas).
        rooms_sam = sorted(df_sam["salle"].dropna().unique())
        room_idx_sam = {r: i for i, r in enumerate(rooms_sam)}
        variables_sam = ["valence", "arousal"]
        n_var_sam = len(variables_sam)
        group_width_sam = 0.7
        box_width_sam = group_width_sam / n_var_sam * 0.9

        for j_var, variable in enumerate(variables_sam):
            offset_sam = (j_var - (n_var_sam - 1) / 2) * (group_width_sam / n_var_sam)
            x_pos_sam = df_sam["salle"].map(room_idx_sam) + offset_sam
            fig_sam_stats.add_trace(go.Box(
                x=x_pos_sam, y=df_sam[variable],
                width=box_width_sam,
                marker_color=colors_box_sam[variable],
                line_color=colors_box_sam[variable],
                fillcolor="white",
                name=labels_sam_stats[variable],
                boxmean=False,
            ))
            medians_var = df_sam.groupby("salle")[variable].median()
            for room in rooms_sam:
                cx = room_idx_sam[room] + offset_sam
                fig_sam_stats.add_trace(go.Scatter(
                    x=[cx - box_width_sam / 2, cx + box_width_sam / 2],
                    y=[medians_var[room], medians_var[room]],
                    mode="lines",
                    line=dict(color="black", width=3),
                    showlegend=False,
                    hoverinfo="skip",
                ))

        fig_sam_stats.update_xaxes(
            tickvals=list(room_idx_sam.values()),
            ticktext=[str(r) for r in rooms_sam],
        )
        boxmode_sam = dict()
    else:
        for variable, color in [("valence", "#FF6B6B"), ("arousal", "#7ec8e3")]:
            means = df_sam.groupby("salle")[variable].mean()
            stds = df_sam.groupby("salle")[variable].std()
            fig_sam_stats.add_trace(go.Bar(
                x=means.index.astype(str), y=means.values,
                error_y=dict(type="data", array=stds.values),
                marker_color=color,
                name=labels_sam_stats[variable],
            ))
        boxmode_sam = dict(barmode="group")
    fig_sam_stats.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=350,
        xaxis_title=labels_sam_stats["x"],
        yaxis_title=labels_sam_stats["y"],
        **boxmode_sam,
        font=dict(color="black"),
        legend=dict(font=dict(color="black", size=20)),
        xaxis=dict(title_font=dict(color="black", size=20), tickfont=dict(color="black", size=16)),
        yaxis=dict(title_font=dict(color="black", size=20), tickfont=dict(color="black", size=16)),
    )
    st.plotly_chart(fig_sam_stats, width="stretch")

    for variable in ['valence', 'arousal']:
        st.subheader(variable.capitalize())
        heatmap_participant_salle(
            df_sam,
            variable=variable,
            subject_col='subject',
            titre=f"Heatmap {variable} — Participant × Salle",
        )

    st.divider()
    st.markdown("### Est-ce que SAM lui-même varie selon la condition ?")
    st.caption(
        "Équivalent de la section 'Effectiveness of the virtual park' de Felnhofer et al. "
        "(2015) — avant de regarder si la physio suit le ressenti, on vérifie d'abord que "
        "le ressenti rapporté (SAM) varie significativement selon la salle/key/color. "
        "Adapté en mesures répétées (ANOVA RM + Wilcoxon appairé) puisque ce sont les mêmes "
        "participants dans toutes les salles, contrairement au design entre-groupes de l'article."
    )

    rows_sam_anova = []
    for variable in ["valence", "arousal"]:
        res_sam_anova = run_anova_repeated(df_sam, variable)
        if "error" in res_sam_anova:
            rows_sam_anova.append({"Variable": variable, "Note": res_sam_anova["error"]})
        else:
            rows_sam_anova.append({
                "Variable": variable,
                "F": round(res_sam_anova["F"], 3),
                "p": round(res_sam_anova["p"], 4),
                "Significatif": "✅" if res_sam_anova["p"] < 0.05 else "❌",
                "N sujets": res_sam_anova["n_subjects"],
            })
    st.markdown("**ANOVA à mesures répétées — effet de la salle sur SAM**")
    st.dataframe(pd.DataFrame(rows_sam_anova), width="stretch")

    rows_sam_factorial = []
    for variable in ["valence", "arousal"]:
        res_sam_factorial = run_key_color_factorial(df_sam, variable, SALLE_KEY, SALLE_COLOR)
        for factor_name, r in res_sam_factorial.items():
            if "error" in r:
                rows_sam_factorial.append({"Variable": variable, "Facteur": factor_name, "Note": r["error"]})
            else:
                lvl_lo, lvl_hi = r["levels"]
                rows_sam_factorial.append({
                    "Variable": variable,
                    "Facteur": factor_name,
                    f"Moyenne {lvl_lo}": round(r[f"mean_{lvl_lo}"], 3),
                    f"Moyenne {lvl_hi}": round(r[f"mean_{lvl_hi}"], 3),
                    "p (paired t-test)": round(r["t_p"], 4),
                    "Sig. (t-test)": "✅" if r["t_p"] < 0.05 else "❌",
                    "p (Wilcoxon)": round(r["p"], 4),
                    "Sig. (Wilcoxon)": "✅" if r["p"] < 0.05 else "❌",
                    "N sujets": r["n_subjects"],
                })
    st.markdown("**Test factoriel key (low/high) × color (red/blue) sur SAM**")
    st.dataframe(pd.DataFrame(rows_sam_factorial), width="stretch")

    st.divider()
    st.markdown("### Corrélation signaux physio/mouvement ↔ réponse SAM")
    st.caption(
        "Calculée séparément par salle (à travers les participants) — pas en mélangeant "
        "les salles, pour ne pas confondre un effet de salle avec un vrai lien signal↔ressenti "
        "au sein d'une même condition. Spearman par défaut (rangs, robuste à la non-normalité)."
    )

    signals_sam_corr_all = {
        # Physiologie
        "EDA tonique (moyenne)":              "eda_tonic_mean",
        "EDA tonique (spectral centroid)":    "eda_tonic_spectral_centroid",
        "EDA phasique (moyenne)":             "eda_phasic_mean",
        "SCR rate":                           "scr_rate",
        "HR (moyenne)":                       "hr_mean",
        "HRV RMSSD (moyenne)":                "hrv_rmssd_mean",
        "HRV LF/HF":                          "hrv_lf_hf",
        # Jerk
        "Jerk X (intensité)":                 "jerk_x_mean_abs",
        "Jerk Y (intensité)":                 "jerk_y_mean_abs",
        "Jerk Z (intensité)":                 "jerk_z_mean_abs",
        # Vitesse / mobilité
        "Vitesse (std)":                      "speed_std",
        "Immobilité (ratio)":                 "immobility_ratio",
        "Pic vitesse à l'entrée":             "entry_speed_peak",
        # Pitch (haut/bas)
        "Pitch – moyenne":                    "pitch_mean",
        "Pitch – std":                        "pitch_std",
        "Pitch – variance":                   "pitch_variance",
        "Pitch – RMS":                        "pitch_rms",
        "Pitch – IQR":                        "pitch_iqr",
        "Pitch – peak-to-peak":               "pitch_peak2peak",
        "Pitch – kurtosis":                   "pitch_kurtosis",
        "Pitch – skewness":                   "pitch_skewness",
        "Pitch – ZCR":                        "pitch_zcr",
        "Pitch – autocorr":                   "pitch_autocorr",
        "Pitch – mean abs diff":              "pitch_mean_abs_diff",
        "Pitch – FFT mean":                   "pitch_fft_mean",
        "Pitch – FFT max":                    "pitch_fft_max",
        "Pitch – spectral centroid":          "pitch_spectral_centroid",
        "Pitch – wavelet energy":             "pitch_wavelet_energy",
        "Pitch – wavelet std":                "pitch_wavelet_std",
        "Pitch – entropie exploration":       "pitch_entropy",
        "Pitch – range total":                "head_pitch_range",
        # Yaw (gauche/droite)
        "Yaw – moyenne":                      "yaw_mean",
        "Yaw – std":                          "yaw_std",
        "Yaw – variance":                     "yaw_variance",
        "Yaw – RMS":                          "yaw_rms",
        "Yaw – IQR":                          "yaw_iqr",
        "Yaw – peak-to-peak":                 "yaw_peak2peak",
        "Yaw – kurtosis":                     "yaw_kurtosis",
        "Yaw – skewness":                     "yaw_skewness",
        "Yaw – ZCR":                          "yaw_zcr",
        "Yaw – autocorr":                     "yaw_autocorr",
        "Yaw – mean abs diff":                "yaw_mean_abs_diff",
        "Yaw – FFT mean":                     "yaw_fft_mean",
        "Yaw – FFT max":                      "yaw_fft_max",
        "Yaw – spectral centroid":            "yaw_spectral_centroid",
        "Yaw – wavelet energy":               "yaw_wavelet_energy",
        "Yaw – wavelet std":                  "yaw_wavelet_std",
        "Yaw – entropie exploration":         "yaw_entropy",
        "Yaw – range total":                  "head_yaw_range",
        # Roll
        "Roll – moyenne":                     "roll_mean",
        "Roll – std":                         "roll_std",
        "Roll – variance":                    "roll_variance",
        "Roll – RMS":                         "roll_rms",
        "Roll – IQR":                         "roll_iqr",
        "Roll – peak-to-peak":                "roll_peak2peak",
        "Roll – kurtosis":                    "roll_kurtosis",
        "Roll – skewness":                    "roll_skewness",
        "Roll – ZCR":                         "roll_zcr",
        "Roll – autocorr":                    "roll_autocorr",
        "Roll – mean abs diff":               "roll_mean_abs_diff",
        "Roll – FFT mean":                    "roll_fft_mean",
        "Roll – FFT max":                     "roll_fft_max",
        "Roll – spectral centroid":           "roll_spectral_centroid",
        "Roll – wavelet energy":              "roll_wavelet_energy",
        "Roll – wavelet std":                 "roll_wavelet_std",
    }

    col1, col2 = st.columns([3, 1])
    with col1:
        signals_sam_selected = st.multiselect(
            "Signaux à tester",
            options=list(signals_sam_corr_all.keys()),
            default=list(signals_sam_corr_all.keys()),
            key="signals_sam_corr",
        )
    with col2:
        method_corr = st.radio("Méthode", ["spearman", "pearson"], key="method_sam_corr")

    signals_sam_corr = {k: signals_sam_corr_all[k] for k in signals_sam_selected}

    if signals_sam_corr:
        df_corr_sam = compute_signal_sam_correlation(df_agg, df_sam, signals_sam_corr, method=method_corr)

        n_tests_corr = len(df_corr_sam)
        seuil_bonf_corr = 0.05 / n_tests_corr if n_tests_corr else 0.05
        df_corr_sam["Significatif (Bonferroni)"] = df_corr_sam["p-value"].apply(
            lambda p: "✅" if p < seuil_bonf_corr else "❌"
        )

        only_sig = st.checkbox("Afficher seulement les corrélations significatives (non corrigé)", value=False, key="only_sig_corr")
        df_corr_display = df_corr_sam[df_corr_sam["Significatif"] == "✅"] if only_sig else df_corr_sam
        st.dataframe(df_corr_display.sort_values("p-value"), width="stretch")
        st.caption(
            f"{n_tests_corr} tests effectués → seuil Bonferroni ajusté = {seuil_bonf_corr:.5f}. "
            "Avec ce volume de tests, attends-toi à quelques faux positifs à p<0.05 non corrigé."
        )

        # ── Visualisation des corrélations significatives ────────────────────────
        df_corr_sig = df_corr_sam[df_corr_sam["Significatif"] == "✅"].sort_values("p-value")
        if not df_corr_sig.empty:
            st.divider()
            st.subheader("Visualisation des corrélations significatives")
            st.caption(
                "Scatter plot (relation continue, fidèle au calcul de r/p) et boxplot "
                "(signal réparti selon que le score SAM est sous/au-dessus de sa médiane "
                "dans cette salle — lecture plus simple mais qui discrétise le SAM)."
            )
            label_to_col_sam = {label: col for label, col in signals_sam_corr.items()}
            for _, row_corr in df_corr_sig.iterrows():
                salle_corr = row_corr["Salle"]
                signal_label = row_corr["Signal"]
                sam_var = row_corr["SAM"]
                signal_col = label_to_col_sam.get(signal_label)
                if signal_col is None or signal_col not in df_agg.columns:
                    continue

                df_plot_corr = df_agg[df_agg["salle"] == salle_corr].merge(
                    df_sam[df_sam["salle"] == salle_corr], on=["subject", "salle"], how="inner"
                )[["subject", signal_col, sam_var]].dropna()
                if len(df_plot_corr) < 5:
                    continue

                st.markdown(
                    f"**{signal_label} × {sam_var.capitalize()} — Salle {salle_corr}** "
                    f"(r={row_corr['r']}, p={row_corr['p-value']})"
                )
                col_scatter, col_box = st.columns(2)
                with col_scatter:
                    fig_scatter_corr = px.scatter(
                        df_plot_corr, x=signal_col, y=sam_var, hover_data=["subject"],
                        trendline="ols",
                        labels={signal_col: signal_label, sam_var: sam_var.capitalize()},
                    )
                    fig_scatter_corr.update_layout(height=320, margin=dict(t=30, b=20))
                    st.plotly_chart(fig_scatter_corr, width="stretch")
                with col_box:
                    mediane_sam = df_plot_corr[sam_var].median()
                    df_plot_corr["Groupe SAM"] = df_plot_corr[sam_var].apply(
                        lambda v: f"{sam_var.capitalize()} bas (≤ médiane)" if v <= mediane_sam
                        else f"{sam_var.capitalize()} haut (> médiane)"
                    )
                    fig_box_corr = px.box(
                        df_plot_corr, x="Groupe SAM", y=signal_col, points="all",
                        labels={signal_col: signal_label},
                    )
                    fig_box_corr.update_layout(height=320, margin=dict(t=30, b=20))
                    st.plotly_chart(fig_box_corr, width="stretch")
        elif only_sig:
            st.info("Aucune corrélation significative à afficher.")
    else:
        st.info("Sélectionne au moins un signal à tester.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 - BOXPLOT
# ══════════════════════════════════════════════════════════════════════════════
with tab_box_plot:
    features_boxplot = ["eda_tonic_mean", "hrv_rmssd_mean", "speed_std", "speed_median"]

    for feat in features_boxplot:
        plot_feature_boxplot(df_agg, feat)       


# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 - Evaluation de features & LDA
# ══════════════════════════════════════════════════════════════════════════════
with tab_evaluation:
    all_interesting_features = [
        "scr_auc",
        "eda_tonic_mean", 
        "hrv_rmssd_mean",
        "eda_uS_filtered_mean",

        # TOP 10 FRIEDMANN X^2 
        "head_y_zcr",
        "head_z_spectral_centroid",
        "head_x_spectral_centroid",
        "jerk_y_fft_max",
        "head_y_spectral_centroid",
        "jerk_y_fft_mean",
        "head_y_min",
        "jerk_y_min",
        "jerk_y_std",
        "jerk_y_variance",

        # TOP 10 MANOVA (qui ne sont pas déjà ici)
        "jerk_y_wavelet_std",
        "speed_std",
        "jerk_y_mean",
        "jerk_x_mean",
        "jerk_y_mad",
        "eda_tonic_spectral_centroid",
        "jerk_y_mean_abs_diff",
        "speed_median",
        "pitch_wavelet_std",

        #"scr_count", "scr_auc", 
        #"eda_driver_mean", "eda_phasic_mean", "sdnn","pnn50", "speed_mean", "speed_std","speed_median", "immobility_ratio", "path_length",
        #"angular_velocity_mean", "angular_velocity_std", "head_yaw_range", "head_pitch_range", "hr_bpm_filtered_mean", "hr_bpm_filtered_zscore_mean",
    ]

    # ── Force brute : toutes les combinaisons ────────────────────────────────
    #results = []
    #for r in range(1, len(all_interesting_features) + 1):
    #    for combo in combinations(all_interesting_features, r):
    #        score = evaluate_feature_combo(df_agg, list(combo))
    #        results.append({"features": combo, "accuracy": score})

    #df_results = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    #st.dataframe(df_results.head(20))  # st.dataframe plutôt que print

    # ── Préparer X et y pour le SFS ─────────────────────────────────────────
    # On prend les lignes qui n'ont pas de NaN sur les features EDA
    df_clean = df_agg[all_interesting_features + ["salle"]].dropna()

    X = df_clean[all_interesting_features].values          # numpy array (n_samples, n_features)
    y = df_clean["salle"].values                   # labels : noms des salles

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)             # centré-réduit feature par feature

    # ── SequentialFeatureSelector ────────────────────────────────────────────
    lda = LinearDiscriminantAnalysis()
    sfs = SequentialFeatureSelector(lda, n_features_to_select="auto", cv=LeaveOneOut())
    sfs.fit(X_scaled, y)

    # sfs.get_support() retourne un masque booléen [True, False, True, ...]
    # on l'utilise pour filtrer la liste des noms de features
    selected = [f for f, keep in zip(all_interesting_features, sfs.get_support()) if keep]
    st.write("Features sélectionnées par SFS :", selected)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 10 - AUDIO : sentiment des retours verbaux, par salle
# ══════════════════════════════════════════════════════════════════════════════
with tab_audio:
    st.header("Retours audio — sentiment par salle")
    st.caption(
        "Calculé à part (Whisper medium + BERT, trop lourd pour tourner en live dans "
        "Streamlit) via analyse_audio_par_salle.py puis analyse_audio_sentiment_par_salle.py. "
        "Chaque segment de parole est attribué à une salle soit par mention explicite "
        "(numéro, ordinal 'le deuxième', 'dernier', couleur), soit par fallback temporel "
        "(dit alors que le participant est encore dans la salle), soit classé "
        "'global' (retour rétrospectif de fin de session, non rattachable à une salle "
        "précise). Les segments citant plusieurs salles à la fois sont exclus de "
        "l'agrégation automatique (ambigus — ex: 'pas le premier mais le deuxième et le "
        "troisième c'était horrible') et sauvegardés à part pour relecture manuelle."
    )

    audio_csv_path = Path("../../audio/analyse_audio_sentiment_par_salle.csv")
    if not audio_csv_path.exists():
        st.warning(
            f"Fichier introuvable : `{audio_csv_path}`. Lance d'abord "
            "`analyse_audio_par_salle.py` puis `analyse_audio_sentiment_par_salle.py`."
        )
        st.stop()

    df_audio = load_audio_sentiment(str(audio_csv_path))
    n_indetermine = (df_audio["sentiment"].str.startswith("Indéterminé")).sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Buckets participant × salle", len(df_audio))
    col2.metric("Salles 'global' (fin de session)", (df_audio["salle"] == "global").sum())
    col3.metric("Indéterminés (texte trop court/erreur)", int(n_indetermine))

    st.divider()
    st.subheader("Détail des retours")
    salle_filter = st.multiselect(
        "Filtrer par salle",
        options=sorted(df_audio["salle"].astype(str).unique()),
        default=sorted(df_audio["salle"].astype(str).unique()),
        key="audio_salle_filter",
    )
    df_audio_display = df_audio[df_audio["salle"].astype(str).isin(salle_filter)]
    st.dataframe(
        df_audio_display[["participant", "salle", "n_segments", "sentiment", "sentiment_score", "mots_cles", "texte"]],
        width="stretch",
    )

    st.divider()
    st.subheader("Distribution du sentiment par salle")
    df_sentiment_count = (
        df_audio.groupby(["salle", "sentiment"]).size().reset_index(name="n")
    )
    fig_sentiment_salle = px.bar(
        df_sentiment_count, x="salle", y="n", color="sentiment", barmode="group",
        template="plotly_dark",
        color_discrete_map={"Positif": "#7ec8e3", "Neutre": "#b0b8c1", "Négatif": "#FF6B6B"},
    )
    fig_sentiment_salle.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22", height=350)
    st.plotly_chart(fig_sentiment_salle, width="stretch")

    st.divider()
    st.subheader("Corrélation sentiment audio ↔ SAM (par salle)")
    st.caption(
        "Comme pour les signaux physio/mouvement : calculée séparément par salle, "
        "pas en mélangeant — pour ne pas confondre un effet de salle avec un vrai lien "
        "sentiment↔ressenti au sein d'une même condition. Le bucket 'global' est exclu "
        "(pas de SAM associé à un retour de fin de session non rattaché à une salle)."
    )

    df_sam_audio = extract_sam_scores(subjects_filtered)
    df_corr_audio = compute_audio_sentiment_sam_correlation(df_audio, df_sam_audio, method="spearman")

    if df_corr_audio.empty:
        st.info("Pas assez de données pour calculer une corrélation (N≥5 requis par salle).")
    else:
        st.dataframe(df_corr_audio.sort_values("p-value"), width="stretch")
        n_tests_audio = len(df_corr_audio)
        seuil_audio = 0.05 / n_tests_audio if n_tests_audio else 0.05
        st.caption(f"{n_tests_audio} test(s) effectué(s) → seuil Bonferroni ajusté = {seuil_audio:.5f}.")

    if n_indetermine > 0:
        st.caption(
            f"⚠️ {n_indetermine} bucket(s) 'Indéterminé' exclu(s) automatiquement de la "
            "corrélation (sentiment_signe = NaN)."
        )

    ambigu_path = Path("../../audio/segments_ambigus_a_relire.csv")
    if ambigu_path.exists():
        df_ambigu_display = pd.read_csv(ambigu_path)
        if not df_ambigu_display.empty:
            st.divider()
            st.subheader("⚠️ Segments ambigus à relire à la main")
            st.caption(
                f"{len(df_ambigu_display)} segment(s) mentionnant plusieurs salles à la fois "
                "— non attribués automatiquement, à interpréter manuellement."
            )
            st.dataframe(
                df_ambigu_display[["participant", "source_attribution", "text"]],
                width="stretch",
            )
