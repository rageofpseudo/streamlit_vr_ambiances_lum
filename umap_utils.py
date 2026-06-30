"""
umap_utils.py contient toute la logique "pure" (chargement, fenêtrage, UMAP) et 
app.py ne s'occupe que de l'interface.

C'est important : si demain tu veux faire la même analyse dans un notebook, 
tu importes juste umap_utils sans toucher à Streamlit.
-------------
Prétraitement des données physiologiques/comportementales + calcul UMAP.

Philosophie :
- On travaille sur UN sujet à la fois (un CSV = un sujet)
- On expose des fonctions pures (pas d'état global) pour que Streamlit puisse
  les appeler avec @st.cache_data sans effets de bord

@st.cache_data — Streamlit recalcule tout à chaque interaction. 
Le cache fait que si tu changes juste la couleur des points, l'UMAP n'est pas recalculé. 
Si tu changes n_neighbors, il l'est. 
C'est Streamlit qui gère ça automatiquement via le hash des arguments.

----
Comment on "NORMALISE?"
Le z-score pour EDA et HR c'est la normalisation par rapport à la baseline de chaque sujet.

La formule appliquée
pythonbaseline = df[col].iloc[:baseline_samples]   # 30 premières secondes
df[col + "_zscore"] = (df[col] - baseline.mean()) / baseline.std()
Donc pour chaque sujet :
zscore(t) = (valeur(t) - moyenne_baseline) / écart_type_baseline

Imaginons PARTICIPAN1, EDA baseline moyenne = 5 µS, std = 0.5 µS :
salle 1 : EDA = 5.2 µS  →  zscore = (5.2 - 5) / 0.5  =  +0.4
salle 3 : EDA = 6.5 µS  →  zscore = (6.5 - 5) / 0.5  =  +3.0
salle 5 : EDA = 4.8 µS  →  zscore = (4.8 - 5) / 0.5  =  -0.4
Le zscore répond à la question : "De combien d'écarts-types ce sujet s'éloigne-t-il de sa propre baseline ?"

---
Comparaison ANOVA/MANOVA:

ANOVA teste un signal à la fois :
EDA ~ salle   (une équation)
HR  ~ salle   (une autre équation)
HRV ~ salle   (encore une autre)

MANOVA teste le vecteur [EDA, HR, HRV] simultanément :
[EDA, HR, HRV] ~ salle   (une seule équation multivariée)
Comment elle calcule
Elle cherche une combinaison linéaire de tes signaux qui maximise la séparation entre les salles :
score = a x EDA + b x HR + c x HRV

Wilks' Lambda  →  proche de 0 = bonne séparation
                  proche de 1 = pas de séparation

Pillai's trace →  proche de 1 = bonne séparation
                  proche de 0 = pas de séparation

-----

C'est quoi l'IBI ? Inter Beat Interval en secondes
IBI = temps entre deux battements cardiaques successifs.
Donc HR = 60/IBI

Pour la HRV, on préfère toujours partir des IBI.
"""

import os
# Workaround pour un crash macOS classique : xgboost embarque sa propre lib OpenMP,
# qui entre en conflit avec celle déjà chargée par numpy/umap/numba ("OMP: Error #179:
# Function pthread_mutex_init failed" → segfault). À mettre avant tout import qui
# charge OpenMP. Sans danger ici (un seul vrai runtime OpenMP utilisé en pratique).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pywt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import umap
from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter
from scipy import stats as scipy_stats
import cvxeda

# ANOVA
from scipy import stats
from statsmodels.multivariate.manova import MANOVA
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM  # ANOVA à mesures répétées
from sklearn.decomposition import PCA

# NON-PARAMETRIQUE (Aucune hypothèse sur la distribution)
from scipy.stats import friedmanchisquare, mannwhitneyu, wilcoxon

# NORMALITY
from scipy.stats import shapiro, skew, kurtosis

# VERIFICATION POUR SAVOIR QUELLES SALLES DIFFERENT ENTRE ELLES
from scikit_posthocs import posthoc_dunn

# EXTRAIRE PLUS DE FEATURES
from scipy.signal import find_peaks

# DISCRIMINANT
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import pingouin as pg

# streamlit
import streamlit as st

# extraction features cognitive3D
from scipy.spatial.transform import Rotation

# evaluation de features
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from scipy.signal import resample

# Encoding
from momentfm import MOMENTPipeline
import torch
from umap import UMAP

# Clustering hierarchique
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

# metrics et evaluation
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ── Colonnes utilisées pour l'UMAP ────────────────────────────────────────────

# Colonnes disponibles dans df_raw (signal brut)
DISPLACEMENT_COLS_RAW = ["head_x", "head_y", "head_z"]

# Colonnes disponibles dans df_agg (features agrégées)
DISPLACEMENT_COLS_AGG = ["head_x", "head_y", "head_z"]  # les mêmes existent dans df_agg aussi

NEW_FEATURES_RAW = [
    # Physiologie de base
    "hr_bpm_filtered",
    "eda_uS_filtered",
    "eda_phasic",
    "eda_tonic",
    "eda_driver",
    "hrv_rmssd",
    "hrv_rmssd_log",
    "bvp",
    "temp_C",
    # Versions zscore (normalisées)
    "eda_phasic_zscore",
    "eda_tonic_zscore",
    "eda_driver_zscore",
    "hr_bpm_filtered_zscore",
    "eda_uS_filtered_zscore",
    "hrv_rmssd_zscore",
]

NEW_FEATURES_AGG = [
    "head_y_zcr", "head_z_spectral_centroid", "jerk_y_fft_max",
    "jerk_y_min", "jerk_y_std", "jerk_y_variance",
    "jerk_y_wavelet_std", "pitch_wavelet_std"
]

# Colonnes d'événements utiles pour la viz (pas dans l'UMAP, juste metadata)
EVENT_COLS = ["c3d_event", "ev_Valence", "ev_Arousal", "ev_Room", "ev_salle",
              "ev_statut", "ev_sessionlength", "ev_Reason", "ev_objet"]


def check_password():
    pwd = st.text_input("Mot de passe", type="password")
    if pwd == st.secrets["PASSWORD"]:
        return True
    if pwd:  # évite le message d'erreur au chargement initial (champ vide)
        st.error("Mot de passe incorrect")
    return False

if not check_password():
    st.stop()

def run_normality_checks(df_agg: pd.DataFrame, signals:dict) -> pd.DataFrame:
    """
    Rappel :
    Shapiro-Wilk      → test formel : H0 = "les données suivent une loi normale"
    Skewness          → asymétrie : 0 = symétrique, >0 = queue à droite, <0 = gauche
    Kurtosis          → "épaisseur des queues" : 3 = normale, >3 = queues lourdes
    Q-Q plot          → visuel : les points doivent suivre la diagonale si normal

    ---
    W de Shapiro c'est quoi ?
    W mesure à quel point tes données ressemblent à une droite sur le Q-Q plot. Techniquement c'est le carré de la corrélation entre tes données triées et les quantiles théoriques d'une normale.
    W = 1.0  → parfaitement normal
    W = 0.9136 → légère déviation
    W proche de 0 → très loin de la normale
    En pratique tu regardes surtout le p-value — W seul n'est pas très intuitif. Les deux ensemble racontent l'histoire : W=0.91 p=0.003 signifie "assez loin de la normale, et c'est statistiquement significatif".
    ---

    Pour chaque signal × salle, calcule :
    - Shapiro-Wilk (p-value)
    - Skewness
    - Kurtosis
    - Conclusion : normal ou non
    
    Pourquoi par salle ?
    ANOVA suppose la normalité des résidus DANS chaque groupe (ici chaque salle),
    pas sur l'ensemble des données.
    """
    rows = []
    
    for label, col in signals.items():
        if col not in df_agg.columns:
            continue
            
            #for salle in sorted(df_agg["salle"].dropna().unique()):
            # avant on faisait par salle
            #data = df_agg[df_agg["salle"] == salle][col].dropna()
        data = df_agg.groupby("subject")[col].mean().dropna()

        if len(data) < 3:
            continue
            
            # Shapiro-Wilk : fiable pour n < 50, ce qui est ton cas (47 sujets)
            #stat, p_val = shapiro(data)
        stat, p_val = shapiro(data)
        rows.append({
                "Signal":    label,
                #"Salle":     int(salle),
                "N sujets":         len(data),
                "Shapiro W": round(stat, 4),
                "p-value":   round(p_val, 4),
                "Normal ?":  "✅" if p_val > 0.05 else "❌",
                "Skewness":  round(skew(data), 3),
                "Kurtosis":  round(kurtosis(data), 3),  # excess kurtosis (0 = normale)
        })
    
    return pd.DataFrame(rows)


def run_levene_test(df_agg: pd.DataFrame, signals: dict, group_col: str = "salle") -> pd.DataFrame:
    """
    Test de Levene — homogénéité des variances entre les groupes (salles).

    Pourquoi c'est important ?
    L'ANOVA classique (et dans une moindre mesure l'ANOVA RM) suppose que la
    variance du signal est similaire dans chaque groupe (homoscédasticité). Si une
    salle a une variance beaucoup plus grande qu'une autre, le test F devient
    moins fiable (plus de faux positifs ou de faux négatifs selon le sens du
    déséquilibre).

    Pourquoi Levene et pas le test de Bartlett (plus classique) ?
    Levene est plus robuste à la non-normalité (il teste l'écart à la médiane,
    pas à la moyenne) — cohérent avec le fait qu'on a déjà beaucoup de signaux
    non-normaux ici (cf. Shapiro-Wilk ci-dessus).

    H0 = "les variances sont égales entre les salles".
    p > 0.05 → on ne rejette pas H0 → variances homogènes, hypothèse de l'ANOVA respectée.
    p < 0.05 → variances hétérogènes → interpréter l'ANOVA/MANOVA avec prudence,
    ou préférer un test non-paramétrique (Friedman) qui n'a pas cette hypothèse.
    """
    rows = []

    for label, col in signals.items():
        if col not in df_agg.columns:
            continue

        df_clean = df_agg[[group_col, col]].dropna()
        groups = [
            g[col].values
            for _, g in df_clean.groupby(group_col)
            if len(g) > 1
        ]

        if len(groups) < 2:
            continue

        try:
            stat, p_val = stats.levene(*groups)
        except ValueError:
            continue

        rows.append({
            "Signal":      label,
            "N groupes":   len(groups),
            "Levene stat": round(stat, 4),
            "p-value":     round(p_val, 4),
            "Variances homogènes ?": "✅" if p_val > 0.05 else "❌",
        })

    return pd.DataFrame(rows)


def compute_signal_sam_correlation(
    df_agg: pd.DataFrame,
    df_sam: pd.DataFrame,
    signals: dict,
    method: str = "spearman",
) -> pd.DataFrame:
    """
    Corrélation entre chaque signal physio/mouvement (df_agg) et la réponse SAM
    (valence, arousal), calculée séparément PAR SALLE (à travers les participants).

    Pourquoi par salle et pas toutes salles mélangées ?
    Mélanger les salles confondrait deux choses différentes : "est-ce que le signal
    suit le ressenti subjectif au sein d'une même condition" (ce qu'on veut) vs "est-ce
    que la salle elle-même fait varier le signal ET le SAM en même temps" (un effet
    de salle qui créerait une fausse corrélation même sans lien direct signal↔ressenti).

    method="spearman" par défaut (rangs, pas la valeur brute) — plus robuste à la
    non-normalité, déjà très présente dans ce dataset (cf. Shapiro-Wilk).

    Retourne un DataFrame avec une ligne par (salle, signal, variable SAM).
    """
    df_merged = df_agg.merge(df_sam, on=["subject", "salle"], how="inner")

    corr_fn = stats.spearmanr if method == "spearman" else stats.pearsonr

    rows = []
    for salle, df_s in df_merged.groupby("salle"):
        for label, col in signals.items():
            if col not in df_s.columns:
                continue
            for sam_var in ["valence", "arousal"]:
                d = df_s[[col, sam_var]].dropna()
                if len(d) < 5 or d[col].std() < 1e-9:
                    continue

                r, p = corr_fn(d[col], d[sam_var])
                rows.append({
                    "Salle":        salle,
                    "Signal":       label,
                    "SAM":          sam_var,
                    "r":            round(r, 3),
                    "p-value":      round(p, 4),
                    "Significatif": "✅" if p < 0.05 else "❌",
                    "N":            len(d),
                })

    return pd.DataFrame(rows)


# Fonction générique qui affiche un boxplot propre par salle pour n'importe quelle feature
def plot_feature_boxplot(df_agg, feature, title=None, palette="Set2"):
    """
    df_agg : ton DataFrame agrégé (une ligne par participant × salle)
    feature : nom de la colonne à afficher

    Interprétation des boxplots
        Un boxplot montre :

        la boîte = IQR (25e–75e percentile)
        la ligne centrale = médiane
        les moustaches = 1.5 × IQR
        les points au-delà = outliers

        Si une feature a des boxplots qui ne se chevauchent pas entre salles, c'est un bon signe de discrimination. 
        C'est une inspection visuelle complémentaire à tes tests de Friedman/ANOVA.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    order = sorted(df_agg["salle"].unique())  # ordre cohérent
    
    sns.boxplot(
        data=df_agg,
        x="salle",
        y=feature,
        order=order,
        palette=palette,
        width=0.5,
        linewidth=1.2,
        flierprops=dict(marker='o', markersize=4, alpha=0.6),
        ax=ax
    )
    
    # Superposer les points individuels (transparents) pour voir la distribution réelle
    #   Le stripplot superposé est important : avec peu de participants, le boxplot seul peut être trompeur (médiane basée sur 8 points, ça se voit). 
    #   Les points individuels montrent la vraie dispersion.
    sns.stripplot(
        data=df_agg,
        x="salle",
        y=feature,
        order=order,
        color="black",
        alpha=0.35,
        size=4,
        jitter=True,
        ax=ax
    )
    
    ax.set_title(title or feature, fontsize=13, pad=10)
    ax.set_xlabel("Salle")
    ax.set_ylabel(feature)
    sns.despine()
    st.pyplot(fig)
    plt.close()


def evaluate_feature_combo(df_agg, feature_list, label_col="salle"):
    """
    Évalue une combinaison de features par LOO-CV sur LDA.

    LeaveOneOut : à chaque itération, on entraîne le modèle sur tous les sujets sauf un, puis on teste sur celui qu'on a laissé de côté. C'est le CV le plus honnête quand tu as peu de sujets (< 30).
    LinearDiscriminantAnalysis : cherche la projection linéaire qui maximise la séparation entre tes classes (salles). Parfait comme proxy de "ces features discriminent bien".
    Le score final = % de bonnes prédictions de salle.

    Retourne l'accuracy moyenne.
    """
    X = df_agg[feature_list].dropna()
    y = df_agg.loc[X.index, label_col]
    
    if len(X) < 5:  # pas assez de données
        return np.nan
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    loo = LeaveOneOut()
    lda = LinearDiscriminantAnalysis()
    preds, truths = [], []
    
    for train_idx, test_idx in loo.split(X_scaled):
        lda.fit(X_scaled[train_idx], y.iloc[train_idx])
        preds.append(lda.predict(X_scaled[test_idx])[0])
        truths.append(y.iloc[test_idx].values[0])
    
    return accuracy_score(truths, preds)
 
# Extraction de la feature SCR depuis EDA -> selon certains articles c'est pas tout le temps la plus utile.
def extract_scr_features(phasic: pd.Series, tonic: pd.Series, fs: float = 4.0) -> dict:
    """
    Extrait les features des Skin Conductance Responses (SCR) depuis le signal phasique.
    phasic : eda_phasic (composante rapide, contient les SCR)
    tonic  : eda_tonic  (composante lente, niveau de base)
    fs     : fréquence d'échantillonnage en Hz (E4 = 4 Hz)
    
    Pourquoi le signal phasique et pas le brut ?
    cvxEDA a déjà isolé la composante rapide (réponses aux stimuli).
    Travailler sur le brut mélangerait SCR et dérive lente (tonic).
    
    threshold=0.02 : seuil en µS en dessous duquel on ignore les micro-fluctuations.
    Benedek & Kaernbach (2010) recommandent 0.01-0.05 µS selon le bruit du signal.
    
    distance=fs*2 : deux pics doivent être séparés d'au moins 2s.
    Correspond à la période réfractaire physiologique d'une SCR.
    """
    features = {}

    # ══════════════════════════════════════════════════════════
    # A. Features sur le signal TONIQUE (niveau de base EDA)
    #    Le tonique reflète l'arousal de fond — stats temporelles classiques.
    # ══════════════════════════════════════════════════════════
    t = tonic.dropna().values
    if len(t) >= 10:
        features["eda_tonic_mean"]     = np.mean(t)
        features["eda_tonic_std"]      = np.std(t)
        features["eda_tonic_min"]      = np.min(t)
        features["eda_tonic_max"]      = np.max(t)
        features["eda_tonic_median"]   = np.median(t)
        features["eda_tonic_kurtosis"] = float(scipy_stats.kurtosis(t))
        features["eda_tonic_skewness"] = float(scipy_stats.skew(t))
        features["eda_tonic_variance"] = np.var(t)
        features["eda_tonic_rms"]      = np.sqrt(np.mean(t**2))
        features["eda_tonic_iqr"]      = float(scipy_stats.iqr(t))
        features["eda_tonic_peak2peak"]= np.max(t) - np.min(t)
        features["eda_tonic_mad"]      = np.mean(np.abs(t - np.mean(t)))
        features["eda_tonic_autocorr"] = float(np.corrcoef(t[:-1], t[1:])[0, 1]) if len(t) > 1 else np.nan
        features["eda_tonic_mean_abs_diff"] = np.mean(np.abs(np.diff(t)))

        # FFT sur le tonique (variations lentes de l'arousal)
        fft_vals = np.abs(np.fft.rfft(t))
        freqs    = np.fft.rfftfreq(len(t), d=1/fs)
        features["eda_tonic_fft_mean"] = np.mean(fft_vals)
        features["eda_tonic_fft_max"]  = np.max(fft_vals)
        # Centroïde spectral : fréquence "moyenne" pondérée par l'énergie
        # → signal basse fréquence (variations lentes) = centroïde bas
        features["eda_tonic_spectral_centroid"] = (
            float(np.sum(freqs * fft_vals) / np.sum(fft_vals))
            if np.sum(fft_vals) > 0 else np.nan
        )

        # Wavelet : décompose le signal en niveaux d'échelle temporelle
        # db4 = Daubechies 4 — bon compromis temps/fréquence pour les signaux physiologiques
        # level=3 → 3 niveaux de décomposition (approximation + 3 détails)
        try:
            coeffs = pywt.wavedec(t, 'db4', level=3)
            features["eda_tonic_wavelet_energy"] = float(sum(np.sum(c**2) for c in coeffs))
            features["eda_tonic_wavelet_std"]    = float(np.std(coeffs[0]))  # coefficients d'approximation
        except Exception:
            features["eda_tonic_wavelet_energy"] = np.nan
            features["eda_tonic_wavelet_std"]    = np.nan

    # ══════════════════════════════════════════════════════════
    # B. Features sur le signal PHASIQUE (SCR)
    #    Le phasique = réponses rapides. Stats + détection de pics.
    # ══════════════════════════════════════════════════════════
    p = phasic.dropna().values
    if len(p) >= 10:
        features["scr_auc"]            = float(np.trapz(np.maximum(p, 0)))
        features["eda_phasic_mean"]    = np.mean(p)
        features["eda_phasic_std"]     = np.std(p)
        features["eda_phasic_max"]     = np.max(p)
        features["eda_phasic_kurtosis"]= float(scipy_stats.kurtosis(p))
        features["eda_phasic_skewness"]= float(scipy_stats.skew(p))
        features["eda_phasic_iqr"]     = float(scipy_stats.iqr(p))
        features["eda_phasic_mad"]     = np.mean(np.abs(p - np.mean(p)))
        features["eda_phasic_mean_abs_diff"] = np.mean(np.abs(np.diff(p)))

        # ZCR sur le phasique centré — a du sens car le phasique oscille autour de 0
        p_centered = p - np.mean(p)
        features["eda_phasic_zcr"] = int(np.sum(np.diff(np.sign(p_centered)) != 0))

        # FFT phasique
        fft_vals_p = np.abs(np.fft.rfft(p))
        freqs_p    = np.fft.rfftfreq(len(p), d=1/fs)
        features["eda_phasic_fft_mean"] = np.mean(fft_vals_p)
        features["eda_phasic_fft_max"]  = np.max(fft_vals_p)
        features["eda_phasic_spectral_centroid"] = (
            float(np.sum(freqs_p * fft_vals_p) / np.sum(fft_vals_p))
            if np.sum(fft_vals_p) > 0 else np.nan
        )

        # Détection de pics SCR (seuil 0.01 µS — standard)
        # On cherche les "bosses" du signal phasique
        from scipy.signal import find_peaks
        peaks, props = find_peaks(p, height=0.01, distance=int(fs))
        features["scr_rate"]           = len(peaks) / (len(p) / fs) * 60  # SCR/min
        features["scr_amplitude_mean"] = float(np.mean(props["peak_heights"])) if len(peaks) > 0 else 0.0
        features["scr_amplitude_std"]  = float(np.std(props["peak_heights"]))  if len(peaks) > 1 else 0.0

        try:
            coeffs_p = pywt.wavedec(p, 'db4', level=3)
            features["eda_phasic_wavelet_energy"] = float(sum(np.sum(c**2) for c in coeffs_p))
            features["eda_phasic_wavelet_std"]    = float(np.std(coeffs_p[0]))
        except Exception:
            features["eda_phasic_wavelet_energy"] = np.nan
            features["eda_phasic_wavelet_std"]    = np.nan

    # ══════════════════════════════════════════════════════════
    # C. CF1 / CF2 — features composites (Saha et al. 2025,
    #    "Differentiating presence in virtual reality using physiological signals")
    #    Combinent variance + pente (vitesse de changement) du tonique et du
    #    phasique pour capturer simultanément l'arousal de fond (lent) et les
    #    réponses transitoires (rapides) en une seule métrique.
    # ══════════════════════════════════════════════════════════
    if len(t) >= 10 and len(p) >= 10:
        var_tonic  = float(np.var(t))
        var_phasic = float(np.var(p))

        # Pente = coefficient de régression linéaire du signal contre l'index
        # temporel (frame), ex: Eq. (3) du papier — utilisée comme un angle (radians)
        # dans tan(), donc reste petite en pratique (rad, pas degrés).
        slope_tonic  = float(np.polyfit(np.arange(len(t)), t, 1)[0])
        slope_phasic = float(np.polyfit(np.arange(len(p)), p, 1)[0])

        # CF1 — magnitude d'un "vecteur résultant" combinant tonique et phasique
        features["cf1_presence"] = float(np.sqrt(
            (var_phasic * np.tan(slope_phasic))**2 + (var_tonic * np.tan(slope_tonic))**2
        ))

        # Power phasique : énergie moyenne du signal (moyenne des carrés)
        power_phasic = float(np.mean(p**2))
        # CF2 — variance phasique normalisée par l'énergie phasique : intensité
        # relative des réponses transitoires, indépendamment de l'amplitude absolue.
        features["cf2_presence"] = float(var_phasic / power_phasic) if power_phasic > 1e-12 else np.nan
    else:
        features["cf1_presence"] = np.nan
        features["cf2_presence"] = np.nan

    return features


# Extraction de features SDNN, pNN50, LF/HF ratio, HR range depuis HR.
def extract_hrv_features(ibi_series: pd.Series, hr_series: pd.Series = None, fs_hr: float = 1.0) -> dict:
    """
  
    SDNN.       --- Écart-type de tous les IBI                      --- Variabilité globale (vs RMSSD qui est court-terme)
    pNN50       --- % d'IBI consécutifs différant de >50ms          --- Activité parasympathique
    LF/HF ratio --- FFT sur les IBI, bande 0.04-0.15Hz / 0.15-0.4Hz --- Équilibre sympathique/parasympathique
    HR range    --- max(HR) - min(HR) sur la salle                  --- Réactivité cardiaque totale
    
    ibi_series : intervalles RR en secondes (colonne ibi_s)
    hr_series  : fréquence cardiaque en bpm (colonne hr_bpm) — optionnel
    fs_hr      : fréquence d'échantillonnage du HR (E4 = 1 Hz)
    """
    features = {}

    # ── A. Features HRV classiques sur les IBI ────────────────────────────────
    ibi = ibi_series.dropna().values * 1000  # conversion s → ms
    if len(ibi) >= 10:
        diff = np.diff(ibi)
        features["sdnn"]  = float(np.std(ibi, ddof=1))
        features["rmssd"] = float(np.sqrt(np.mean(diff**2)))
        features["pnn50"] = float(np.sum(np.abs(diff) > 50) / len(diff) * 100)
        features["ibi_mean"]     = float(np.mean(ibi))
        features["ibi_cv"]       = float(np.std(ibi) / np.mean(ibi))  # coefficient de variation
        features["ibi_skewness"] = float(scipy_stats.skew(ibi))
        features["ibi_kurtosis"] = float(scipy_stats.kurtosis(ibi))

        # LF/HF ratio via FFT sur les IBI interpolés
        # Les IBI sont irrégulièrement espacés → on doit interpoler avant FFT
        # On réinterprète chaque IBI comme un point dans le temps
        try:
            t_ibi = np.cumsum(ibi) / 1000.0  # temps en secondes
            t_ibi -= t_ibi[0]                 # commence à 0
            # Rééchantillonnage à 4 Hz (standard HRV)
            fs_resample = 4.0
            t_uniform = np.arange(0, t_ibi[-1], 1/fs_resample)
            ibi_uniform = np.interp(t_uniform, t_ibi, ibi)

            fft_vals = np.abs(np.fft.rfft(ibi_uniform))
            freqs    = np.fft.rfftfreq(len(ibi_uniform), d=1/fs_resample)

            # Bandes standard HRV
            lf_mask = (freqs >= 0.04) & (freqs < 0.15)
            hf_mask = (freqs >= 0.15) & (freqs < 0.40)
            lf_power = np.sum(fft_vals[lf_mask]**2)
            hf_power = np.sum(fft_vals[hf_mask]**2)

            features["hrv_lf_power"] = float(lf_power)
            features["hrv_hf_power"] = float(hf_power)
            features["hrv_lf_hf"]    = float(lf_power / hf_power) if hf_power > 0 else np.nan
            # LF/HF > 1 → dominance sympathique (stress)
            # LF/HF < 1 → dominance parasympathique (repos)
        except Exception:
            features["hrv_lf_power"] = np.nan
            features["hrv_hf_power"] = np.nan
            features["hrv_lf_hf"]    = np.nan

    # ── B. Stats temporelles sur HR (bpm) ─────────────────────────────────────
    if hr_series is not None:
        hr = hr_series.dropna().values
        if len(hr) >= 10:
            features["hr_mean"]      = float(np.mean(hr))
            features["hr_std"]       = float(np.std(hr))
            features["hr_min"]       = float(np.min(hr))
            features["hr_max"]       = float(np.max(hr))
            features["hr_range"]     = float(np.max(hr) - np.min(hr))
            features["hr_median"]    = float(np.median(hr))
            features["hr_kurtosis"]  = float(scipy_stats.kurtosis(hr))
            features["hr_skewness"]  = float(scipy_stats.skew(hr))
            features["hr_iqr"]       = float(scipy_stats.iqr(hr))
            features["hr_rms"]       = float(np.sqrt(np.mean(hr**2)))
            features["hr_mean_abs_diff"] = float(np.mean(np.abs(np.diff(hr))))
            features["hr_autocorr"]  = (
                float(np.corrcoef(hr[:-1], hr[1:])[0, 1]) if len(hr) > 1 else np.nan
            )

    return features


def compute_exploration_entropy(angles: np.ndarray, n_bins: int = 18) -> float:
    """
    Entropie de Shannon de la distribution d'un angle (pitch ou yaw) sur la salle.

    Littérature attention/stress : sous menace ou anxiété, le regard se "rétrécit"
    (attentional narrowing) — la distribution des angles se concentre dans une plage
    étroite, donc l'entropie baisse. À l'inverse, une exploration large et variée
    (curiosité, confort) donne une entropie élevée.

    n_bins=18 → des bins de 20° si l'angle couvre 360°, résolution raisonnable pour
    de la rotation de tête sans être trop sensible au bruit frame-à-frame.

    Retourne l'entropie en bits (log base 2) — 0 = toujours le même angle,
    log2(n_bins) = parfaitement uniforme sur toute la plage observée.
    """
    angles = angles[~np.isnan(angles)]
    if len(angles) < 10:
        return np.nan

    hist, _ = np.histogram(angles, bins=n_bins, density=False)
    probs = hist[hist > 0] / hist.sum()
    return float(-np.sum(probs * np.log2(probs)))


def compute_room_entry_response(df_salle: pd.DataFrame, window_sec: float = 3.0) -> dict:
    """
    Réaction immédiate à l'entrée dans la salle (les `window_sec` premières secondes
    après l'événement `entree_nouvelle_room`), plutôt qu'une moyenne sur toute la visite.

    Pourquoi ? La littérature sur la réponse de sursaut/orientation ("startle response")
    regarde une fenêtre courte juste après le changement de stimulus — une réaction
    brève peut être complètement diluée si on moyenne sur 1-5 minutes de visite, ce
    qu'on a fait partout jusqu'ici. Ici on capture spécifiquement le pic d'activité
    juste après le changement d'ambiance lumineuse.

    Retourne NaN partout si l'événement n'existe pas pour cette salle (ex: salle 1,
    qui n'a pas d'`entree_nouvelle_room` puisque le participant y commence directement).
    """
    empty = {
        "entry_speed_peak": np.nan, "entry_speed_mean": np.nan,
        "entry_jerk_peak": np.nan, "entry_angular_velocity_peak": np.nan,
    }

    entry_rows = df_salle[df_salle["c3d_event"] == "entree_nouvelle_room"]
    if entry_rows.empty:
        return empty

    t_entry = entry_rows["timestamp"].iloc[0]
    window = df_salle[
        (df_salle["timestamp"] >= t_entry) & (df_salle["timestamp"] < t_entry + window_sec)
    ]
    if len(window) < 5:
        return empty

    dx = window["head_x"].diff()
    dy = window["head_y"].diff()
    dz = window["head_z"].diff()
    dt = window["timestamp"].diff().replace(0, np.nan)

    speed = (np.sqrt(dx**2 + dy**2 + dz**2) / dt).dropna()
    result = {
        "entry_speed_peak": float(speed.max()) if len(speed) else np.nan,
        "entry_speed_mean": float(speed.mean()) if len(speed) else np.nan,
    }

    vel_axis = (dx / dt)  # un axe suffit pour une mesure de "sursaut" relative
    jerk = (vel_axis.diff() / dt).dropna()
    result["entry_jerk_peak"] = float(jerk.abs().max()) if len(jerk) else np.nan

    q_cols = ["head_qx", "head_qy", "head_qz", "head_qw"]
    if all(c in window.columns for c in q_cols):
        quats = window[q_cols].dropna().values
        if len(quats) > 1:
            rots = Rotation.from_quat(quats)
            delta_rots = rots[:-1].inv() * rots[1:]
            angular_disp = delta_rots.magnitude()
            dt_clean = dt.dropna().values
            n = min(len(angular_disp), len(dt_clean))
            if n > 0:
                angular_vel = angular_disp[:n] / (dt_clean[:n] + 1e-9)
                result["entry_angular_velocity_peak"] = float(np.max(angular_vel))
            else:
                result["entry_angular_velocity_peak"] = np.nan
        else:
            result["entry_angular_velocity_peak"] = np.nan
    else:
        result["entry_angular_velocity_peak"] = np.nan

    return result


def compute_time_to_find_watch(subjects_data: dict) -> pd.DataFrame:
    """
    Temps mis par chaque participant pour interagir une première fois avec la montre
    dans chaque salle (latence entre l'entrée dans la salle et l'événement
    `premiere_interaction_montre`) — une mesure de performance/efficacité de recherche,
    indépendante des features physio/mouvement déjà testées.

    Pour la salle 1 (pas d'événement `entree_nouvelle_room`, le participant y commence
    directement), le temps de référence est le premier timestamp de la session.

    Retourne un DataFrame avec une ligne par (subject, salle) où la montre a été trouvée
    (les salles sans `premiere_interaction_montre` détecté sont absentes, pas mises à NaN).
    """
    rows = []
    for subject, df in subjects_data.items():
        for salle in sorted(df["ev_salle"].dropna().unique()):
            df_salle = df[df["ev_salle"] == salle]

            entry_rows = df_salle[df_salle["c3d_event"] == "entree_nouvelle_room"]
            t_entry = entry_rows["timestamp"].iloc[0] if not entry_rows.empty else df_salle["timestamp"].iloc[0]

            montre_rows = df_salle[df_salle["c3d_event"] == "premiere_interaction_montre"]
            if montre_rows.empty:
                continue

            t_montre = montre_rows["timestamp"].iloc[0]
            rows.append({
                "subject": subject,
                "salle": salle,
                "temps_trouver_montre": float(t_montre - t_entry),
            })

    return pd.DataFrame(rows)


def extract_c3d_features(df: pd.DataFrame) -> dict:
    """
    Extrait des features scalaires de mouvement depuis un DataFrame brut (1 sujet × 1 salle).

    Structure :
    ──────────────────────────────────────────────────────────────
    1. Vitesse de déplacement (speed)          → scalaires simples
    2. Jerk (dérivée de la vitesse)            → brusquerie du mouvement
    3. Statisme (immobility)                   → ratio + bouts
    4. Rotation de tête (quaternions)          → vitesse angulaire + amplitude
    5. Features stats/temporelles/spectrales   → sur head_x/y/z, pitch/roll/yaw, jerk_x/y/z
    ──────────────────────────────────────────────────────────────
    """

    features = {}

    # ── Constantes ────────────────────────────────────────────────────────────
    IMMOBILITY_THRESHOLD = 0.05   # m/s — en dessous = "figé"
    # Taux d'échantillonnage réel (calculé depuis les timestamps dans load_subject,
    # propagé via la colonne "fs_reel") plutôt qu'une valeur supposée — fallback à
    # 9 Hz (médiane observée sur les données réelles) si la colonne est absente.
    FS = float(df["fs_reel"].iloc[0]) if "fs_reel" in df.columns and len(df) > 0 else 9.0

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. VITESSE DE DÉPLACEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    # diff() sur x/y/z donne le déplacement entre deux frames consécutives.
    # On divise par dt pour avoir une vitesse en m/s.
    # La 1ère ligne vaut NaN (pas de "frame précédente") → on drop proprement.

    dx = df["head_x"].diff()
    dy = df["head_y"].diff()
    dz = df["head_z"].diff()
    dt = df["timestamp"].diff().replace(0, np.nan)

    speed = np.sqrt(dx**2 + dy**2 + dz**2) / dt   # m/s, NaN sur la 1ère ligne
    speed_clean = speed.dropna()

    features["speed_mean"]   = speed_clean.mean()
    features["speed_std"]    = speed_clean.std()
    features["speed_median"] = speed_clean.median()
    features["path_length"]  = (np.sqrt(dx**2 + dy**2 + dz**2)).sum()

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. JERK (dérivée de la vitesse → brusquerie du mouvement)
    # ═══════════════════════════════════════════════════════════════════════════
    # Le jerk = d(vitesse)/dt. On l'approche par diff(speed) / dt.
    # Un jerk élevé = mouvement saccadé, imprévisible.
    # Pertinent pour détecter stress ou désorientation en VR.
    #
    # Pourquoi calculer jerk_x/y/z séparément plutôt que sur speed scalaire ?
    # → On peut distinguer l'axe de brusquerie (ex: jerk vertical élevé = sursaut)

    for axis, d_axis in zip(["x", "y", "z"], [dx, dy, dz]):
        vel_axis = d_axis / dt                          # vitesse sur cet axe (m/s)
        jerk_axis = vel_axis.diff() / dt               # variation de vitesse / dt → jerk (m/s²/s)
        jerk_clean = jerk_axis.dropna().values
        # Nom distinct de jerk_{axis}_mean (signé, calculé plus bas par _enrich_signal)
        # pour éviter une collision silencieuse : celui-ci est l'intensité (valeur absolue),
        # l'autre est la dérive directionnelle nette.
        features[f"jerk_{axis}_mean_abs"] = float(np.mean(np.abs(jerk_clean)))
        features[f"jerk_{axis}_std"]      = float(np.std(jerk_clean))

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. STATISME (immobility)
    # ═══════════════════════════════════════════════════════════════════════════

    is_immobile = speed_clean < IMMOBILITY_THRESHOLD
    features["immobility_ratio"] = is_immobile.mean()

    # Nombre de "bouts" = passages False→True dans la série booléenne.
    # ratio=0.5 peut être 1 long épisode ou 50 courts → le bout_count distingue les deux.
    bouts = (is_immobile.astype(int).diff() == 1).sum()
    features["immobility_bouts"] = int(bouts)

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. ROTATION DE TÊTE (quaternions → vitesse angulaire + amplitude Euler)
    # ═══════════════════════════════════════════════════════════════════════════
    # On ne peut pas faire diff() sur des quaternions (espace non-euclidien).
    # On calcule la rotation *relative* entre frames consécutives avec inv() * suivant.
    # .magnitude() donne l'angle de cette rotation delta en radians.

    q_cols = ["head_qx", "head_qy", "head_qz", "head_qw"]
    if all(c in df.columns for c in q_cols):
        quats = df[q_cols].dropna().values
        if len(quats) > 1:
            rots = Rotation.from_quat(quats)
            delta_rots = rots[:-1].inv() * rots[1:]
            angular_displacements = delta_rots.magnitude()

            dt_clean = dt.dropna().values
            n = min(len(angular_displacements), len(dt_clean))
            angular_velocity = angular_displacements[:n] / (dt_clean[:n] + 1e-9)

            features["angular_velocity_mean"] = float(np.mean(angular_velocity))
            features["angular_velocity_std"]  = float(np.std(angular_velocity))

            euler = rots.as_euler("yxz", degrees=True)   # [pitch, yaw, roll]
            features["head_yaw_range"]   = float(np.ptp(euler[:, 1]))
            features["head_pitch_range"] = float(np.ptp(euler[:, 0]))

            # On extrait aussi pitch/roll/yaw comme séries pour les features enrichies ci-dessous
            pitch_vals = euler[:, 0]
            yaw_vals   = euler[:, 1]
            roll_vals  = euler[:, 2]
        else:
            for k in ["angular_velocity_mean", "angular_velocity_std",
                      "head_yaw_range", "head_pitch_range"]:
                features[k] = np.nan
            pitch_vals = yaw_vals = roll_vals = np.array([])
    else:
        pitch_vals = yaw_vals = roll_vals = np.array([])

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. FEATURES ENRICHIES : stats / temporelles / spectrales / wavelets
    #    Sur : head_x, head_y, head_z, pitch, roll, yaw, jerk_x, jerk_y, jerk_z
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # Pourquoi ces features sur le signal brut ?
    # ─────────────────────────────────────────────
    # head_x/y/z bruts : la position absolue encode où le participant se tient
    #   dans la salle — pas seulement à quelle vitesse il se déplace.
    #   Ex: quelqu'un qui reste dans un coin vs. quelqu'un qui explore tout l'espace.
    #
    # pitch/roll/yaw : l'amplitude et la variabilité de l'orientation de tête
    #   capturent l'exploration visuelle et la curiosité.
    #
    # jerk_x/y/z : la brusquerie axe par axe — un jerk vertical élevé peut indiquer
    #   un sursaut, un jerk latéral élevé une désorientation.
    #
    # Features calculées pour chaque signal :
    # ─────────────────────────────────────────
    # STATISTIQUES : mean, std, min, max, median, kurtosis, skewness, variance,
    #                rms, iqr, peak2peak, mad
    # TEMPORELLES  : zcr (zero-crossing rate), autocorr lag-1, mean_abs_diff
    # SPECTRALES   : fft_mean, fft_max, spectral_centroid
    # WAVELETS     : wavelet_energy, wavelet_std (décomposition db4 niveau 3)

    def _enrich_signal(name: str, x: np.ndarray):
        """
        Calcule ~20 features sur un signal 1D et les insère dans `features`.
        `name` est le préfixe utilisé pour nommer chaque feature (ex: 'head_x').
        """
        if len(x) < 10:
            return   # pas assez de données → on laisse ces features absentes (NaN implicite)

        # -- Statistiques de base --
        # mean signé (pas abs) : pour jerk_x/y/z, c'est une dérive directionnelle
        # nette, pas une intensité — voir jerk_{axis}_mean_abs pour l'intensité.
        features[f"{name}_mean"]      = float(np.mean(x))
        features[f"{name}_std"]       = float(np.std(x))
        features[f"{name}_min"]       = float(np.min(x))
        features[f"{name}_max"]       = float(np.max(x))
        features[f"{name}_median"]    = float(np.median(x))
        features[f"{name}_kurtosis"]  = float(stats.kurtosis(x))   # "piquant" de la distribution
        features[f"{name}_skewness"]  = float(stats.skew(x))       # asymétrie
        features[f"{name}_variance"]  = float(np.var(x))
        features[f"{name}_rms"]       = float(np.sqrt(np.mean(x**2)))  # énergie globale
        features[f"{name}_iqr"]       = float(stats.iqr(x))            # écart interquartile (robuste)
        features[f"{name}_peak2peak"] = float(np.max(x) - np.min(x))   # amplitude totale
        features[f"{name}_mad"]       = float(np.mean(np.abs(x - np.mean(x))))  # déviation absolue moyenne

        # -- Temporelles --
        # ZCR : combien de fois le signal croise zéro → mesure d'agitation/oscillation
        # Attention : n'a de sens que si le signal est centré (peut osciller autour de 0).
        # Pour head_x/y/z (positions absolues), le ZCR sera souvent 0 → peu informatif.
        # Pour pitch/yaw/roll et jerk, c'est pertinent.
        features[f"{name}_zcr"]           = int(np.sum(np.diff(np.sign(x)) != 0))
        features[f"{name}_autocorr"]      = float(np.corrcoef(x[:-1], x[1:])[0, 1])  # autocorr lag-1
        features[f"{name}_mean_abs_diff"] = float(np.mean(np.abs(np.diff(x))))        # variation frame à frame

        # -- Spectrales (FFT) --
        # La FFT décompose le signal en fréquences.
        # fft_mean : énergie spectrale globale
        # fft_max  : pic d'énergie dominant
        # spectral_centroid : "centre de gravité" fréquentiel — signal basse fréquence (mouvements lents)
        #                     vs haute fréquence (tremblements, saccades)
        fft_vals = np.abs(np.fft.rfft(x))
        freqs    = np.fft.rfftfreq(len(x), d=1.0 / FS)
        features[f"{name}_fft_mean"]          = float(np.mean(fft_vals))
        features[f"{name}_fft_max"]           = float(np.max(fft_vals))
        features[f"{name}_spectral_centroid"] = float(
            np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-9)
        )

        # -- Wavelets --
        # La décomposition wavelet (db4, 3 niveaux) décompose le signal en composantes
        # à différentes échelles temporelles (comme une FFT mais localisée dans le temps).
        # coeffs[0] = approximation basse fréquence (tendance globale)
        # coeffs[1:] = détails haute fréquence (variations rapides)
        # wavelet_energy : énergie totale du signal décomposé
        # wavelet_std    : variabilité de la composante lente (tendance de fond)
        coeffs = pywt.wavedec(x, "db4", level=3)
        features[f"{name}_wavelet_energy"] = float(sum(np.sum(c**2) for c in coeffs))
        features[f"{name}_wavelet_std"]    = float(np.std(coeffs[0]))

    # ── Application sur chaque signal ─────────────────────────────────────────

    # Positions brutes
    _enrich_signal("head_x", df["head_x"].dropna().values)
    _enrich_signal("head_y", df["head_y"].dropna().values)
    _enrich_signal("head_z", df["head_z"].dropna().values)

    # Orientations (Euler) — disponibles seulement si les quaternions étaient présents
    if len(pitch_vals) > 0:
        _enrich_signal("pitch", pitch_vals)
        _enrich_signal("yaw",   yaw_vals)
        _enrich_signal("roll",  roll_vals)

        # Entropie d'exploration du regard — voir compute_exploration_entropy()
        features["pitch_entropy"] = compute_exploration_entropy(pitch_vals)
        features["yaw_entropy"]   = compute_exploration_entropy(yaw_vals)

    # Jerk par axe (signal déjà calculé en section 2 — on le recalcule proprement ici)
    for axis, d_axis in zip(["x", "y", "z"], [dx, dy, dz]):
        vel_axis  = (d_axis / dt).dropna()
        jerk_axis = (vel_axis.diff() / dt).dropna().values
        _enrich_signal(f"jerk_{axis}", jerk_axis)

    features = {k: float(v) if isinstance(v, (int, np.integer)) else v 
            for k, v in features.items()}
    
    return features

def compute_effect_sizes(df_agg: pd.DataFrame, signals: list, subject_col: str = 'subject') -> pd.DataFrame:
    results = []
    
    for signal in signals:
        df_clean = df_agg[[subject_col, 'salle', signal]].dropna()
        
        if df_clean[subject_col].nunique() < 5:
            continue
        
        try:
            aov = pg.rm_anova(
                data=df_clean,
                dv=signal,
                within='salle',
                subject=subject_col
            )
            results.append({
                'signal':  signal,
                'F':       round(aov['F'].values[0], 3),
                'p_value': round(aov['p-unc'].values[0], 4),
                'eta2_p':  round(aov['np2'].values[0], 3),
                'effet': (
                    'grand'  if aov['np2'].values[0] > 0.14 else
                    'moyen'  if aov['np2'].values[0] > 0.06 else
                    'petit'
                )
            })
        except Exception as e:
            # On garde l'erreur pour débugger, mais séparément
            results.append({'signal': signal, 'F': np.nan, 'p_value': np.nan, 'eta2_p': np.nan, 'effet': f'erreur: {e}'})
    
    df_results = pd.DataFrame(results)
    
    # Sort seulement si la colonne existe et a des valeurs non-NaN
    if 'eta2_p' in df_results.columns and df_results['eta2_p'].notna().any():
        df_results = df_results.sort_values('eta2_p', ascending=False)
    
    return df_results


def run_lda_profile(df_agg: pd.DataFrame, features: list, subject_col: str = 'subject'):
    """
    Deux choses en une :
    
    A) LDA : trouve les axes qui séparent le mieux les salles
       → explained_variance_ratio_ : quelle part de la séparation chaque axe capture
       → coef_ : quel signal contribue le plus à chaque axe
    
    B) Profil moyen par salle : moyenne de chaque signal par salle
       → radar plot pour visualiser la "signature physiologique" de chaque salle
    
    Pourquoi StandardScaler avant LDA ?
    Les signaux ont des unités différentes (µS, bpm, ms).
    Sans normalisation, les signaux avec grande amplitude dominent artificiellement.
    """
    df_clean = df_agg[features + ['salle', subject_col]].dropna()
    
    X = df_clean[features].values
    y = df_clean['salle'].values
    
    # Normalisation (LDA est sensible aux échelles)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # LDA
    lda = LinearDiscriminantAnalysis()
    X_lda = lda.fit_transform(X_scaled, y)
    
    # Variance expliquée par axe discriminant
    var_ratio = pd.Series(
        lda.explained_variance_ratio_,
        index=[f'LD{i+1}' for i in range(len(lda.explained_variance_ratio_))]
    ).round(3)
    
    # Coefficients : contribution de chaque signal à chaque axe
    coef_df = pd.DataFrame(
        lda.coef_,
        columns=features,
        index=[f'Salle {c}' for c in lda.classes_]
    ).round(3)
    
    # Profil moyen par salle (pour radar plot)
    # On remet les données à l'échelle originale pour l'interprétabilité
    profile = df_clean.groupby('salle')[features].mean().round(3)
    
    # Projection des sujets dans l'espace LDA (pour scatter plot)
    df_proj = pd.DataFrame(X_lda, columns=[f'LD{i+1}' for i in range(X_lda.shape[1])])
    df_proj['salle']   = y
    df_proj['subject'] = df_clean[subject_col].values
    
    return {
        'variance_par_axe': var_ratio,   # pd.Series
        'coefficients':     coef_df,     # pd.DataFrame
        'profil_salle':     profile,     # pd.DataFrame — pour radar
        'projection':       df_proj      # pd.DataFrame — pour scatter LD1 vs LD2
    }

import matplotlib.pyplot as plt
import numpy as np

def plot_radar_salles(profil: pd.DataFrame):
    """
    profil : sortie de lda_results['profil_salle']
    lignes = salles, colonnes = features
    
    Chaque salle devient un polygone sur le radar.
    La forme du polygone = signature physiologique de la salle.
    """
    features  = list(profil.columns)
    n_features = len(features)
    salles     = profil.index.tolist()

    # Angles régulièrement espacés sur le cercle
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # fermer le polygone

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for salle in salles:
        values = profil.loc[salle].tolist()
        values += values[:1]  # fermer le polygone
        ax.plot(angles, values, label=f'Salle {salle}', linewidth=2)
        ax.fill(angles, values, alpha=0.1)

    # Labels des axes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, size=9)
    ax.set_title("Profil physiologique moyen par salle", pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def run_normality_checks_per_subject(subjects_data: dict) -> pd.DataFrame:
    """
    Pour chaque sujet, teste la normalité sur la distribution
    complète du signal (toutes les lignes à 30Hz).
    
    C'est beaucoup plus pertinent : on a plusieurs milliers de points
    par sujet, Shapiro est donc vraiment informatif.
    """
    signals = {
        "EDA RAW":      "eda_uS",
        "EDA filtered": "eda_uS_filtered",
        "EDA (zscore)": "eda_uS_filtered_zscore",
        "HR filtered":  "hr_bpm_filtered",
        "HR (zscore)":  "hr_bpm_filtered_zscore",
        "HRV (RMSSD)":  "hrv_rmssd"
    }
    
    rows = []
    
    for subject,df in subjects_data.items():        
        for label, col in signals.items():
            if col not in df.columns:
                continue
            
            data = df[col].dropna()
            
            if len(data) < 3:
                continue

            if len(data) > 500:
                data = data.sample(500, random_state=42)

            stat, p_val = shapiro(data)
            
            rows.append({
                "Sujet":     subject,
                "Signal":    label,
                "N salles":  len(data),
                "Shapiro W": round(stat, 4),
                "p-value":   round(p_val, 4),
                "Normal ?":  "✅" if p_val > 0.05 else "❌",
                "Skewness":  round(skew(data), 3),
                "Kurtosis":  round(kurtosis(data), 3),
            })
    return pd.DataFrame(rows)

def low_pass_filter(data, cutoff=0.1, fs=30, order=4):  
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    if len(data) < 3 * max(len(a), len(b)):
        print(f"Attention : données trop courtes ({len(data)} points), filtrage ignoré")
        return data
    
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def compute_hrv_rmssd_from_ibi(ibi_series, window_beats=10):
    """
    RMSSD correct : on travaille sur les vrais battements, pas sur 30 Hz.
    
    Pourquoi window_beats et pas window_sec ?
    Parce que RMSSD est défini sur N intervalles RR consécutifs,
    pas sur une durée fixe. 10-20 battements est standard pour
    des fenêtres courtes (Laborde et al., 2017).
    """
    # Garde seulement les lignes où IBI change → nouveau battement détecté
    # -> on garde seulement un point par battement.
    ibi_beats = ibi_series[ibi_series.diff() != 0].dropna()
    
    # On convertit en millisecondes (car les mesures de HRV sont traditionnellement exprimées en ms)
    rr_ms = ibi_beats * 1000.0          # secondes → ms

    # rr_ms   =  [800, 780, 820]
    # diff_rr =  [NaN, -20, 40]
    diff_rr = rr_ms.diff()              # différences successives
    
    # RMSSD sur fenêtre glissante de N battements
    hrv = (
        (diff_rr ** 2)  #  le carré -> pour tout avoir en positif
        .rolling(window=window_beats, min_periods=window_beats // 2)  # RMSSD local sur 10 battements.
        .mean() ** 0.5 # moyenne et racine carrée -> RMSSD=N−11​∑i=1N−1​(RRi+1​−RRi​)2​
    )
    
    return hrv  # index = index original des lignes "nouveau battement"


def aggregate_subjects(
        subjects_data: dict, 
        dfParticipants: pd.DataFrame,
        mode: str = "full",      # "full" | "pre_sam"
        window_sec: float = 30.0
        ) -> pd.DataFrame:
    """
    mode="full"    → moyenne sur toute la durée de la salle (actuel)
    mode="pre_sam" → moyenne sur les N secondes avant SAM_Validated

    Construit un DataFrame agrégé : 1 ligne par (sujet × salle).
    
    subjects_data : dict { "PARTICIPAN1": df, "PARTICIPAN2": df, ... }
                    chaque df est le résultat de load_subject()
    dfParticipants : le dfNew du tab Participants (SEXE, VR, etc.)
    
    Retourne un DataFrame avec colonnes :
      subject, salle, sexe, eda_mean, eda_std, hr_mean, hr_std, hrv_mean, ...
    """
    rows = []
    
    signals = [
        "eda_uS",                      
        "eda_uS_filtered",               
        "eda_uS_filtered_zscore",
        "eda_phasic",              
        "eda_phasic_zscore",      
        "eda_tonic",               
        "eda_tonic_zscore",        
        "eda_driver",              
        "eda_driver_zscore",
        #"scr_rate",
        #"scr_auc",      
        #"sdnn",
        #"pnn50",
        #"rmssd2",
        "hr_bpm_filtered",               
        "hr_bpm_filtered_zscore",
        "hrv_rmssd",
        "hrv_rmssd_zscore",
    ]
    
    for subject, df in subjects_data.items():
        # Récupère le sexe depuis dfParticipants
        sexe = dfParticipants.loc[subject, "SEXE"] if subject in dfParticipants.index else None
        
        # Ignore les sujets sans sexe défini (les "X" = non-binaire/inconnu)
        # Tu peux changer ce comportement selon tes besoins
        
        for salle in sorted(df["ev_salle"].dropna().unique()):
            #mask = df["ev_salle"] == salle
            #df_salle = df[mask]
            df_salle = df[df["ev_salle"] == salle]

            
            if mode == "pre_sam":
                # Cherche le SAM_Validated dans cette salle
                sam_rows = df_salle[df_salle["c3d_event"] == "SAM_Validated"]
                
                if sam_rows.empty:
                    continue  # pas de SAM dans cette salle → on skip
                
                sam_t = df_salle.loc[sam_rows.index[0], "timestamp"]
                
                # Fenêtre : [sam_t - window_sec, sam_t]
                mask_window = (
                    (df_salle["timestamp"] >= sam_t - window_sec) &
                    (df_salle["timestamp"] <  sam_t)
                )
                df_window = df_salle[mask_window]
                
                if df_window.empty:
                    continue
                
                df_to_agg = df_window

            else:  # mode="full"
                df_to_agg = df_salle

            # Dans aggregate_subjects — bloc final de la boucle interne
            row = {"subject": subject, "salle": salle, "sexe": sexe}

            for sig in signals:
                if sig in df_salle.columns:
                    row[f"{sig}_mean"] = df_to_agg[sig].mean()
                    row[f"{sig}_std"]  = df_to_agg[sig].std()

            #row.update(extract_scr_features(df_salle["eda_phasic"]))
            #row.update(extract_hrv_features(df_salle["ibi_s"]))
            fs_salle = df_salle["fs_reel"].iloc[0] if "fs_reel" in df_salle.columns else 9.0
            row.update(extract_scr_features(df_salle["eda_phasic"], df_salle["eda_tonic"], fs=fs_salle))
            row.update(extract_hrv_features(df_salle["ibi_s"], df_salle["hr_bpm"]))
            row.update(extract_c3d_features(df_salle))  # ← df_salle, pas df
            row.update(compute_room_entry_response(df_salle))  # réaction aux 3s après l'entrée

            rows.append(row)  # une seule fois
    
    return pd.DataFrame(rows)


def run_anova_repeated(df_agg: pd.DataFrame, signal_col: str) -> dict:
    """
    ANOVA à mesures répétées : chaque sujet passe par toutes les salles.
    C'est le bon test ici — pas une ANOVA classique — parce que
    le même sujet apparaît 5 fois (une par salle). Les mesures sont
    donc dépendantes, pas indépendantes.

    ---
    F = variance entre les groupes / variance à l'intérieur des groupes
    variance entre groupes  =  est-ce que les salles ont des moyennes EDA différentes ?
    variance intra groupes  =  est-ce que les sujets varient beaucoup dans chaque salle ?
    F élevé  →  les salles se distinguent bien, les sujets sont cohérents entre eux
    F proche de 1  →  la variation entre salles est du même ordre que le bruit intra-salle
    ---

    Utilise AnovaRM de statsmodels.
    
    Retourne un dict avec les résultats clés.
    """
    # AnovaRM a besoin de données complètes (pas de NaN, toutes les salles)
    df_clean = df_agg[["subject", "salle", signal_col]].dropna()
    
    # Garde seulement les sujets complets sur toutes les salles présentes dans df_agg
    # (nombre dynamique : peut être 4 si la salle 1 a été exclue en amont)
    n_salles = df_clean["salle"].nunique()
    counts = df_clean.groupby("subject")["salle"].count()
    complete_subjects = counts[counts == n_salles].index
    df_clean = df_clean[df_clean["subject"].isin(complete_subjects)]
    
    if len(complete_subjects) < 5:
        return {"error": f"Pas assez de sujets complets ({len(complete_subjects)})"}

    # Garde-fou : un signal quasi-constant (variance ~0, ex: une feature qui vaut
    # presque toujours 0) fait planter le calcul de F en interne (division par une
    # erreur résiduelle ~0) → F explose artificiellement, donnant un p et un η²p
    # qui ont l'air "parfaits" mais qui sont un artefact numérique, pas un effet réel.
    if df_clean[signal_col].std() < 1e-9:
        return {"error": "Signal quasi-constant (variance ~0) — test non fiable, ignoré"}

    try:
        aovrm = AnovaRM(
            data=df_clean,
            depvar=signal_col,      # variable dépendante
            subject="subject",      # identifiant sujet
            within=["salle"]        # facteur intra-sujet
        )
        result = aovrm.fit()
        table = result.anova_table

        f_value = table["F Value"].values[0]
        if not np.isfinite(f_value):
            return {"error": "F non-fini (variance résiduelle ~0) — test non fiable, ignoré"}

        return {
            "F": f_value,
            "p": table["Pr > F"].values[0],
            "df_num": table["Num DF"].values[0],
            "df_den": table["Den DF"].values[0],
            "n_subjects": len(complete_subjects),
            "table": table
        }
    except Exception as e:
        return {"error": str(e)}


def run_anova_sexe(df_agg: pd.DataFrame, signal_col: str, salle: int = None) -> dict:
    """
    ANOVA entre-sujets : effet du sexe (H vs F) sur un signal.
    
    Si salle est spécifié, filtre sur cette salle.
    Sinon, utilise la moyenne toutes salles confondues.
    
    On exclut les sexe="X" pour rester sur H/F comparables.
    """
    df = df_agg[df_agg["sexe"].isin(["H", "F"])].copy()
    
    if salle is not None:
        df = df[df["salle"] == salle]
    else:
        # Moyenne par sujet toutes salles
        df = df.groupby(["subject", "sexe"])[signal_col].mean().reset_index()
    
    df_clean = df[["sexe", signal_col]].dropna()
    
    group_H = df_clean[df_clean["sexe"] == "H"][signal_col]
    group_F = df_clean[df_clean["sexe"] == "F"][signal_col]
    
    if len(group_H) < 2 or len(group_F) < 2:
        return {"error": "Pas assez de données"}
    
    # Test de Levene d'abord : est-ce que les variances sont homogènes ?
    # (condition de l'ANOVA classique)
    levene_stat, levene_p = stats.levene(group_H, group_F)
    
    # Si variances homogènes → ANOVA (= t-test ici car 2 groupes)
    # Sinon → Welch t-test (plus robuste)
    if levene_p > 0.05:
        f_stat, p_val = stats.f_oneway(group_H, group_F)
        test_used = "ANOVA (variances égales)"
    else:
        t_stat, p_val = stats.ttest_ind(group_H, group_F, equal_var=False)
        f_stat = t_stat ** 2  # F = t² pour 2 groupes
        test_used = "Welch t-test (variances inégales)"
    
    return {
        "F": f_stat,
        "p": p_val,
        "n_H": len(group_H),
        "n_F": len(group_F),
        "mean_H": group_H.mean(),
        "mean_F": group_F.mean(),
        "levene_p": levene_p,
        "test_used": test_used
    }




def run_manova_pca(df_agg, signals, n_components=5):
    df_clean = df_agg[signals + ["salle"]].dropna()
    
    X = df_clean[signals].values
    X_scaled = StandardScaler().fit_transform(X)
    
    # Réduction en n_components dimensions
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    variance_expliquee = pca.explained_variance_ratio_.cumsum()[-1]
    st.info(f"PCA : {n_components} composantes expliquent {variance_expliquee:.1%} de la variance")
    
    # DataFrame avec les composantes comme variables
    cols_pca = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=cols_pca)
    df_pca["salle"] = df_clean["salle"].values
    
    # MANOVA sur les composantes
    return run_manova(df_pca, cols_pca)


def run_manova(df_agg: pd.DataFrame, signals) -> dict:
    """
    MANOVA : effet de la salle sur ... simultanément.

    Rappel1: 
    Intercept
    C'est la question : "Est-ce que les moyennes globales sont différentes de zéro ?"
    Autrement dit : tes signaux (EDA, HR, HRV) sont-ils globalement non-nuls sur l'ensemble de l'expérience ?
    Avec des z-scores normalisés par rapport à la baseline → les moyennes devraient être proches de 0. Donc l'intercept sera probablement non-significatif. 
    Tu t'en fous — ce n'est pas ce qui t'intéresse.

    Rappel2: C(salle)
    C'est la question qui t'intéresse : 
    "Est-ce que les signaux varient significativement selon la salle ?"
    Le C() indique que salle est une variable catégorielle (1, 2, 3, 4, 5 sont des labels, pas des nombres ordonnés). 
    Statsmodels compare alors chaque salle contre une référence (salle 1 par défaut).

    Explication résultat avec "eda_uS_filtered_zscore_mean" et "hrv_rmssd_mean",
       
    Intercept significatif (p=0) → tes signaux sont globalement différents de zéro sur l'expérience. 
    Normal — tes sujets ont réagi physiologiquement.
    C(salle) non significatif (p=0.99) → la salle n'a pas d'effet significatif sur la combinaison EDA + HR + HRV.
    Ça peut vouloir dire plusieurs choses :

    Les ambiances lumineuses n'affectent pas la physiologie mesurée
    Ton n=47 est insuffisant pour détecter un effet (manque de puissance)
    Les signaux sont trop bruités / mal normalisés
    L'effet existe mais sur un seul signal → les ANOVA individuelles par signal seront plus informatives que la MANOVA




    Utilise la moyenne par salle × sujet.
    Formule statsmodels : "EDA + HR + HRV ~ C(salle)"
    C() indique que salle est une variable catégorielle.
    """
    df_clean = df_agg[["salle"] + signals].dropna()
    
    if len(df_clean) < 10:
        return {"error": "Pas assez de données"}
    
    formula = " + ".join(signals) + " ~ C(salle)"
    
    try:
        manova = MANOVA.from_formula(formula, data=df_clean)
        result = manova.mv_test()
        rows = []

        for effect_name, effect_data in result.results.items():
            stat_table = effect_data["stat"]
            for criterion in stat_table.index:
                row = stat_table.loc[criterion]
                rows.append({
                    "Effet":       effect_name,
                    "Critère":     criterion,
                    "Valeur":      round(row["Value"], 4),
                    "F":           round(row["F Value"], 4),
                    "df num":      round(row["Num DF"], 1),
                    "df den":      round(row["Den DF"], 1),
                    "p-value":     round(row["Pr > F"], 4),
                    "Significatif": "✅" if row["Pr > F"] < 0.05 else "❌"
                })
        
        return {"rows": rows, "raw": result, "signals": signals}
    except Exception as e:
        return {"error": str(e)}
    
def run_friedman(df_agg: pd.DataFrame, signals: dict) -> pd.DataFrame:
    """
    Friedman : alternative non-paramétrique à l'ANOVA RM.
    
    Pour chaque signal, on passe les 5 groupes (un par salle)
    à friedmanchisquare() — il attend N vecteurs, un par condition.

    χ² élevé  →  certaines salles ont systématiquement des rangs plus hauts
    χ² ~0     →  les rangs sont répartis au hasard entre les salles

    Mes résultats:
    EDA (toutes versions) — p < 0.05 ✅ 
    La salle a un effet significatif sur l'EDA. Les ambiances lumineuses différencient bien les réponses électrodermales.
    
    EDA tonic — p=0.0035 c'est le meilleur résultat.
    Le niveau d'arousal général (composante lente) varie selon la salle — plus informatif que l'EDA brut.
    
    HR et HRV — non significatifs
    La salle n'affecte pas le rythme cardiaque ni la variabilité. 
    Deux interprétations possibles : 
        soit l'effet n'existe pas, 
        soit HR/HRV sont trop bruités pour le détecter avec n=42.
    """
    rows = []
    
    for label, col in signals.items():
        if col not in df_agg.columns:
            continue
        
        # Garde seulement les sujets complets sur toutes les salles présentes
        df_clean = df_agg[["subject", "salle", col]].dropna()
        n_salles = df_clean["salle"].nunique()
        counts = df_clean.groupby("subject")["salle"].count()
        complete_subjects = counts[counts == n_salles].index
        df_clean = df_clean[df_clean["subject"].isin(complete_subjects)]
        
        if len(complete_subjects) < 5:
            rows.append({"Signal": label, "stat": "—", "p-value": "—",
                        "Significatif": "—", "Note": f"Seulement {len(complete_subjects)} sujets complets"})
            continue
        
        # Pour chaque salle → un vecteur de valeurs (une par sujet)
        # friedmanchisquare attend : groupe1, groupe2, groupe3, ...
        groups = [
            df_clean[df_clean["salle"] == salle][col].values
            for salle in sorted(df_clean["salle"].unique())
        ]
        
        stat, p_val = friedmanchisquare(*groups)
        
        rows.append({
            "Signal":       label,
            "N sujets":     len(complete_subjects),
            "χ²":           round(stat, 4),
            "p-value":      round(p_val, 4),
            "Significatif": "✅" if p_val < 0.05 else "❌",
        })

    return pd.DataFrame(rows)


def compare_two_salles(
    df_agg: pd.DataFrame,
    signal_col: str,
    salle_a,
    salle_b,
    subject_col: str = "subject",
) -> dict:
    """
    Compare la moyenne d'un signal entre deux salles précises (ex: salle 2 vs salle 3).

    Ce sont les mêmes sujets dans les deux salles (mesures répétées) → deux tests
    appariés calculés en parallèle, pas un t-test indépendant :
    - Paired t-test (paramétrique, suppose la normalité des différences appariées)
    - Wilcoxon signed-rank (non-paramétrique, sur les rangs des différences —
      robuste si la normalité n'est pas respectée, ce qui est fréquent ici)

    Retourne les moyennes des deux salles, et stat/p pour les deux tests, sur les
    sujets ayant une valeur valide dans les deux salles.
    """
    df = df_agg[[subject_col, "salle", signal_col]].dropna().copy()
    df["salle"] = df["salle"].astype(float)

    pivot = df.pivot_table(index=subject_col, columns="salle", values=signal_col)
    if salle_a not in pivot.columns or salle_b not in pivot.columns:
        return {"error": f"Salle {salle_a} ou {salle_b} absente des données pour ce signal."}

    pivot = pivot[[salle_a, salle_b]].dropna()
    if len(pivot) < 5:
        return {"error": f"Pas assez de sujets complets sur les deux salles ({len(pivot)})"}

    a, b = pivot[salle_a].values, pivot[salle_b].values

    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = wilcoxon(a, b)
    except ValueError as e:
        return {"error": str(e)}

    return {
        "salle_a": salle_a, "salle_b": salle_b,
        "t_stat": float(t_stat), "t_p": float(t_p),
        "w_stat": float(w_stat), "w_p": float(w_p),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "n_subjects": len(pivot),
    }


def run_key_color_factorial(
    df_agg: pd.DataFrame,
    signal_col: str,
    salle_key: dict,
    salle_color: dict,
    subject_col: str = "subject",
    exclude_salle1: bool = True,
) -> dict:
    """
    Teste directement le design factoriel Q3 (key low/high × color red/blue)
    plutôt que salle par salle.

    Pour chaque sujet : moyenne du signal sur les salles "low" vs "high"
    (test apparié), et moyenne sur les salles "red" vs "blue" (test apparié).
    Wilcoxon signed-rank — non-paramétrique, adapté aux mesures répétées
    intra-sujet (chaque participant fournit sa propre paire de valeurs).

    salle_key / salle_color : dict {numéro de salle: "low"/"high" ou "red"/"blue"/"baseline"}.
    exclude_salle1 : la salle 1 (baseline, sans key/color) n'a pas de sens dans
    ce design 2×2 — exclue par défaut.
    """
    df = df_agg[[subject_col, "salle", signal_col]].dropna().copy()
    df["salle"] = df["salle"].astype(int)
    if exclude_salle1:
        df = df[df["salle"] != 1]

    df["key"] = df["salle"].map(salle_key)
    df["color"] = df["salle"].map(salle_color)

    results = {}

    for factor_name, factor_col, levels in [
        ("key", "key", ("low", "high")),
        ("color", "color", ("red", "blue")),
    ]:
        pivot = df.groupby([subject_col, factor_col])[signal_col].mean().unstack(factor_col)
        pivot = pivot.dropna(subset=list(levels))

        if len(pivot) < 5:
            results[factor_name] = {"error": f"Pas assez de sujets complets ({len(pivot)})"}
            continue

        a, b = pivot[levels[0]].values, pivot[levels[1]].values

        t_stat, t_p = stats.ttest_rel(a, b)
        try:
            stat, p_val = wilcoxon(a, b)
        except ValueError as e:
            results[factor_name] = {"error": str(e)}
            continue

        results[factor_name] = {
            "n_subjects": len(pivot),
            "levels": levels,
            f"mean_{levels[0]}": float(a.mean()),
            f"mean_{levels[1]}": float(b.mean()),
            "stat": float(stat),
            "p": float(p_val),
            "t_stat": float(t_stat),
            "t_p": float(t_p),
        }

    return results



def run_posthoc_dunn(df_agg: pd.DataFrame, signals: dict) -> dict:
    """
    Test post-hoc de Dunn avec correction de Bonferroni.
    
    Pourquoi Dunn ?
    C'est le post-hoc standard après Friedman — il travaille sur les rangs
    comme Friedman, donc cohérent. La correction de Bonferroni divise
    le seuil α par le nombre de comparaisons pour éviter les faux positifs.
    
    Avec 5 salles → 10 paires possibles (1-2, 1-3, 1-4, 1-5, 2-3, 2-4, 2-5, 3-4, 3-5, 4-5)
    → α ajusté = 0.05 / 10 = 0.005
    
    Retourne un dict { "EDA tonic": DataFrame 5×5 de p-values, ... }
    """
    
    results = {}
    
    for label, col in signals.items():
        if col not in df_agg.columns:
            continue
        
        df_clean = df_agg[["subject", "salle", col]].dropna()
        n_salles = df_clean["salle"].nunique()
        counts = df_clean.groupby("subject")["salle"].count()
        complete_subjects = counts[counts == n_salles].index
        df_clean = df_clean[df_clean["subject"].isin(complete_subjects)]
        
        if len(complete_subjects) < 5:
            continue
        
        # posthoc_dunn attend : valeurs + groupes
        p_matrix = posthoc_dunn(
            df_clean,
            val_col=col,
            group_col="salle",
            p_adjust="bonferroni"  # correction Bonferroni
        )
        
        results[label] = p_matrix
    
    return results

def run_cvxeda(eda_series, fs=30):
    """
    Décompose l'EDA en composante phasique et tonique via cvxEDA.
    Retourne (phasic, tonic) comme Series pandas, ou (None, None) si échec.

    Variable. Nom.  Ce que c'est.
    r  -- phasic (SCR)          -- Réponses rapides — émotions, stimuli ponctuels
    p  -- sparse SMNA driver    -- Signal nerveux sympathique estimé
    t  -- tonic (SCL)           -- Niveau de base lent — arousal général
    l  -- log (tonic)           -- Version log du tonic
    d  -- offset                -- Décalage DC
    e  -- résidus               -- Bruit non expliqué

    eda_phasic  →  réactivité émotionnelle ponctuelle (pic = stimulus)
        est-ce qu'il y a plus de pics de réactivité dans certaines salles ? (ambiances qui "surprennent")
    eda_tonic   →  niveau d'arousal général sur la durée
        est-ce que le niveau d'arousal général monte ou descend selon l'ambiance lumineuse ?
    """
    yn = eda_series.values.copy().astype(float)
    
    # Nettoyage NaN / Inf
    if np.isnan(yn).any():
        mask = np.isnan(yn)
        yn[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), yn[~mask])
    if np.isinf(yn).any():
        yn = np.nan_to_num(yn, nan=0.0, posinf=yn[~np.isinf(yn)].max(), neginf=yn[~np.isinf(yn)].min())
    if yn.min() < 0:
        yn = yn - yn.min() + 1e-6
    if yn.std() < 1e-6 or len(yn) < 100:
        return None, None
    
    # Standardisation (nécessaire pour cvxEDA)
    yn = (yn - yn.mean()) / yn.std()
    
    try:
        r, p, t, l, d, e, obj = cvxeda.cvxEDA(yn, 1.0 / fs)
        return (
            pd.Series(r, index=eda_series.index), 
            pd.Series(t, index=eda_series.index),
            pd.Series(p, index=eda_series.index),  
        )
    except Exception as ex:
        print(f"cvxEDA a échoué : {ex}")
        return None, None
    
def heatmap_participant_salle(
    df_agg: pd.DataFrame,
    variable: str = 'valence',
    titre: str = None,
    selected_subjects: list = None,  # ← nouveau paramètre optionnel
    subject_col: str = 'subject'
):
    # Filtrer les participants si une sélection est passée
    # Même logique que dans l'onglet Stats : on travaille sur un sous-ensemble
    if selected_subjects is not None:
        # On raccourcit les noms AVANT de filtrer,
        # parce que selected_subjects contient les noms complets ("PARTICIPAN3")
        df_agg = df_agg[df_agg[subject_col].isin(selected_subjects)]

    pivot = df_agg.pivot_table(
        index=subject_col,   
        columns='salle',
        values=variable,
        aggfunc='mean'
    )


    pivot.index = pivot.index.str.replace('PARTICIPAN', 'P')

    fig, ax = plt.subplots(figsize=(8, 10))

    sns.heatmap(
        pivot,
        ax=ax,
        cmap='RdYlGn',
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        vmin=0, vmax=4,
        cbar_kws={'label': variable}
    )

    titre = titre or f"Heatmap {variable} — Participant × Salle"
    ax.set_title(titre, fontsize=13, pad=15)
    ax.set_xlabel('Salle')
    ax.set_ylabel('Participant')
    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    # --- Résumé chiffré Variance ---
    #st.markdown("**Variance par salle** — est-ce que la salle crée des différences entre participants ?")
    #st.dataframe(pivot.var(axis=0).round(3).rename("variance").to_frame())

    #st.markdown("**Variance par participant** — est-ce que ce participant réagit différemment selon les salles ?")
    #st.dataframe(
    #    pivot.var(axis=1).round(3)
    #    .rename("variance")
    #    .to_frame()
    #    .sort_values("variance", ascending=False)
    #)

    # ── Résumé chiffré ──────────────────────────────────────────────

    # Moyenne par salle (axis=0 → on agrège les lignes = participants)
    mean_salle = pivot.mean(axis=0).round(3).rename("moyenne")
    var_salle  = pivot.var(axis=0).round(3).rename("variance")

    # Moyenne par participant (axis=1 → on agrège les colonnes = salles)
    mean_subj = pivot.mean(axis=1).round(3).rename("moyenne")
    var_subj  = pivot.var(axis=1).round(3).rename("variance")

    st.markdown("**Par salle** — tendance centrale + dispersion inter-participants")
    # pd.concat(..., axis=1) assemble deux Series en DataFrame côte à côte
    # axis=1 = "aligner sur les index, empiler en colonnes"
    st.dataframe(pd.concat([mean_salle, var_salle], axis=1))

    st.markdown("**Par participant** — tendance centrale + dispersion inter-salles")
    st.dataframe(
        pd.concat([mean_subj, var_subj], axis=1)
        .sort_values("variance", ascending=False)  # les plus "sensibles" en premier
    )

    # --- Résumé chiffré Moyenne ---


def agreger_par_salle(df, feature_cols):
    agg = (df.groupby(['participant', 'salle'])[feature_cols + ['valence', 'arousal']]
             .mean()
             .reset_index())
    # Colonne combinée pour l'annotation
    agg['participant_salle'] = agg['participant'] + '_' + agg['salle']
    return agg



def get_feat_all(df_agg):
    """Garde uniquement les features de mouvement (head, pitch, yaw, jerk, speed, etc.)"""
    feat_cols = []
    # Colonnes à exclure (métadonnées + physiologie)
    EXCLUDE = ["subject", "salle", "sexe", "valence", "arousal"]
    PHYSIO_PREFIXES = ["bvp_", "temp_", "ibi_", "scr_", "sdnn", "rmssd", "pnn50"]

    for col in df_agg.columns:
        if col in EXCLUDE:
            continue
        if any(col.startswith(p) for p in PHYSIO_PREFIXES):
            continue
        feat_cols.append(col)
    return feat_cols


def suggest_dbscan_eps(X_pca: np.ndarray, min_samples: int) -> dict:
    """
    Suggère une valeur d'eps pour DBSCAN à partir de la distance au k-ième plus
    proche voisin (k=min_samples), méthode standard (Ester et al. 1996 / "k-distance plot").

    Pourquoi c'est nécessaire ?
    DBSCAN est extrêmement sensible à l'échelle de X — sur des données PCA avec
    peu de points (~40-200), un eps "par défaut" comme 0.5 peut être bien trop
    petit et classer 100% des points en outliers (-1), ce qui ressemble à tort
    à "pas de structure" alors que c'est juste un mauvais réglage.

    Retourne {"eps_p50": médiane, "eps_p90": 90e percentile} des distances au
    min_samples-ième voisin — p50 donne un point de départ raisonnable, p90 un
    eps plus permissif si p50 isole encore trop de points.
    """
    n_neighbors = min(min_samples + 1, len(X_pca) - 1)  # +1 car le point lui-même compte
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X_pca)
    distances, _ = nn.kneighbors(X_pca)
    k_distances = distances[:, -1]  # distance au n_neighbors-ième voisin, pour chaque point
    return {
        "eps_p50": float(np.median(k_distances)),
        "eps_p90": float(np.percentile(k_distances, 90)),
    }


    # ══════════════════════════════════════════════════════════════════════════
    # FONCTION UTILITAIRE — pipeline commun (imputation → PCA → UMAP → clustering)
    # ══════════════════════════════════════════════════════════════════════════
    # On factorise ici pour ne pas répéter le même code dans les 2 sections.
    # X_raw : array numpy brut (avant imputation)
    # feat_cols : liste des colonnes features (pour PCA)
    # Retourne : X_pca, X_umap, labels par méthode

def run_clustering_pipeline(X_raw, df_meta, k, eps, min_samples, mode_residus=True):
        """
        Prend une matrice de features brutes (avec NaN possibles),
        applique imputation → normalisation → PCA → clustering × 3.
        Retourne X_pca, X_umap, et un dict de labels par méthode.
        """
        if mode_residus:
            # Pour chaque sujet, on soustrait sa moyenne sur toutes ses salles
            # → ce qui reste = déviation par rapport au comportement habituel du sujet
            df_tmp = pd.DataFrame(X_raw, columns=[f"f{i}" for i in range(X_raw.shape[1])])
            df_tmp["subject"] = df_meta["subject"].values
            moyennes = df_tmp.groupby("subject").transform("mean")
            X_raw = (df_tmp.drop(columns="subject") - moyennes).values

        X = SimpleImputer(strategy="median").fit_transform(X_raw)
        X_scaled = StandardScaler().fit_transform(X)

        pca = PCA(n_components=0.95, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        reducer = umap.UMAP(n_neighbors=15, 
                            min_dist=0.1,
                             metric='euclidean',
                            random_state=42)
        X_umap = reducer.fit_transform(X_pca)

        labels = {}

        labels["K-Means"] = KMeans(
            n_clusters=k, random_state=42, n_init=10
        ).fit_predict(X_pca)

        Z = linkage(pdist(X_pca, metric="euclidean"), method="ward")
        labels["Hiérarchique"] = fcluster(Z, t=k, criterion="maxclust")

        # Variante : clustering directement sur les coordonnées UMAP 2D plutôt
        # que sur l'espace PCA. UMAP "pousse" les points dans des paquets bien
        # séparés en 2D même quand la séparation réelle en haute dimension est
        # faible — ça peut révéler une structure locale que PCA (linéaire) rate,
        # mais c'est aussi plus circulaire/sensible aux hyperparamètres UMAP
        # (n_neighbors, min_dist, random_state). À valider avec
        # compare_clustering_to_known_groupings avant de la considérer comme
        # une vraie preuve, pas seulement un artefact de mise en page.
        Z_umap = linkage(X_umap, method="ward")
        labels["Hiérarchique (UMAP)"] = fcluster(Z_umap, t=k, criterion="maxclust")

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_pca)
        labels["DBSCAN"] = db.labels_

        return X_pca, X_umap, labels

def afficher_resultats(df_plot, X_pca, X_umap, labels_dict, id_col):
        """
        Affiche silhouettes + UMAP + crosstabs pour les 3 méthodes.
        df_plot doit avoir une colonne 'salle' et id_col (ex: 'subject' ou 'fenetre_id').
        """
        df_plot = df_plot.copy()
        df_plot["umap1"] = X_umap[:, 0]
        df_plot["umap2"] = X_umap[:, 1]

        def safe_silhouette(X, labels):
            unique = set(labels)
            unique.discard(-1)
            if len(unique) < 2:
                return None
            mask = np.array(labels) != -1
            if mask.sum() < 2:
                return None
            return silhouette_score(X[mask], np.array(labels)[mask])

        # ── Silhouettes ───────────────────────────────────────────────────────
        cols = st.columns(len(labels_dict))
        for col, (method, labels) in zip(cols, labels_dict.items()):
            sil = safe_silhouette(X_pca, labels)
            n_outliers = (np.array(labels) == -1).sum()
            col.metric(
                f"Silhouette {method}",
                f"{sil:.3f}" if sil else "N/A",
                f"{n_outliers} outliers" if n_outliers > 0 else None
            )

        # ── UMAP ─────────────────────────────────────────────────────────────
        for method, labels in labels_dict.items():
            st.write(f"**{method}**")
            df_plot[f"cluster_{method}"] = labels

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 5))
                scatter = ax.scatter(
                    df_plot["umap1"], df_plot["umap2"],
                    c=df_plot["salle"], cmap="tab10", s=60, alpha=0.8
                )
                plt.colorbar(scatter, ax=ax, label="Salle")
                ax.set_title(f"{method} — couleur salle")
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots(figsize=(6, 5))
                scatter = ax.scatter(
                    df_plot["umap1"], df_plot["umap2"],
                    c=df_plot[f"cluster_{method}"], cmap="viridis", s=60, alpha=0.8
                )
                plt.colorbar(scatter, ax=ax, label="Cluster")
                ax.set_title(f"{method} — couleur cluster")
                st.pyplot(fig)

        # ── Crosstabs ─────────────────────────────────────────────────────────
        st.write("**Distribution par salle**")
        cols = st.columns(len(labels_dict))
        for col, (method, labels) in zip(cols, labels_dict.items()):
            df_plot[f"cluster_{method}"] = labels
            col.write(method)
            col.dataframe(pd.crosstab(df_plot["salle"], df_plot[f"cluster_{method}"], margins=True))


def compare_clustering_to_known_groupings(X_pca, ks, sil_km, sil_hc, groupings: dict):
    """
    Compare le clustering non supervisé à des regroupements connus (salle, type
    d'éclairage, color grading...) plutôt que de chercher à "retomber" sur 5 salles
    par hasard.

    Affiche pour chaque regroupement connu :
    - sa silhouette "si on force ces labels" (sert de référence)
    - la meilleure silhouette non supervisée trouvée (K-Means / Hiérarchique, tous k)
    - l'ARI (Adjusted Rand Index) entre le clustering effectif (k=len(groupe)) et le regroupement connu

    groupings : dict { "Salle (5 groupes)": pd.Series ou array de labels connus, ... }
    """
    best_k_idx = int(np.argmax(sil_km))
    best_sil_km = sil_km[best_k_idx]
    best_k_km = list(ks)[best_k_idx]

    best_k_idx_hc = int(np.argmax(sil_hc))
    best_sil_hc = sil_hc[best_k_idx_hc]
    best_k_hc = list(ks)[best_k_idx_hc]

    rows = []
    for name, known_labels in groupings.items():
        known_labels = np.asarray(known_labels)
        n_groups = len(set(known_labels))

        if n_groups < 2:
            continue

        sil_known = silhouette_score(X_pca, known_labels)

        # Clustering non supervisé au même nombre de groupes, pour comparer à ARI équitable
        km_same_k = KMeans(n_clusters=n_groups, random_state=42, n_init=10).fit_predict(X_pca)
        ari_km = adjusted_rand_score(known_labels, km_same_k)

        rows.append({
            "Regroupement connu":            name,
            "N groupes":                     n_groups,
            "Silhouette (labels connus)":     round(sil_known, 3),
            f"Meilleure silhouette K-Means (k={best_k_km})":      round(best_sil_km, 3),
            f"Meilleure silhouette Hiérarchique (k={best_k_hc})": round(best_sil_hc, 3),
            "ARI vs K-Means (même k)":        round(ari_km, 3),
        })

    df_compare = pd.DataFrame(rows)
    st.dataframe(df_compare, width="stretch")

    st.caption(
        "Lecture : si la silhouette \"labels connus\" est nettement plus basse que la "
        "meilleure silhouette non supervisée, le regroupement connu n'organise pas bien "
        "l'espace des features — le clustering trouve une autre structure, pas un échec "
        "de pipeline. Un ARI proche de 0 = pas d'accord entre les deux partitions ; "
        "proche de 1 = le clustering retrouve quasiment le regroupement connu."
    )

    return df_compare


def run_xgboost_importance(
    df_agg: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    subject_col: str = "subject",
    n_splits: int = 5,
) -> dict:
    """
    Entraîne un XGBoost pour prédire `target_col` (ex: 'salle', 'key', 'color') à
    partir de `feature_cols`, et retourne les features les plus importantes.

    Pourquoi GroupKFold (par sujet) et pas un split aléatoire classique ?
    → Comme pour la MANOVA, chaque sujet contribue plusieurs lignes (une par salle).
      Un split aléatoire classique pourrait mettre 3 salles d'un même sujet en train
      et sa 4e salle en test — le modèle "reconnaît" alors le sujet (sa baseline
      physio/mouvement individuelle) plutôt que d'apprendre un vrai effet de la cible.
      GroupKFold garantit qu'un sujet entier est soit en train, soit en test.

    Retourne l'accuracy moyenne en validation croisée (à comparer à la baseline =
    fréquence de la classe majoritaire) et les importances de features, calculées
    sur un modèle final entraîné sur toutes les données (à but interprétatif, pas
    pour évaluer la performance — c'est l'accuracy CV qui sert à ça).
    """
    if not XGBOOST_AVAILABLE:
        return {"error": "xgboost n'est pas installé (ajoute `xgboost` à requirements.txt et `pip install xgboost`)."}

    df_clean = df_agg[[subject_col, target_col] + feature_cols].dropna()
    if df_clean.empty:
        return {"error": "Aucune ligne sans NaN sur ces colonnes."}

    X = df_clean[feature_cols].values
    classes, y = np.unique(df_clean[target_col].values, return_inverse=True)
    groups = df_clean[subject_col].values

    if len(classes) < 2:
        return {"error": "Moins de 2 classes dans la cible — rien à prédire."}

    n_subjects = df_clean[subject_col].nunique()
    n_splits_eff = max(2, min(n_splits, n_subjects))
    if n_subjects < 2:
        return {"error": "Pas assez de sujets distincts pour une validation croisée groupée."}

    gkf = GroupKFold(n_splits=n_splits_eff)
    accs = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            eval_metric="mlogloss", random_state=42, verbosity=0, n_jobs=1,
        )
        model.fit(X[train_idx], y[train_idx])
        accs.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))

    # Modèle final sur toutes les données — pour les importances uniquement.
    final_model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        eval_metric="mlogloss", random_state=42, verbosity=0, n_jobs=1,
    )
    final_model.fit(X, y)

    importances = pd.Series(
        final_model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    _, counts = np.unique(y, return_counts=True)
    baseline_accuracy = float(counts.max() / counts.sum())

    return {
        "cv_accuracy_mean": float(np.mean(accs)),
        "cv_accuracy_std":  float(np.std(accs)),
        "baseline_accuracy": baseline_accuracy,
        "n_splits": n_splits_eff,
        "n_classes": len(classes),
        "classes": classes.tolist(),
        "importances": importances,
    }


def afficher_resultats_participants(df_plot, X_pca, X_umap, labels_dict, group_cols):
    """
    Affiche silhouettes + UMAP + crosstabs pour les 3 méthodes de clustering,
    vue "1 point = 1 participant" (moyenne sur ses salles).

    Question posée : existe-t-il des profils/types de participants (indépendamment
    de la salle), par ex. liés au sexe ou à l'expérience VR ?

    df_plot doit contenir une colonne 'subject' + les colonnes listées dans group_cols
    (ex: 'SEXE', 'VR') pour les crosstabs.
    """
    df_plot = df_plot.copy()
    df_plot["umap1"] = X_umap[:, 0]
    df_plot["umap2"] = X_umap[:, 1]

    def safe_silhouette(X, labels):
        unique = set(labels)
        unique.discard(-1)
        if len(unique) < 2:
            return None
        mask = np.array(labels) != -1
        if mask.sum() < 2:
            return None
        return silhouette_score(X[mask], np.array(labels)[mask])

    cols = st.columns(len(labels_dict))
    for col, (method, labels) in zip(cols, labels_dict.items()):
        sil = safe_silhouette(X_pca, labels)
        n_outliers = (np.array(labels) == -1).sum()
        col.metric(
            f"Silhouette {method}",
            f"{sil:.3f}" if sil else "N/A",
            f"{n_outliers} outliers" if n_outliers > 0 else None
        )

    for method, labels in labels_dict.items():
        st.write(f"**{method}**")
        df_plot[f"cluster_{method}"] = labels

        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(
            df_plot["umap1"], df_plot["umap2"],
            c=df_plot[f"cluster_{method}"], cmap="viridis", s=80, alpha=0.8
        )
        plt.colorbar(scatter, ax=ax, label="Cluster")
        ax.set_title(f"{method} — profils participants")
        st.pyplot(fig)

        crosstab_cols = [c for c in group_cols if c in df_plot.columns]
        if crosstab_cols:
            cols_ct = st.columns(len(crosstab_cols))
            for col_ct, group_col in zip(cols_ct, crosstab_cols):
                col_ct.write(f"Distribution par {group_col}")
                col_ct.dataframe(pd.crosstab(df_plot[group_col], df_plot[f"cluster_{method}"], margins=True))

        # ── Liste des sujets par cluster ────────────────────────────────────
        # Utile pour repérer si un cluster minoritaire est en fait 1-2 outliers
        # isolés (silhouette artificiellement haute) plutôt qu'un vrai profil.
        st.write(f"Sujets par cluster ({method})")
        cluster_sizes = df_plot[f"cluster_{method}"].value_counts().sort_index()
        cluster_cols = st.columns(len(cluster_sizes))
        for col_cl, (cluster_id, size) in zip(cluster_cols, cluster_sizes.items()):
            label = "outliers (-1)" if cluster_id == -1 else f"cluster {cluster_id}"
            subjects_in_cluster = sorted(df_plot.loc[df_plot[f"cluster_{method}"] == cluster_id, "subject"])
            col_cl.write(f"**{label}** ({size})")
            col_cl.caption(", ".join(subjects_in_cluster))


def load_subject(filepath) -> pd.DataFrame:
    """
    Charge un CSV sujet et applique les types corrects.
    Le CSV n'a PAS de header → on force les noms de colonnes.

    filepath : str (chemin local) OU UploadedFile (Streamlit)
    pd.read_csv() accepte les deux sans modification.
    """
    col_names = [
        "heure_locale", "timestamp",
        "head_x", "head_y", "head_z",
        "head_qx", "head_qy", "head_qz", "head_qw",
        "HMD_Battery_Level", "HMD_Battery_Status",
        "eda_uS", "bvp", "temp_C", "steps", "ibi_s", "hr_bpm",
        "c3d_event",
        "ev_fichier", "ev_objet",
        "ev_Valence", "ev_Arousal", "ev_Room",
        "ev_salle", "ev_statut", "ev_sessionlength", "ev_Reason"
    ]

    df = pd.read_csv(
        filepath,
        header=None,          # pas de ligne de header dans le fichier
        names=col_names,
        dtype=str,            # tout en string d'abord pour gérer les NaN proprement
        on_bad_lines="skip"   # ignore les lignes tronquées
    )

    # ── Conversion types ──────────────────────────────────────────────────────
    numeric_cols = [
        "timestamp", "head_x", "head_y", "head_z",
        "head_qx", "head_qy", "head_qz", "head_qw",
        "HMD_Battery_Level", "eda_uS", "bvp", "temp_C",
        "steps", "ibi_s", "hr_bpm",
        "ev_Valence", "ev_Arousal", "ev_Room", "ev_salle", "ev_sessionlength"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Tri chronologique ─────────────────────────────────────────────────────
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Taux d'échantillonnage réel ───────────────────────────────────────────
    # Ne PAS supposer 30 Hz — le casque/Cognitive3D échantillonne en pratique
    # autour de 8-9 Hz (vérifié sur les timestamps bruts). Médiane des écarts
    # de timestamp = robuste aux quelques frames manquées/dupliquées.
    dt_median = df["timestamp"].diff().median()
    fs_reel = 1.0 / dt_median if dt_median and dt_median > 0 else 9.0
    df["fs_reel"] = fs_reel  # propagé pour les fonctions appelées plus tard (extract_scr_features...)

    # ── Propagation du numéro de salle ────────────────────────────────────────
    # ev_salle n'est rempli que sur les lignes "entree_nouvelle_room".
    # On propage vers le bas (ffill) pour couvrir tous les timestamps suivants.
    # On propage aussi vers le haut (bfill) pour couvrir la salle 1, qui n'a
    # pas d'événement d'entrée — le participant y commence directement.
    index_fin = df[df["c3d_event"] == "fin_experience"].index.min()
    if pd.notna(index_fin):
        df = df.loc[:index_fin - 1].reset_index(drop=True)

    # On sait que tout ce qui précède la première entree_nouvelle_room est salle 1
    first_room_event = df[df["c3d_event"] == "entree_nouvelle_room"].index.min()
    df.loc[:first_room_event - 1, "ev_salle"] = 1.0

    # Maintenant ffill propage correctement 1.0 puis 2.0 puis 3.0...
    df["ev_salle"] = df["ev_salle"].ffill()
    

    # ── Filtrage passe-bas ──────────────────────────────────────────────
    # On filtre sur les valeurs non-NaN uniquement, sinon filtfilt plante
    # L'idée : extraire les indices valides, filtrer, réinjecter

    for col, col_filtered in [("eda_uS", "eda_uS_filtered"), ("hr_bpm", "hr_bpm_filtered")]:
        valid_mask = df[col].notna()
        if valid_mask.sum() > 10:  # assez de points pour filtrer
            filtered_values = low_pass_filter(df.loc[valid_mask, col].values, fs=fs_reel)
            df[col_filtered] = np.nan
            df.loc[valid_mask, col_filtered] = filtered_values
        else:
            df[col_filtered] = df[col]  # fallback : copie brute

    # Méthode 1 : hrv_rmssd brut (valeurs aux battements, ffill pour aligner à 30Hz)
    df["hrv_rmssd"] = np.nan
    hrv = compute_hrv_rmssd_from_ibi(df["ibi_s"])
    df.loc[hrv.index, "hrv_rmssd"] = hrv.values
    df["hrv_rmssd"] = df["hrv_rmssd"].ffill()

    # Méthode 2 : log → zscore sur baseline 60s (temps réel, pas samples)
    df["hrv_rmssd_log"] = np.log(df["hrv_rmssd"] + 1e-6)

    t0 = df["timestamp"].iloc[0]
    baseline_mask = df["timestamp"] <= t0 + 60  # 60 secondes réelles

    baseline_hrv = df.loc[baseline_mask, "hrv_rmssd_log"].dropna()
    # dropna() : exclut les NaN des premières fenêtres incomplètes
    # (RMSSD nécessite min_periods battements avant d'être stable)

    baseline_hrv_mean = baseline_hrv.mean()
    baseline_hrv_std  = baseline_hrv.std()

    if baseline_hrv_std > 1e-6:
        df["hrv_rmssd_zscore"] = (df["hrv_rmssd_log"] - baseline_hrv_mean) / baseline_hrv_std
    else:
        df["hrv_rmssd_zscore"] = 0.0

    # ── Normalisation z-score par sujet ────────────────────────────────
    # Baseline = les 30 premières secondes (avant la salle 1)
    # Pourquoi 30s ? Fenêtre standard en psychophysio pour estimer la baseline
    baseline_samples = int(round(30 * fs_reel))  # 30 secondes × taux réel (pas 30 Hz en dur)
    for col in ["eda_uS_filtered", "hr_bpm_filtered", "hrv_rmssd"]:
        baseline = df[col].iloc[:baseline_samples]
        baseline_mean = baseline.mean()
        baseline_std  = baseline.std()
        if baseline_std > 1e-6:  # évite division par zéro
            df[col + "_zscore"] = (df[col] - baseline_mean) / baseline_std
        else:
            df[col + "_zscore"] = 0.0

    # ── cvxEDA ──────────────────────────────────────────────────────────
    # Long à calculer (~quelques secondes), on le fait une seule fois au chargement
    phasic, tonic, driver = run_cvxeda(df["eda_uS_filtered"], fs=fs_reel)   #.dropna() si jamais bug
    # dropna() car cvxEDA veut un signal continu sans trous
    if phasic is not None:
        df["eda_phasic"] = phasic
        df["eda_tonic"]  = tonic
        df["eda_driver"] = driver  # driver SMNA
        
        # Zscore sur les 3 composantes
        # Phasic et tonic sont déjà standardisés par cvxEDA (centré réduit)
        # mais on normalise quand même par baseline pour cohérence inter-sujets
        for col in ["eda_phasic", "eda_tonic", "eda_driver"]:
            baseline = df[col].iloc[:baseline_samples]
            b_mean, b_std = baseline.mean(), baseline.std()
            if b_std > 1e-6:
                df[col + "_zscore"] = (df[col] - b_mean) / b_std
            else:
                df[col + "_zscore"] = 0.0
    else:
        for col in ["eda_phasic", "eda_tonic", "eda_driver",
                    "eda_phasic_zscore", "eda_tonic_zscore", "eda_driver_zscore"]:
            df[col] = np.nan

    # ── Vitesse de déplacement (norme du vecteur différentiel de position) ────
    # On calcule le déplacement entre deux timestamps consécutifs,
    # divisé par le dt pour avoir des m/s.
    #
    # Pourquoi ? La position absolue (head_x, head_y, head_z) est moins
    # informative que la vitesse d'exploration. Un participant qui bouge
    # rapidement explore activement ; immobile = peut-être absorbé ou perdu.
    dx = df["head_x"].diff() # différence entre position actuelle et précédente
    dy = df["head_y"].diff()
    dz = df["head_z"].diff()
    dt = df["timestamp"].diff().replace(0, np.nan)  # évite division par zéro
    #dt = df["timestamp"].diff() # différence de temps

    #df["speed_mps"] = np.sqrt(dx**2 + dy**2 + dz**2) / dt

    # .diff() calcule simplement la différence entre chaque ligne et la précédente. 
    #  On obtient un déplacement en mètres, divisé par le temps écoulé → une vitesse en m/s. 
    #  C'est plus utile que la position absolue, 
    #  car ce qui nous intéresse c'est est-ce que le participant bouge ou est-il statique.

    # ── Timestamp relatif (secondes depuis début de session) ──────────────────
    df["t_rel"] = df["timestamp"] - df["timestamp"].iloc[0]

    # ── Orientation de tête (quaternions → Euler) ─────────────────────────────
# On convertit les quaternions en angles Euler pour avoir des signaux
# temporels interprétables (pitch = haut/bas, yaw = gauche/droite).
# On ne stocke PAS les quaternions bruts — ils sont dans un espace non-euclidien
# (diff() sur des quaternions n'a pas de sens géométrique).
#
# Pourquoi ici et pas dans extract_c3d_features ?
# → extract_c3d_features retourne des SCALAIRES (mean, std...) qui sont perdus
#   après agrégation. Ici on stocke les SÉRIES frame par frame dans df,
#   disponibles pour le foundation model plus tard.

    q_cols = ["head_qx", "head_qy", "head_qz", "head_qw"]
    q_valid = df[q_cols].dropna()

    df["pitch"] = np.nan
    df["roll"]  = np.nan
    df["yaw"]   = np.nan

    if len(q_valid) > 1:
        rots  = Rotation.from_quat(q_valid.values)
        euler = rots.as_euler("xyz", degrees=True)  # xyz → [pitch, roll, yaw]

        df.loc[q_valid.index, "pitch"] = euler[:, 0]
        df.loc[q_valid.index, "roll"]  = euler[:, 1]
        df.loc[q_valid.index, "yaw"]   = euler[:, 2]
    else:
        df["pitch"] = np.nan
        df["yaw"]   = np.nan

    return df



# Etape 7: On évalue avec des metrics
def evaluate_clustering(embeddings_matrix, Z, k_range=range(2, 8)):
    """
    Pour chaque k possible, calcule deux métriques :

    Silhouette score (-1 → 1) :
      Mesure si chaque point est bien dans son cluster.
      = (distance moyenne aux autres clusters - distance moyenne dans son cluster)
        / max des deux
      → 1 = parfait, 0 = chevauchement, -1 = mauvaise assignation
      ↑ MAXIMISER

    Davies-Bouldin score (0 → ∞) :
      Ratio entre la dispersion intra-cluster et la distance inter-cluster.
      → 0 = clusters parfaitement séparés et compacts
      ↓ MINIMISER
    """
    results = []
    for k in k_range:
        labels = fcluster(Z, t=k, criterion="maxclust")
        sil = silhouette_score(embeddings_matrix, labels, metric="cosine")
        db  = davies_bouldin_score(embeddings_matrix, labels)
        results.append({"k": k, "silhouette": sil, "davies_bouldin": db})
        print(f"k={k}  silhouette={sil:.3f}  davies_bouldin={db:.3f}")




    return pd.DataFrame(results)



# Etape six: clustering hierarchique
def hierarchical_clustering(
    embeddings_matrix: np.ndarray,
    keys: list,
    df_umap: pd.DataFrame,
    n_clusters: int = 4,
) -> pd.DataFrame:
    """
    Clustering hiérarchique agglomératif sur les embeddings MOMENT.

    Pourquoi hiérarchique plutôt que KMeans ?
    → KMeans suppose des clusters sphériques et de taille égale.
      Le clustering hiérarchique n'a aucune hypothèse sur la forme
      des clusters — il fusionne itérativement les points les plus
      proches jusqu'à obtenir n_clusters groupes.

    Pourquoi ward comme linkage ?
    → Ward minimise la variance intra-cluster à chaque fusion.
      C'est le critère le plus stable en pratique pour des données
      continues comme des embeddings.

    Distances cosine → on convertit en matrice condensée pour scipy.
    """

    # ── 1. Matrice de distances cosine ───────────────────────────────────────
    # pdist calcule toutes les distances paires → vecteur condensé (pas matrice carrée)
    # C'est le format attendu par linkage()
    dist_condensed = pdist(embeddings_matrix, metric="cosine")

    # ── 2. Linkage hiérarchique ───────────────────────────────────────────────
    # Z est la matrice de linkage : à chaque étape, quels clusters ont fusionné
    # et à quelle distance. C'est l'arbre complet (dendrogramme).
    Z = linkage(dist_condensed, method="ward")

    # ── 3. Dendrogramme — pour choisir n_clusters visuellement ───────────────

    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(
        Z,
        ax=ax,
        truncate_mode="lastp",  # affiche seulement les p dernières fusions
        p=30,                   # → plus lisible que les 203 feuilles
        leaf_rotation=90,
        color_threshold=0.7 * max(Z[:, 2]),  # colore les clusters principaux
    )
    ax.set_title("Dendrogramme — clustering hiérarchique sur embeddings MOMENT")
    ax.set_xlabel("Séries (participant × salle)")
    ax.set_ylabel("Distance cosine (Ward)")
    plt.tight_layout()
    #plt.savefig("png-full_analyse/dendrogram_moment.png", dpi=300)
    plt.show()

    # ── 4. Couper l'arbre à n_clusters ───────────────────────────────────────
    # fcluster coupe le dendrogramme pour obtenir exactement n_clusters groupes
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    # labels : array de shape (203,) avec valeurs 1..n_clusters

    # ── 5. Ajouter les labels au DataFrame UMAP ───────────────────────────────
    df_result = df_umap.copy()
    df_result["cluster"] = labels

    return df_result, Z, fig


# Cinquieme étape pour le fondation model: UMAP depuis embeddings entraine par MOMENT
def build_umap_from_embeddings(
    embeddings_matrix: np.ndarray,
    keys: list,
    n_neighbors: int = 15,
    min_dist: float = 0.1
) -> pd.DataFrame:
    """
    Réduit les embeddings (203, 1024) → (203, 2) via UMAP,
    puis construit un DataFrame exploitable dans ton app.

    Pourquoi UMAP sur les embeddings et pas directement PCA ?
    → UMAP préserve la structure locale non-linéaire — deux points
      proches dans l'espace 1024D restent proches en 2D.
      PCA ne préserve que la variance globale (structure linéaire).
    """
    reducer = UMAP(
        n_neighbors=n_neighbors,  # combien de voisins considérer pour la structure locale
        min_dist=min_dist,        # compacité des clusters (0.0 = très serrés, 1.0 = étalés)
        n_components=2,
        random_state=42,
        metric="cosine"           # cosine est standard pour les embeddings de transformers
                                  # (les directions comptent plus que les magnitudes)
    )

    coords = reducer.fit_transform(embeddings_matrix)
    # shape : (203, 2)

    # Reconstruire un DataFrame avec les métadonnées
    df_result = pd.DataFrame({
        "UMAP_1":  coords[:, 0],
        "UMAP_2":  coords[:, 1],
        "subject": [k[0] for k in keys],
        "salle":   [k[1] for k in keys],
    })

    return df_result


# Quatrième étape pour le fondation model: encode
def encode_with_moment(series_normalized: dict) -> tuple[np.ndarray, list]:
    """
    Encode chaque série (512, 5) en un vecteur d'embedding via MOMENT.

    Architecture de MOMENT :
    ────────────────────────
    MOMENT est un transformer entraîné sur ~1 milliard de points de séries
    temporelles. Il découpe chaque série en patches (comme un ViT pour les
    images), les encode via self-attention, puis produit un embedding global
    qui résume la série entière.

    Input  : (batch, n_signals, seq_len) = (1, 5, 512)
    Output : embedding de dimension 1024 par série

    Pourquoi 1024 ? C'est la dimension cachée du transformer MOMENT-large.
    Ce vecteur dense capture les patterns temporels que tes features manuelles
    (mean, std, spectral_centroid...) ne voient pas.
    """

    # ── Chargement du modèle ──────────────────────────────────────────────────
    # task_name="embedding" → on veut les représentations, pas des prédictions
    # On charge une seule fois (lourd ~400MB)
    print("Chargement de MOMENT...")
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "embedding"},
    )
    model.init()
    model.eval()  # mode inférence : désactive dropout etc.
    print("Modèle chargé.")

    embeddings = []
    keys       = []

    # Méthodes disponibles :
    #st.write([m for m in dir(model) if not m.startswith("_")])
    #import inspect
    #st.write(inspect.signature(model.embed))

    with torch.no_grad():  # pas de gradient → plus rapide, moins de mémoire
        for (subj, salle), arr in series_normalized.items():
            tensor = torch.tensor(arr.T, dtype=torch.float32).unsqueeze(0)

            # input_mask : 1 = frame valide, 0 = padding
            # Pour nous tout est valide (on a interpolé les NaN) → masque de 1 partout
            # Pour gérer les Nan proprement, 
            # AVEC MASK
            #mask = torch.ones(1, tensor.shape[2], dtype=torch.long)  # (1, 512)
            #output = model.embed(x_enc=tensor, input_mask=mask)


            output = model.embed(x_enc=tensor)
            # shape output.embeddings : (1, 1024)

            emb = output.embeddings.squeeze(0).numpy()
            embeddings.append(emb)
            keys.append((subj, salle))

    embeddings_matrix = np.array(embeddings)
    # shape finale : (203, 1024)
    
    st.write(f"Embeddings shape : {embeddings_matrix.shape}")
    return embeddings_matrix, keys



# Troisième étape pour le fondation model
def normalize_series(series_dict: dict) -> dict:
    """
    Normalise chaque signal de chaque série (mean=0, std=1).

    Pourquoi par série et pas globalement ?
    → Les foundation models sont entraînés avec des séries normalisées.
      Une normalisation globale (sur tous les participants) laisserait
      des différences d'amplitude inter-sujets qui noieraient les patterns
      temporels — or c'est exactement les patterns temporels qu'on veut capturer.

    Pourquoi pas le zscore par baseline comme dans load_subject ?
    → Ici on veut que le modèle compare les FORMES des signaux,
      pas les niveaux absolus. Mean=0 std=1 par série est le standard
      pour les foundation models time series.
    """
    normalized = {}

    for key, arr in series_dict.items():
        arr_norm = arr.copy()
        for i in range(arr.shape[1]):
            col = arr[:, i]
            mu  = col.mean()
            std = col.std()
            if std > 1e-6:
                arr_norm[:, i] = (col - mu) / std
            else:
                arr_norm[:, i] = 0.0  # signal constant → zéro
        normalized[key] = arr_norm

    return normalized


# Deuxième étape pour le fondation model
def resample_series(series_dict: dict, target_len: int = 512) -> dict:
    """
    Ramène toutes les séries à une longueur fixe (target_len frames).

    Pourquoi scipy.signal.resample et pas juste tronquer/padder ?
    → resample fait une interpolation dans le domaine fréquentiel (FFT).
      Il préserve mieux la forme du signal qu'un simple tronquage ou
      qu'un np.interp linéaire — important pour EDA et les signaux physio
      qui ont des patterns basse fréquence significatifs.

    Pour les séries plus longues  : sous-échantillonnage (compression)
    Pour les séries plus courtes  : sur-échantillonnage (étirement)
    Dans les deux cas             : même nombre de frames en sortie
    """
    resampled = {}

    for (subj, salle), arr in series_dict.items():
        # arr shape : (T, n_signals)
        # resample travaille sur l'axe 0 (axe temporel)
        arr_resampled = resample(arr, target_len, axis=0)
        # shape finale : (target_len, n_signals) = (512, 5)

        resampled[(subj, salle)] = arr_resampled

    return resampled



# Première étape pour le fondation model: extract_series => on choisit les données sur quoi travailler.

# Résultat::
#1: "eda_uS_filtered", "head_y", "head_z", "pitch", "yaw"::
#Avec EDA — k=2 gagne encore mais moins nettement :
#k=2  silhouette=0.385  davies_bouldin=1.64  ← toujours le meilleur
#k=5/6 silhouette~0.18  ← plateau, pas de structure claire au-delà

#2:"head_y", "head_z", "pitch", "yaw"::
#Sans EDA — k=2 gagne nettement :
#k=2  silhouette=0.537  davies_bouldin=1.17  ← clair vainqueur
#k=3  silhouette=0.246  ← chute brutale

#3: "eda_uS_filtered"::
# k=2 silhouette=0.78 à k=2, c'est extraordinairement élevé pour des données physiologiques réelles.
# k=6/7 silhouette~0.35  ← plateau
# 

# Conclusion :
#EDA seule         → silhouette=0.78  deux groupes très nets
#Mouvement seul    → silhouette=0.54  deux groupes nets
#EDA + mouvement   → silhouette=0.38  structure brouillée
def extract_series_per_room(
    subjects_data: dict,
    signals: list = ["head_y", "head_z", "pitch", "yaw"] 
) -> dict:
    """
    Extrait les séries temporelles brutes (frame par frame) pour chaque
    combinaison participant × salle.

    subjects_data : dict { "PARTICIPANT1": df, ... }  ← ton dict existant
    signals       : colonnes à extraire de df_raw

    Retourne un dict { (subject, salle): np.array(T, n_signals) }
    Les NaN sont interpolés linéairement puis remplis par les bords.
    """
    series_dict = {}

    for subject, df in subjects_data.items():
        for salle in sorted(df["ev_salle"].dropna().unique()):
            df_salle = df[df["ev_salle"] == salle].copy()

            # Garder seulement les colonnes qu'on veut, dans l'ordre
            available = [s for s in signals if s in df_salle.columns]
            if not available:
                continue

            arr = df_salle[available].values.astype(float)
            # shape : (T, n_signals)

            # Interpolation linéaire des NaN colonne par colonne
            # Pourquoi ? Le foundation model ne tolère pas les NaN.
            # On interpole plutôt que de dropper pour garder la continuité temporelle.
            for i in range(arr.shape[1]):
                col = arr[:, i]
                mask_nan = np.isnan(col)
                if mask_nan.all():
                    arr[:, i] = 0.0  # signal entièrement absent → zéro
                elif mask_nan.any():
                    idx = np.arange(len(col))
                    # np.interp ignore les NaN → on interpole sur les indices valides
                    arr[:, i] = np.interp(
                        idx,
                        idx[~mask_nan],
                        col[~mask_nan]
                    )

            series_dict[(subject, salle)] = arr

    return series_dict


def extract_window(
    df: pd.DataFrame,
    mode: str,
    window_sec: float = 30.0
) -> pd.DataFrame:
    """
    Extrait une fenêtre temporelle selon le mode choisi.

    Modes :
    - "full"        : toute la session (aucun filtrage)
    - "before_watch": fenêtre de `window_sec` secondes AVANT chaque événement
                      'premiere_interaction_montre'
    - "before_sam"  : fenêtre de `window_sec` secondes AVANT chaque 'SAM_Validated'

    Pour les modes "before_*", on retourne TOUS les segments (une fenêtre par
    événement trouvé), concaténés, avec une colonne `window_id` pour identifier
    chaque segment.

    Pourquoi une fenêtre avant l'événement et pas après ?
    → On veut capturer l'état psychophysiologique PENDANT l'exploration,
      pas la réaction à l'événement. Le SAM est rempli APRÈS avoir trouvé
      la montre ; ce qui nous intéresse c'est l'état juste avant.
    """
    if mode == "full":
        df = df.copy()
        df["window_id"] = 0
        return df

    # Identifier les timestamps cibles selon le mode
    if mode == "before_watch":
        event_label = "premiere_interaction_montre"
    elif mode == "before_sam":
        event_label = "SAM_Validated"
    else:
        raise ValueError(f"Mode inconnu : {mode}")

    # Trouver les timestamps où l'événement se produit
    # (premier timestamp de chaque bloc d'événement consécutif)
    event_mask = df["c3d_event"] == event_label
    # Détecter les fronts montants (début de chaque bloc)
    event_starts = df[event_mask & (~event_mask.shift(1, fill_value=False))]

    if event_starts.empty:
        # Pas d'événement trouvé → on retourne un df vide avec les bonnes colonnes
        df_empty = df.iloc[0:0].copy()
        df_empty["window_id"] = pd.Series(dtype=int)
        return df_empty

    segments = []
    for i, (idx, row) in enumerate(event_starts.iterrows()):
        t_end = row["timestamp"]
        t_start = t_end - window_sec

        seg = df[(df["timestamp"] >= t_start) & (df["timestamp"] < t_end)].copy()
        seg["window_id"] = i
        # t_rel_window : temps relatif dans la fenêtre (0 = début, window_sec = fin)
        seg["t_rel_window"] = seg["timestamp"] - t_start
        segments.append(seg)

    if not segments:
        return df.iloc[0:0].copy()

    return pd.concat(segments, ignore_index=True)


def compute_umap(
    df: pd.DataFrame,
    source: str = "raw",   # "raw" ou "agg"
    use_displacement: bool = False,
    new_features: bool = True,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Calcule l'UMAP sur les colonnes sélectionnées et ajoute UMAP_1, UMAP_2 au df.

    Étapes internes :
    1. Sélection des features (physio et/ou déplacement)
    2. Imputation des NaN (médiane) — nécessaire car IBI est sparse, BVP peut
       avoir des artéfacts, etc.
    3. Normalisation RobustScaler (résistant aux outliers physiologiques)
    4. UMAP 2D

    Pourquoi RobustScaler et pas StandardScaler ?
    → Les signaux physiologiques ont des pics ponctuels (arythmies, artéfacts EDA).
      StandardScaler divise par l'écart-type, très sensible aux outliers.
      RobustScaler utilise la médiane et l'IQR → plus stable.

    Pourquoi imputer par la médiane et pas supprimer les NaN ?
    → Si on supprime, on perd les timestamps avec BVP=NaN (très fréquent entre
      deux battements). La médiane est le choix le plus neutre pour des signaux
      physiologiques.
    """

    is_agg = (source == "agg")
    feature_cols = []  # ← toujours initialisé, même si tous les booléens sont False

    if use_displacement:
        cols = DISPLACEMENT_COLS_AGG if is_agg else DISPLACEMENT_COLS_RAW
        feature_cols += [c for c in cols if c in df.columns]
    if new_features:
        cols = NEW_FEATURES_AGG if is_agg else NEW_FEATURES_RAW
        feature_cols += [c for c in cols if c in df.columns]

    if not feature_cols:
        raise ValueError("Aucune feature sélectionnée pour l'UMAP.")

    X = df[feature_cols].copy()

    # 1. Imputation NaN → médiane par colonne
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # 2. Normalisation robuste
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # 3. UMAP
    # n_neighbors : taille du voisinage local considéré.
    #   Petit (5-10) → structure très locale, plus de "clusters" fins
    #   Grand (30-50) → structure globale, topologie générale
    # min_dist : distance minimale entre points dans l'espace 2D.
    #   Petit (0.0-0.1) → points très compressés en clusters
    #   Grand (0.5-1.0) → points plus dispersés, meilleure vue d'ensemble
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric="euclidean"
    )
    embedding = reducer.fit_transform(X_scaled)

    df = df.copy()
    df["UMAP_1"] = embedding[:, 0]
    df["UMAP_2"] = embedding[:, 1]
    df["_features_used"] = str(feature_cols)  # pour debug/info
    if is_agg:
        df["_umap_source"] = "agg"   # ← doit être présent
    return df



def top_features_par_cluster(df_clustered: pd.DataFrame, feature_cols: list, top_n: int = 20):
    """
    Identifie les features qui discriminent le mieux les clusters,
    via le ratio F (ANOVA one-way).
    
    Pourquoi F et pas juste la variance inter-cluster ?
    - La variance inter-cluster dit "les moyennes sont éloignées"
    - Le ratio F dit "les moyennes sont éloignées ET les points sont groupés serrés"
    - C'est bien plus informatif pour des données physiologiques bruitées
    
    F = variance_entre_clusters / variance_intra_cluster
    """
    
    # 1. Grouper les valeurs de chaque feature par cluster
    #    → liste de arrays, un par cluster
    groups = [
        df_clustered[df_clustered['cluster'] == c][feature_cols].values
        for c in sorted(df_clustered['cluster'].unique())
    ]
    # groups[i] = matrice (n_points_dans_cluster_i, n_features)
    
    # 2. Pour chaque feature, extraire les valeurs par cluster et calculer F
    results = {}
    for i, feat in enumerate(feature_cols):
        # Valeurs de cette feature dans chaque cluster (liste de 1D arrays)
        feat_groups = [g[:, i] for g in groups]
        
        # scipy.stats.f_oneway : ANOVA one-way
        # Retourne (F_statistic, p_value)
        # On ignore les NaN en les filtrant
        feat_groups_clean = [g[~np.isnan(g)] for g in feat_groups]
        
        # Besoin d'au moins 2 groupes non-vides avec >1 valeur chacun
        valid = [g for g in feat_groups_clean if len(g) > 1]
        if len(valid) >= 2:
            f_stat, p_val = stats.f_oneway(*valid)
            results[feat] = {'F': f_stat, 'p_value': p_val}
        else:
            results[feat] = {'F': np.nan, 'p_value': np.nan}
    
    # 3. Construire un DataFrame trié par F décroissant
    df_results = (
        pd.DataFrame(results).T
        .sort_values('F', ascending=False)
        .head(top_n)
    )
    
    # 4. Labels lisibles pour l'axe X
    labels = [f.replace('_mean', '').replace('_std', '') for f in df_results.index]
    
    # 5. Visualisation avec double info : hauteur = F, couleur = p-value
    fig, ax = plt.subplots(figsize=(13, 6))
    
    # Colorer les barres selon la significativité (p < 0.05 → vert, sinon orange)
    colors = ['steelblue' if p < 0.05 else 'lightcoral' 
              for p in df_results['p_value']]
    
    bars = ax.bar(range(len(df_results)), df_results['F'], color=colors, edgecolor='white')
    
    ax.set_xticks(range(len(df_results)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_title(f"Top {top_n} features discriminantes entre clusters (ratio F ANOVA)")
    ax.set_ylabel("Ratio F  (↑ = mieux séparé entre clusters vs au sein des clusters)")
    ax.set_xlabel("Feature")
    
    # Légende manuelle pour les couleurs
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='p < 0.05 (significatif)'),
        Patch(facecolor='lightcoral', label='p ≥ 0.05 (non significatif)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Nom de fichier sûr (pas de liste)
    plt.savefig(f'png-full_analyse/top_features_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nTop {top_n} features les plus discriminantes :")
    print(df_results.to_string())
    
    return df_results, fig

def get_event_label(row: pd.Series) -> str:
    """
    Retourne un label lisible pour l'événement d'une ligne.
    Utilisé pour le marqueur dans le scatter plot.
    """
    ev = row.get("c3d_event", "")
    if pd.isna(ev) or ev == "":
        return "—"
    # Simplification des labels pour la lisibilité
    mapping = {
        "premiere_interaction_montre": "🕰️ Montre",
        "SAM_Validated": "📊 SAM",
        "entree_nouvelle_room": "🚪 Nouvelle salle",
    }
    return mapping.get(ev, ev)


# ══════════════════════════════════════════════════════════════════════════════
# RETOURS AUDIO — sentiment par salle (cf. analyse_audio_par_salle.py +
# analyse_audio_sentiment_par_salle.py, exécutés à part car trop lourds/lents
# pour tourner dans l'app Streamlit — Whisper medium + BERT sur ~47 fichiers).
# ══════════════════════════════════════════════════════════════════════════════
SENTIMENT_TO_SIGNED_SCORE = {
    "Positif": 1.0,
    "Neutre": 0.0,
    "Négatif": -1.0,
}


def load_audio_sentiment(filepath: str) -> pd.DataFrame:
    """
    Charge le CSV produit par analyse_audio_sentiment_par_salle.py et ajoute
    une colonne "sentiment_signe" continue (signe du label × confiance BERT),
    utilisable directement dans une corrélation Spearman/Pearson.

    Les buckets "Indéterminé" (texte trop court ou erreur BERT) ont un
    sentiment_signe = NaN — exclus automatiquement des corrélations (dropna).
    """
    df = pd.read_csv(filepath)
    df["sentiment_signe"] = df.apply(
        lambda r: SENTIMENT_TO_SIGNED_SCORE.get(r["sentiment"], np.nan) * r["sentiment_score"]
        if r["sentiment"] in SENTIMENT_TO_SIGNED_SCORE else np.nan,
        axis=1,
    )
    return df


def compute_audio_sentiment_sam_correlation(
    df_audio: pd.DataFrame,
    df_sam: pd.DataFrame,
    method: str = "spearman",
) -> pd.DataFrame:
    """
    Corrélation entre le sentiment audio (par salle, hors bucket "global") et
    la réponse SAM (valence/arousal), calculée par salle — même logique que
    compute_signal_sam_correlation (pas de mélange inter-salles, pour ne pas
    confondre effet de salle et lien direct sentiment↔ressenti).
    """
    df_audio_salle = df_audio[df_audio["salle"] != "global"].copy()
    df_audio_salle["salle"] = df_audio_salle["salle"].astype(float)
    df_audio_salle = df_audio_salle.rename(columns={"participant": "subject"})

    df_merged = df_audio_salle.merge(df_sam, on=["subject", "salle"], how="inner")

    corr_fn = stats.spearmanr if method == "spearman" else stats.pearsonr

    rows = []
    for salle, df_s in df_merged.groupby("salle"):
        for sam_var in ["valence", "arousal"]:
            d = df_s[["sentiment_signe", sam_var]].dropna()
            if len(d) < 5 or d["sentiment_signe"].std() < 1e-9:
                continue
            r, p = corr_fn(d["sentiment_signe"], d[sam_var])
            rows.append({
                "Salle": salle,
                "SAM": sam_var,
                "r": round(r, 3),
                "p-value": round(p, 4),
                "Significatif": "✅" if p < 0.05 else "❌",
                "N": len(d),
            })
    return pd.DataFrame(rows)
