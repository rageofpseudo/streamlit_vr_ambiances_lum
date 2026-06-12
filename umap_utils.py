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

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import umap
from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter
import cvxeda

# ANOVA
from scipy import stats
from statsmodels.multivariate.manova import MANOVA
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM  # ANOVA à mesures répétées


# NON-PARAMETRIQUE (Aucune hypothèse sur la distribution)
from scipy.stats import friedmanchisquare, mannwhitneyu

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


# ── Colonnes utilisées pour l'UMAP ────────────────────────────────────────────
# On distingue physio et déplacement pour pouvoir les activer/désactiver.
PHYSIO_COLS = ["eda_uS", "bvp", "hr_bpm", "temp_C"]
PHYSIO_FILTERED_COLS = ["hrv_rmssd", "eda_uS_filtered"]
DISPLACEMENT_COLS = ["head_x", "head_y", "head_z"]
# head_qx/qy/qz/qw (quaternion) sont utiles mais très corrélés → on les laisse
# de côté par défaut pour ne pas noyer le signal. On pourra les ajouter plus tard.

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
 
# Extraction de la feature SCR depuis EDA -> selon certains articles c'est pas tout le temps la plus utile.
def extract_scr_features(eda_phasic: pd.Series, fs: int = 30, threshold: float = 0.02) -> dict:
    """
    Extrait les features des Skin Conductance Responses (SCR) depuis le signal phasique.
    
    Pourquoi le signal phasique et pas le brut ?
    cvxEDA a déjà isolé la composante rapide (réponses aux stimuli).
    Travailler sur le brut mélangerait SCR et dérive lente (tonic).
    
    threshold=0.02 : seuil en µS en dessous duquel on ignore les micro-fluctuations.
    Benedek & Kaernbach (2010) recommandent 0.01-0.05 µS selon le bruit du signal.
    
    distance=fs*2 : deux pics doivent être séparés d'au moins 2s.
    Correspond à la période réfractaire physiologique d'une SCR.
    """
    clean = eda_phasic.dropna()
    
    if len(clean) < fs * 5:  # moins de 5s de signal → pas fiable
        return {'scr_count': np.nan, 'scr_amp_mean': np.nan, 'scr_auc': np.nan}
    
    peaks, props = find_peaks(
        clean,
        height=threshold,   # hauteur minimale d'un pic
        distance=fs * 2     # distance minimale entre deux pics (2s)
    )
    
    n_peaks = len(peaks)
    
    return {
        'scr_count':    n_peaks,
        'scr_amp_mean': float(np.mean(props['peak_heights'])) if n_peaks > 0 else 0.0,
        # AUC : intégrale du signal phasique positif, divisée par fs pour avoir des µS·s
        # clip(lower=0) : on ignore les valeurs négatives (artefacts cvxEDA)
        'scr_auc':      float(clean.clip(lower=0).sum() / fs)
    }

# Extraction de features SDNN, pNN50, LF/HF ratio, HR range depuis HR.
def extract_hrv_features(ibi_series: pd.Series):
    """
    SDNN.       --- Écart-type de tous les IBI                      --- Variabilité globale (vs RMSSD qui est court-terme)
    pNN50       --- % d'IBI consécutifs différant de >50ms          --- Activité parasympathique
    LF/HF ratio --- FFT sur les IBI, bande 0.04-0.15Hz / 0.15-0.4Hz --- Équilibre sympathique/parasympathique
    HR range    --- max(HR) - min(HR) sur la salle                  --- Réactivité cardiaque totale
    """
    ibi = ibi_series.dropna().values * 1000  # en ms
    if len(ibi) < 10:   
        return {}
    
    sdnn  = np.std(ibi, ddof=1)
    diff  = np.diff(ibi)
    pnn50 = np.sum(np.abs(diff) > 50) / len(diff) * 100
    rmssd2 = np.sqrt(np.mean(diff**2))
    
    return {'sdnn': sdnn, 'pnn50': pnn50, 'rmssd2': rmssd2}


def extract_c3d_features(df: pd.DataFrame) -> dict:
    """
    Extrait des features scalaires de mouvement depuis un DataFrame brut (1 sujet x 1 salle).
    
    Explication de toutes les features:
    -> Position/déplacement
    - path_length : distance totale parcourue — un participant qui explore beaucoup vs qui reste statique
    - speed_mean / speed_std : vitesse moyenne + sa variabilité — est-ce qu'il se déplace régulièrement ou par à-coups ?

    -> Rotation de tête (quaternions head_qx/y/z/qw)
    - angular_velocity_mean : vitesse angulaire — est-ce qu'il regarde partout rapidement ?
    - head_rotation_var : variance totale des rotations — amplitude d'exploration visuelle

    -> Statique ou pas statique ? 
    - immobility_ratio : proportion du temps avec vitesse < seuil — est-ce qu'il est "figé" ?
    - immobility_bouts : nombre d'épisodes immobiles — distinct de la proportion (2 x 30s ≠ 60 x 1s)


    Retourne un dict : { "speed_mean": ..., "path_length": ..., ... }
    """

    # ── 1. Vitesse de déplacement ─────────────────────────────────────────────
    # On a déjà speed_mps si tu la calcules en amont, sinon on la recalcule ici.
    # .diff() donne NaN sur la 1ère ligne → on les ignore avec dropna() dans les stats.
    
    dx = df["head_x"].diff()
    dy = df["head_y"].diff()
    dz = df["head_z"].diff()
    dt = df["timestamp"].diff().replace(0, np.nan)
    
    speed = np.sqrt(dx**2 + dy**2 + dz**2) / dt  # m/s, NaN sur la 1ère ligne

    # Scalaires vitesse
    speed_clean = speed.dropna()
    features = {
        "speed_mean":    speed_clean.mean(),   # vitesse moyenne d'exploration
        "speed_std":     speed_clean.std(),    # variabilité (mouvements réguliers vs erratiques)
        "speed_median":  speed_clean.median(), # robuste aux pics (saccades rapides)
        "path_length":   (np.sqrt(dx**2 + dy**2 + dz**2)).sum(),  # distance totale cumulée
    }

    # ── 2. Statisme ────────────────────────────────────────────────────────────
    # Seuil empirique : en dessous de 0.05 m/s, on considère le participant immobile.
    # Ce seuil peut être ajusté — 0.05 m/s = ~5cm/s, bruit de jitter HMD typique.
    
    IMMOBILITY_THRESHOLD = 0.05  # m/s

    is_immobile = speed_clean < IMMOBILITY_THRESHOLD  # Series de booléens

    # Ratio : proportion du temps immobile (0.0 → 1.0)
    features["immobility_ratio"] = is_immobile.mean()

    # Nombre de "bouts" d'immobilité = nombre de passages False→True dans la série.
    # .astype(int).diff() donne +1 au début de chaque bout.
    # Pourquoi c'est utile ? ratio=0.5 peut être 1×long_épisode ou 50×courts_épisodes.
    bouts = (is_immobile.astype(int).diff() == 1).sum()
    features["immobility_bouts"] = int(bouts)

    # ── 3. Rotation de tête (quaternions → vitesse angulaire) ─────────────────
    # Les quaternions (qx, qy, qz, qw) encodent l'orientation 3D de la tête.
    # On ne peut pas faire diff() directement sur un quaternion (espace non-euclidien).
    # Solution : calculer la "distance" angulaire entre 2 orientations successives.
    #
    # scipy.spatial.transform.Rotation.inv() * Rotation_suivante = Rotation_delta
    # .magnitude() donne l'angle de cette rotation delta en radians.
    
    q_cols = ["head_qx", "head_qy", "head_qz", "head_qw"]
    
    if all(c in df.columns for c in q_cols):
        # Rotation.from_quat() attend l'ordre [x, y, z, w]
        quats = df[q_cols].dropna().values  # shape (N, 4)
        
        if len(quats) > 1:
            rots = Rotation.from_quat(quats)  # objet Rotation vectorisé
            
            # Rotation relative entre chaque paire consécutive
            # rots[:-1].inv() * rots[1:] = "combien ai-je tourné entre t et t+1 ?"
            delta_rots = rots[:-1].inv() * rots[1:]
            
            # .magnitude() = angle en radians de chaque rotation delta
            angular_displacements = delta_rots.magnitude()  # shape (N-1,)
            
            # dt correspondant (on exclut la 1ère ligne comme pour speed)
            dt_clean = dt.dropna().values
            n = min(len(angular_displacements), len(dt_clean))
            
            # Vitesse angulaire en rad/s
            angular_velocity = angular_displacements[:n] / (dt_clean[:n] + 1e-9)
            
            features["angular_velocity_mean"] = float(np.mean(angular_velocity))
            features["angular_velocity_std"]  = float(np.std(angular_velocity))
            #  C'est une mesure de dispersion — à quel point les valeurs s'éloignent de la moyenne.
            # Salle A : [0.5, 0.5, 0.5, 0.5]  → mean=0.5, std=0.0  (signal stable, plat)
            # Salle B : [0.1, 0.9, 0.2, 0.8]  → mean=0.5, std=0.35 (signal qui fluctue beaucoup)
            
            # Variance totale de l'orientation (proxy d'amplitude d'exploration visuelle)
            # On utilise les angles d'Euler (yaw, pitch) — plus interprétables que quaternions bruts
            euler = rots.as_euler("yxz", degrees=True)  # [pitch, yaw, roll]
            features["head_yaw_range"]   = float(np.ptp(euler[:, 1]))  # amplitude yaw  (gauche-droite)
            features["head_pitch_range"] = float(np.ptp(euler[:, 0]))  # amplitude pitch (haut-bas)
        else:
            # Pas assez de données rotation
            for k in ["angular_velocity_mean", "angular_velocity_std", 
                       "head_yaw_range", "head_pitch_range"]:
                features[k] = np.nan
    
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
        #"scr_count",
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

            row.update(extract_scr_features(df_salle["eda_phasic"]))
            row.update(extract_hrv_features(df_salle["ibi_s"]))
            row.update(extract_c3d_features(df_salle))  # ← df_salle, pas df

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
    
    # Garde seulement les sujets qui ont les 5 salles
    counts = df_clean.groupby("subject")["salle"].count()
    complete_subjects = counts[counts == 5].index
    df_clean = df_clean[df_clean["subject"].isin(complete_subjects)]
    
    if len(complete_subjects) < 5:
        return {"error": f"Pas assez de sujets complets ({len(complete_subjects)})"}
    
    try:
        aovrm = AnovaRM(
            data=df_clean,
            depvar=signal_col,      # variable dépendante
            subject="subject",      # identifiant sujet
            within=["salle"]        # facteur intra-sujet
        )
        result = aovrm.fit()
        table = result.anova_table
        
        return {
            "F": table["F Value"].values[0],
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
        
        # Garde seulement les sujets qui ont les 5 salles
        df_clean = df_agg[["subject", "salle", col]].dropna()
        counts = df_clean.groupby("subject")["salle"].count()
        complete_subjects = counts[counts == 5].index
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
        counts = df_clean.groupby("subject")["salle"].count()
        complete_subjects = counts[counts == 5].index
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
            filtered_values = low_pass_filter(df.loc[valid_mask, col].values)
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
    baseline_samples = 30 * 30  # 30 secondes × 30 Hz
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
    phasic, tonic, driver = run_cvxeda(df["eda_uS_filtered"])   #.dropna() si jamais bug
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

    return df





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
    use_physio: bool = True,
    use_physio_filtered: bool = False,
    use_displacement: bool = True,
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
    feature_cols = []
    if use_physio:
        feature_cols += [c for c in PHYSIO_COLS if c in df.columns]
    if use_physio_filtered:
        feature_cols += [c for c in PHYSIO_FILTERED_COLS if c in df.columns]
    if use_displacement:
        feature_cols += [c for c in DISPLACEMENT_COLS if c in df.columns]

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

    return df


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
