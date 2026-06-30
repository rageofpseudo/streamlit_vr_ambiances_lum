"""
feature_glossary.py
--------------------
Glossaire des features extraites par umap_utils.extract_c3d_features (mouvement)
et aggregate_subjects (physio EDA/HR/HRV).

Une feature se lit comme : <SIGNAL>_<STATISTIQUE>
  ex: jerk_y_wavelet_std = brusquerie verticale (jerk_y) + variabilité de la
      tendance lente après décomposition en ondelettes (wavelet_std)

Utilise describe_feature(nom) pour obtenir une description complète, ou
SIGNAL_DESCRIPTIONS / STAT_DESCRIPTIONS directement pour un lookup brut.
"""

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAUX — ce que mesure la variable de base
# ══════════════════════════════════════════════════════════════════════════════
SIGNAL_DESCRIPTIONS = {
    "head_x":            "Position de la tête sur l'axe X (latéral, gauche/droite).",
    "head_y":             "Position de la tête sur l'axe Y (vertical, haut/bas).",
    "head_z":             "Position de la tête sur l'axe Z (profondeur, avant/arrière).",
    "pitch":               "Inclinaison de la tête haut/bas (regarder vers le haut ou le bas).",
    "yaw":                  "Rotation de la tête gauche/droite (regarder à gauche ou à droite).",
    "roll":                 "Inclinaison latérale de la tête (pencher la tête sur le côté).",
    "speed":                "Vitesse de déplacement de la tête (norme du déplacement / dt, en m/s).",
    "path":                 "Longueur totale du chemin parcouru par la tête (somme des déplacements).",
    "jerk_x":               "Brusquerie du mouvement sur l'axe X — variation rapide de la vitesse latérale.",
    "jerk_y":               "Brusquerie du mouvement sur l'axe Y — variation rapide de la vitesse verticale (ex: sursauts).",
    "jerk_z":               "Brusquerie du mouvement sur l'axe Z — variation rapide de la vitesse en profondeur.",
    "immobility":           "Statisme — proportion du temps où le participant est quasi immobile (vitesse < seuil).",
    "angular_velocity":     "Vitesse de rotation de la tête (combinaison pitch/yaw/roll) — exploration visuelle/curiosité.",
    "eda_uS":               "Conductance cutanée brute (EDA), en microsiemens.",
    "eda_tonic":            "Composante lente de l'EDA (niveau de base) — reflète l'arousal général de fond.",
    "eda_phasic":           "Composante rapide de l'EDA — réponses ponctuelles à un stimulus (SCR).",
    "eda_driver":           "Signal nerveux sympathique estimé (driver SMNA, sortie de cvxEDA).",
    "hr":                   "Fréquence cardiaque, en battements par minute (bpm).",
    "hrv":                  "Variabilité de la fréquence cardiaque (RMSSD) — équilibre sympathique/parasympathique.",
    "ibi":                  "Inter-Beat Interval — temps entre deux battements cardiaques successifs (secondes).",
    "scr":                  "Skin Conductance Response — pics de réactivité phasique de l'EDA.",
    "sdnn":                 "Écart-type de tous les intervalles RR — variabilité cardiaque globale (long terme).",
    "pnn50":                "% d'intervalles RR consécutifs différant de plus de 50ms — activité parasympathique.",
}

# ══════════════════════════════════════════════════════════════════════════════
# STATISTIQUES — ce que mesure le suffixe appliqué au signal
# ══════════════════════════════════════════════════════════════════════════════
STAT_DESCRIPTIONS = {
    "mean":               "Moyenne du signal sur la fenêtre/la salle — niveau général (signé : peut être négatif, indique une dérive directionnelle pour jerk_x/y/z).",
    "mean_abs":           "Moyenne de la valeur absolue — intensité moyenne, indépendamment du sens/direction (toujours positive).",
    "std":                "Écart-type — dispersion du signal autour de sa moyenne.",
    "median":             "Médiane — valeur centrale, robuste aux valeurs extrêmes.",
    "min":                "Valeur minimale observée.",
    "max":                "Valeur maximale observée.",
    "variance":           "Variance — dispersion au carré (std²).",
    "kurtosis":           "Aplatissement de la distribution — élevé = pics/queues extrêmes fréquents, ~0 = proche d'une normale.",
    "skewness":           "Asymétrie de la distribution — positif = étalée vers les valeurs hautes, négatif = vers les basses.",
    "rms":                "Root Mean Square — énergie globale du signal.",
    "iqr":                "Écart interquartile (Q3-Q1) — dispersion robuste aux valeurs extrêmes.",
    "peak2peak":          "Amplitude totale (max - min).",
    "mad":                "Déviation absolue moyenne — mesure de dispersion robuste, alternative au std.",
    "zcr":                "Zero-Crossing Rate — nombre de changements de signe du signal ; mesure d'agitation/oscillation.",
    "autocorr":           "Autocorrélation à 1 pas de temps — régularité du signal d'un instant au suivant (proche de 1 = très lisse).",
    "mean_abs_diff":      "Variation moyenne frame à frame — vitesse de changement local du signal.",
    "fft_mean":           "Énergie spectrale moyenne après transformée de Fourier.",
    "fft_max":            "Pic d'énergie spectrale dominant (fréquence la plus marquée).",
    "spectral_centroid":  "Fréquence 'moyenne' pondérée par l'énergie — bas = mouvements lents, haut = saccades/tremblements.",
    "wavelet_energy":     "Énergie totale du signal après décomposition en ondelettes (toutes échelles temporelles confondues).",
    "wavelet_std":        "Variabilité de la composante lente (tendance de fond) après décomposition en ondelettes.",
    "ratio":              "Proportion (0 à 1) — ex: immobility_ratio = % du temps immobile.",
    "bouts":              "Nombre d'épisodes distincts (ex: immobility_bouts = nombre de fois où le participant s'immobilise).",
    "range":              "Étendue (max - min) — ex: head_yaw_range = amplitude totale de rotation gauche/droite.",
    "rate":               "Fréquence d'occurrence, généralement par minute (ex: scr_rate = SCR par minute).",
    "auc":                "Aire sous la courbe (Area Under Curve) — intensité cumulée du signal.",
    "amplitude_mean":     "Amplitude moyenne des pics détectés.",
    "amplitude_std":      "Variabilité de l'amplitude des pics détectés.",
    "lf_power":           "Puissance basse fréquence (0.04-0.15Hz) de la HRV — composante sympathique.",
    "hf_power":           "Puissance haute fréquence (0.15-0.40Hz) de la HRV — composante parasympathique.",
    "lf_hf":              "Ratio LF/HF — >1 dominance sympathique (stress), <1 dominance parasympathique (repos).",
    "cv":                 "Coefficient de variation (std/mean) — dispersion relative à l'échelle du signal.",
    "entropy":            "Entropie de Shannon de la distribution de l'angle — élevée = regard qui explore largement (curiosité/confort), basse = regard rétréci sur une plage étroite (rétrécissement attentionnel, associé au stress/à la menace dans la littérature).",
}

# ══════════════════════════════════════════════════════════════════════════════
# Cas particuliers — noms de features qui ne suivent pas le patron signal+stat
# ══════════════════════════════════════════════════════════════════════════════
FULL_NAME_OVERRIDES = {
    "immobility_ratio":      "Proportion du temps où le participant est quasi immobile (vitesse < 0.05 m/s).",
    "immobility_bouts":      "Nombre d'épisodes distincts d'immobilité (pas juste leur durée totale).",
    "angular_velocity_mean": "Vitesse moyenne de rotation de la tête — indicateur d'exploration visuelle.",
    "angular_velocity_std":  "Variabilité de la vitesse de rotation de la tête.",
    "head_yaw_range":        "Amplitude totale de rotation gauche/droite (yaw) sur la salle.",
    "head_pitch_range":      "Amplitude totale d'inclinaison haut/bas (pitch) sur la salle.",
    "path_length":           "Longueur totale du chemin parcouru par la tête sur la salle.",
    "entry_speed_peak":              "Pic de vitesse de déplacement de la tête dans les 3 premières secondes après l'entrée dans la salle — réaction immédiate à l'ambiance, pas une moyenne sur toute la visite.",
    "entry_speed_mean":              "Vitesse moyenne de déplacement de la tête dans les 3 premières secondes après l'entrée dans la salle.",
    "entry_jerk_peak":               "Pic de brusquerie (jerk) dans les 3 premières secondes après l'entrée — proxy d'un sursaut/réaction de surprise à l'ambiance.",
    "entry_angular_velocity_peak":   "Pic de vitesse de rotation de tête dans les 3 premières secondes après l'entrée — réaction d'orientation immédiate (où regarde-t-on en premier en arrivant).",
    "cf1_presence": "Feature composite (Saha et al. 2025) combinant variance et pente du tonique + du phasique EDA — capture simultanément l'arousal de fond (lent) et les réponses transitoires (rapides). Dans l'article original, la feature la plus corrélée individuellement avec le niveau de présence rapporté (jusqu'à 82%).",
    "cf2_presence": "Feature composite (Saha et al. 2025) = variance phasique / puissance phasique — intensité relative des réponses transitoires (SCR), normalisée par l'énergie du signal, indépendamment de l'amplitude absolue.",
}


def describe_feature(name: str) -> str:
    """
    Construit une description lisible pour une feature donnée, en combinant
    le signal de base et la statistique appliquée (ex: 'jerk_y_wavelet_std').

    Cherche d'abord dans FULL_NAME_OVERRIDES (noms qui ne suivent pas le
    patron signal+stat), puis essaie de découper en signal connu + suffixe connu.
    """
    if name in FULL_NAME_OVERRIDES:
        return FULL_NAME_OVERRIDES[name]

    # Essaie le préfixe signal le plus long possible (ex: "eda_tonic" avant "eda")
    candidates = sorted(SIGNAL_DESCRIPTIONS.keys(), key=len, reverse=True)
    for signal in candidates:
        prefix = signal + "_"
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            if suffix in STAT_DESCRIPTIONS:
                return f"{SIGNAL_DESCRIPTIONS[signal]} — {STAT_DESCRIPTIONS[suffix]}"
            return SIGNAL_DESCRIPTIONS[signal]

    return "Description non disponible — feature non répertoriée dans le glossaire."


if __name__ == "__main__":
    # Exemples sur les features les plus discriminantes trouvées en LDA/ANOVA
    exemples = [
        "jerk_y_wavelet_std", "speed_std", "jerk_y_mean", "jerk_z_mean",
        "jerk_x_mean", "eda_tonic_spectral_centroid", "eda_tonic_mean",
        "pitch_wavelet_std", "immobility_ratio", "angular_velocity_mean",
    ]
    for f in exemples:
        print(f"{f:35s} → {describe_feature(f)}")
