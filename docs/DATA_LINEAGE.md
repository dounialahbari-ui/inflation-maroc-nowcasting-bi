# Traçabilité des données

Ce projet distingue explicitement trois catégories de données, pour que la chaîne de traitement soit vérifiable de bout en bout.

## 1. `data/raw/` — Données sources, non modifiées dans leur structure

| Fichier | Origine | Contenu |
|---|---|---|
| `WDI_Comparaison_Indicators.csv` | Banque Mondiale (World Development Indicators) | Inflation, PIB/habitant, pauvreté, part des salariés — 6 pays, 2017-2025 |
| `Prediction_WEO_FMI.csv` | FMI, World Economic Outlook, avril 2026 | Série historique et prévisionnelle de l'inflation Maroc (indicateur PCPIPCH), 2017-2027 |

Ces fichiers sont téléchargés directement depuis les portails officiels (databank.worldbank.org, imf.org/weo) et ne subissent qu'un export brut, sans recalcul.

## 2. `data/processed/` — Donnée nettoyée, utilisée par le dashboard Power BI

| Fichier | Construction |
|---|---|
| `data_clean.csv` | Fusion mensuelle de 3 sources (HCP : IPC général + 6 sous-indices sectoriels ; Bank Al-Maghrib : inflation BKAM, inflation sous-jacente, taux directeur, M3 ; Investing.com : taux USD/MAD, EUR/MAD). Nettoyage : harmonisation des dates, gestion des valeurs manquantes (glissement annuel non calculable sur 2017, mois 2026 non publiés au moment de l'extraction). |

C'est ce fichier qui alimente directement le modèle de données Power BI (`dashboard/Dashboard_Inflation_Maroc.pbix`).

### Détail colonne par colonne de `data_clean.csv`

| Colonne | Source exacte | Fréquence native | Traitement appliqué |
|---|---|---|---|
| `IPC_General`, `IPC_Alimentaire`, `IPC_Logement`, `IPC_Transports`, `IPC_Sante`, `IPC_Restaurants_Hotels` | HCP — Indice des prix à la consommation, base 2017 | Mensuelle | Aucun recalcul, alignement du format de date uniquement |
| `Inflation_BKAM`, `Inflation_Sous_Jacente` | Bank Al-Maghrib — publications officielles (glissement annuel) | Mensuelle | Aucun recalcul ; glissement annuel non calculable sur les 12 premiers mois (2017) → valeurs manquantes conservées telles quelles (non imputées) |
| `Taux_Directeur_pct` | Bank Al-Maghrib — décisions du Conseil | Événementielle (reportée au mois) | Report de la dernière valeur connue sur chaque mois jusqu'au changement suivant |
| `M3` | Bank Al-Maghrib — agrégats monétaires | Mensuelle | Aucun recalcul |
| `USD_MAD`, `EUR_MAD` | Investing.com — cours de clôture mensuels | Quotidienne à l'origine | Agrégation en moyenne mensuelle |

**Alignement des fréquences** : toutes les séries sont ramenées à une fréquence mensuelle commune avant fusion (clé de jointure : année + mois). Aucune interpolation n'est appliquée aux valeurs manquantes ; elles sont conservées en `NaN` et explicitement exclues des calculs concernés (ex. : 2017 exclu du backtest car l'inflation en glissement annuel n'y est pas calculable).

## 3. `data/model_outputs/` — Données générées par l'exécution du pipeline

Tous les fichiers de ce dossier sont **produits automatiquement** en exécutant `src/prediction_inflation_maroc.py` sur `data/processed/data_clean.csv` et `data/raw/Prediction_WEO_FMI.csv`. Ils ne sont jamais modifiés à la main.

| Fichier | Contenu |
|---|---|
| `resultats_backtest_court_vs_long.csv` | RMSE / MAE / MAPE / MASE des 7 modèles, séparés court terme (1-6 mois) et long terme (7-19 mois) |
| `previsions_futures_tous_modeles.csv` | Prévisions mensuelles des 7 modèles, juin 2026 → décembre 2027 |
| `prevision_court_terme_6mois.csv` / `prevision_longue_2027.csv` | Sous-ensembles du fichier précédent |
| `comparaison_FMI_2027.csv` | Prévision 2027 de chaque modèle vs FMI (2 méthodes de calcul) + validation croisée 2026 |
| `figures/*.png` | Graphiques générés par le script (matplotlib) |

**Reproductibilité vérifiée** : ce dépôt a été testé en exécutant `python src/prediction_inflation_maroc.py` depuis un environnement vierge (dépendances listées dans `requirements.txt`) — les fichiers obtenus sont identiques à ceux fournis dans `data/model_outputs/`.
