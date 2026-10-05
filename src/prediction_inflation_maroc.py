# -*- coding: utf-8 -*-
"""
PREDICTION DE L'INFLATION AU MAROC - Comparaison de 5 modeles
ARIMA | VAR | Random Forest | XGBoost | LSTM (PyTorch)
vs prevision officielle FMI (WEO avril 2026)

Auteure : Lahbari Dounia - Master Econometrie & Data Science
Donnees : data_clean.csv (HCP, Bank Al-Maghrib, Investing.com)
          Prediction_WEO_FMI.csv (FMI, World Economic Outlook avril 2026)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

from xgboost import XGBRegressor

import torch
import torch.nn as nn

# ============================================================
# 0. CONFIGURATION
# ============================================================
import glob
import os

# Chemins robustes : fonctionnent quel que soit le dossier depuis lequel le
# script est lance (racine du depot, dossier src/, etc.), car ils sont
# calcules a partir de l'emplacement reel de ce fichier, pas du dossier courant.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # racine du depot
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "data_clean.csv")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "model_outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

TARGET = "Inflation_BKAM"
FORECAST_END = "2027-12-01"
SHORT_HORIZON = 6                       # mois 1-6 = horizon "operationnel"
LOOK_BACK = 12                          # fenetre pour features/LSTM
ADF_ALPHA = 0.05                        # seuil du test de stationnarite

torch.manual_seed(42)
np.random.seed(42)

MODELES = ["ARIMA", "VAR", "Random Forest", "XGBoost", "LSTM", "Naif", "Naif_saisonnier"]

# ============================================================
# 1. LECTURE DE LA PREVISION OFFICIELLE DU FMI (plus de chiffre en dur)
# ============================================================
def find_fmi_file():
    """Cherche le fichier FMI meme si son nom varie (espaces vs underscores,
    majuscules/minuscules) - evite un FileNotFoundError si le fichier a ete
    renomme ou telecharge avec un nom legerement different."""
    candidats = [
        os.path.join(RAW_DIR, "Prediction_WEO_FMI.csv"),
        os.path.join(RAW_DIR, "Prediction WEO FMI.csv"),
    ]
    for c in candidats:
        if pd.io.common.file_exists(c):
            return c
    # recherche large en dernier recours : n'importe quel csv contenant "WEO" ou "FMI"
    # dans data/raw/ (pas dans le dossier courant, qui peut varier selon d'ou le script est lance)
    trouves = [f for f in glob.glob(os.path.join(RAW_DIR, "*.csv"))
               if "weo" in f.lower() or "fmi" in f.lower()]
    if trouves:
        return trouves[0]
    raise FileNotFoundError(
        f"Aucun fichier FMI trouve dans {RAW_DIR}. Verifie qu'un fichier contenant les "
        "previsions WEO/FMI (ex: 'Prediction_WEO_FMI.csv') s'y trouve bien."
    )


FMI_PATH = find_fmi_file()
fmi_df = pd.read_csv(FMI_PATH)
fmi_row = fmi_df[fmi_df["COUNTRY"] == "Morocco"].iloc[0]
FMI_2026 = float(fmi_row["2026"])
FMI_2027 = float(fmi_row["2027"])
print(f"[FMI] Lu depuis {FMI_PATH} : inflation Maroc 2026 = {FMI_2026}% | "
      f"2027 = {FMI_2027}% (indicateur {fmi_row['SERIES_CODE']})")

# ============================================================
# 2. CHARGEMENT ET PREPARATION DES DONNEES
# ============================================================
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
df_ipc_full = df.copy()   # copie AVANT filtrage, gardee pour la reconstruction d'IPC (section 6)
df = df[df[TARGET].notna()].reset_index(drop=True)   # 2017 exclu (inflation glissante non calculable)

df["M3_growth"] = df["M3"].pct_change() * 100

print(f"\nDonnees : {len(df)} lignes, {df['date'].min().date()} -> {df['date'].max().date()}")

for lag in [1, 2, 3, 6, 12]:
    df[f"infl_lag{lag}"] = df[TARGET].shift(lag)
df["M3_growth_lag1"] = df["M3_growth"].shift(1)
df["EUR_MAD_lag1"] = df["EUR_MAD"].shift(1)
df["USD_MAD_lag1"] = df["USD_MAD"].shift(1)
df["Taux_Directeur_lag1"] = df["Taux_Directeur_pct"].shift(1)
df["Month"] = df["date"].dt.month

FEATURES = ["infl_lag1", "infl_lag2", "infl_lag3", "infl_lag6", "infl_lag12",
            "M3_growth_lag1", "EUR_MAD_lag1", "USD_MAD_lag1",
            "Taux_Directeur_lag1", "Month"]

df_ml = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
print(f"Apres lags : {len(df_ml)} lignes, {df_ml['date'].min().date()} -> "
      f"{df_ml['date'].max().date()}")


# ============================================================
# 3. FONCTIONS DE PREVISION REUTILISABLES
#    (utilisees a la fois pour le backtest et la prevision reelle)
# ============================================================
def find_best_arima_order(series):
    best_aic, best_order = np.inf, (1, 0, 0)
    for p in range(0, 4):
        for d in [0, 1]:
            for q in range(0, 4):
                try:
                    m = ARIMA(series, order=(p, d, q)).fit()
                    if m.aic < best_aic:
                        best_aic, best_order = m.aic, (p, d, q)
                except Exception:
                    continue
    return best_order


def forecast_arima(train_series, steps, order):
    model = ARIMA(train_series, order=order).fit()
    return model.forecast(steps=steps).values


def check_stationarity(series, name, alpha=ADF_ALPHA, verbose=True):
    """Test de Dickey-Fuller augmente (ADF).
    H0 = la serie a une racine unitaire (NON stationnaire).
    Si p-value < alpha, on rejette H0 -> serie stationnaire."""
    pval = adfuller(series.dropna())[1]
    stationary = pval < alpha
    if verbose:
        etat = "stationnaire" if stationary else "NON stationnaire -> differenciation d'ordre 1"
        print(f"    ADF [{name}] p-value={pval:.4f} -> {etat}")
    return stationary


def forecast_var(train_df, cols, steps, maxlags=6, verbose=True):
    """VAR avec verification de stationnarite (ADF) sur chaque variable.
    Les variables non stationnaires sont differenciees avant l'estimation,
    puis la prevision de la cible est reconvertie en niveau (cumsum a partir
    de la derniere valeur connue) si elle a ete differenciee."""
    data = train_df.set_index("date")[cols].dropna()

    if verbose:
        print("  Test de stationnarite (ADF) avant estimation du VAR :")
    diff_flags = {c: not check_stationarity(data[c], c, verbose=verbose) for c in cols}

    data_est = data.copy()
    for c in cols:
        if diff_flags[c]:
            data_est[c] = data[c].diff()
    data_est = data_est.dropna()

    lag_order = VAR(data_est).select_order(maxlags=maxlags).aic
    lag_order = max(lag_order, 1)
    model = VAR(data_est).fit(lag_order)
    fc = model.forecast(data_est.values[-lag_order:], steps=steps)
    fc_target = fc[:, 0]   # 1ere colonne = TARGET

    if diff_flags[cols[0]]:
        # la cible a ete differenciee -> on re-cumule a partir du dernier niveau connu
        last_level = data[cols[0]].iloc[-1]
        fc_target = last_level + np.cumsum(fc_target)

    return fc_target


def forecast_naive(train_series, steps):
    """Naif simple (benchmark standard en prevision) : la prevision est la
    derniere valeur connue, repetee sur tout l'horizon. Si un modele plus
    complexe ne fait pas mieux que ca, sa sophistication n'apporte rien."""
    last_val = train_series.iloc[-1]
    return np.full(steps, last_val)


def forecast_naive_seasonal(hist_dict, future_dates):
    """Naif saisonnier : infl(t) = infl(t-12). Pertinent ici car la cible
    est deja un taux glissant annuel (forte autocorrelation a 12 mois).
    Au-dela de 12 mois d'horizon, on reutilise les valeurs deja "predites"
    par ce meme benchmark (chainage), comme pour les autres modeles."""
    hist = dict(hist_dict)
    preds = []
    for d in future_dates:
        base_date = d - pd.DateOffset(months=12)
        val = hist.get(base_date)
        if val is None:
            val = preds[-1] if preds else list(hist.values())[-1]
        preds.append(val)
        hist[d] = val
    return np.array(preds)


def forecast_tree_model(model, train_df, steps, future_dates):
    """Prevision recursive pour Random Forest / XGBoost. Hypothese :
    les variables exogenes (M3_growth, taux de change, taux directeur)
    restent constantes a leur derniere valeur connue (scenario "sans choc")."""
    last_row = train_df.iloc[-1]
    infl_hist = list(train_df[TARGET].values)
    m3_flat, eur_flat = last_row["M3_growth"], last_row["EUR_MAD"]
    usd_flat, taux_flat = last_row["USD_MAD"], last_row["Taux_Directeur_pct"]

    preds = []
    for d in future_dates:
        feat = pd.DataFrame([{
            "infl_lag1": infl_hist[-1], "infl_lag2": infl_hist[-2],
            "infl_lag3": infl_hist[-3], "infl_lag6": infl_hist[-6],
            "infl_lag12": infl_hist[-12], "M3_growth_lag1": m3_flat,
            "EUR_MAD_lag1": eur_flat, "USD_MAD_lag1": usd_flat,
            "Taux_Directeur_lag1": taux_flat, "Month": d.month,
        }])[FEATURES]
        p = model.predict(feat)[0]
        preds.append(p)
        infl_hist.append(p)
    return np.array(preds)


class InflationLSTM(nn.Module):
    def __init__(self, hidden_size=50):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                             num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_lstm(train_series_scaled, look_back=LOOK_BACK, epochs=200, patience=15):
    def make_seq(data, lb):
        Xs, ys = [], []
        for i in range(len(data) - lb):
            Xs.append(data[i:i + lb, 0])
            ys.append(data[i + lb, 0])
        return np.array(Xs), np.array(ys)

    X, y = make_seq(train_series_scaled, look_back)
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    model = InflationLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    best_loss, counter, best_state = np.inf, 0, None
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_t)
        loss = loss_fn(out, y_t)
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def forecast_lstm(model, last_window_scaled, steps, look_back=LOOK_BACK):
    window = list(last_window_scaled)
    preds_scaled = []
    with torch.no_grad():
        for _ in range(steps):
            x_in = torch.tensor(window[-look_back:],
                                 dtype=torch.float32).view(1, look_back, 1)
            p = model(x_in).item()
            preds_scaled.append(p)
            window.append(p)
    return np.array(preds_scaled)


def evaluate_split(y_true, y_pred, name, short_h=SHORT_HORIZON):
    """RMSE/MAE globaux + separes horizon court (1..short_h) et long (le reste).
    MAPE exclut les mois ou |actuel|<=0.5 (evite l'explosion pres de zero)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    def metrics(yt, yp):
        rmse = np.sqrt(mean_squared_error(yt, yp))
        mae = mean_absolute_error(yt, yp)
        mask = np.abs(yt) > 0.5
        mape = np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100 if mask.sum() else np.nan
        return rmse, mae, mape

    rmse_g, mae_g, mape_g = metrics(y_true, y_pred)
    rmse_s, mae_s, mape_s = metrics(y_true[:short_h], y_pred[:short_h])
    rmse_l, mae_l, mape_l = metrics(y_true[short_h:], y_pred[short_h:])

    print(f"{name:15s} | GLOBAL  RMSE={rmse_g:.3f} MAE={mae_g:.3f} MAPE={mape_g:.1f}%")
    print(f"{'':15s} | COURT(1-{short_h}m)  RMSE={rmse_s:.3f} MAE={mae_s:.3f} MAPE={mape_s:.1f}%")
    print(f"{'':15s} | LONG({short_h+1}-{len(y_true)}m)  RMSE={rmse_l:.3f} MAE={mae_l:.3f} MAPE={mape_l:.1f}%")

    return {"Modele": name, "RMSE_global": rmse_g, "MAE_global": mae_g, "MAPE_global": mape_g,
            "RMSE_court": rmse_s, "MAE_court": mae_s, "MAPE_court": mape_s,
            "RMSE_long": rmse_l, "MAE_long": mae_l, "MAPE_long": mape_l}


# ============================================================
# 4. BACKTEST : on cache les 19 derniers mois CONNUS et on les predit
#    a l'aveugle, pour mesurer comment l'erreur grandit avec l'horizon
# ============================================================
future_dates_real = pd.date_range(df_ml["date"].max() + pd.offsets.MonthBegin(1),
                                   FORECAST_END, freq="MS")
BACKTEST_H = len(future_dates_real)   # meme longueur que la vraie prevision
print(f"\n=== BACKTEST sur les {BACKTEST_H} derniers mois connus "
      f"(horizon court=1-{SHORT_HORIZON}, long={SHORT_HORIZON+1}-{BACKTEST_H}) ===\n")

train_bt = df_ml.iloc[:-BACKTEST_H].reset_index(drop=True)
test_bt = df_ml.iloc[-BACKTEST_H:].reset_index(drop=True)
test_dates_bt = test_bt["date"]
y_true_bt = test_bt[TARGET].values

print(f"Train backtest : {len(train_bt)} mois (jusqu'a {train_bt['date'].max().date()})")
print(f"Test backtest  : {len(test_bt)} mois ({test_bt['date'].min().date()} -> "
      f"{test_bt['date'].max().date()})\n")

results_bt = []

# --- ARIMA ---
arima_order = find_best_arima_order(train_bt.set_index("date")[TARGET])
pred_arima_bt = forecast_arima(train_bt.set_index("date")[TARGET], BACKTEST_H, arima_order)
results_bt.append(evaluate_split(y_true_bt, pred_arima_bt, "ARIMA"))

# --- VAR ---
var_cols = [TARGET, "M3_growth", "Taux_Directeur_pct"]
print("VAR (backtest) :")
pred_var_bt = forecast_var(train_bt, var_cols, BACKTEST_H)
results_bt.append(evaluate_split(y_true_bt, pred_var_bt, "VAR"))

# --- Random Forest ---
X_train_bt, y_train_bt = train_bt[FEATURES], train_bt[TARGET]
rf_bt = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
rf_bt.fit(X_train_bt, y_train_bt)
pred_rf_bt = forecast_tree_model(rf_bt, train_bt, BACKTEST_H, test_dates_bt)
results_bt.append(evaluate_split(y_true_bt, pred_rf_bt, "Random Forest"))

# --- XGBoost ---
xgb_bt = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
xgb_bt.fit(X_train_bt, y_train_bt)
pred_xgb_bt = forecast_tree_model(xgb_bt, train_bt, BACKTEST_H, test_dates_bt)
results_bt.append(evaluate_split(y_true_bt, pred_xgb_bt, "XGBoost"))

# --- LSTM ---
full_series_bt = train_bt.set_index("date")[TARGET].values.reshape(-1, 1)
scaler_bt = MinMaxScaler()
scaled_bt = scaler_bt.fit_transform(full_series_bt)
lstm_bt = train_lstm(scaled_bt)
pred_lstm_scaled_bt = forecast_lstm(lstm_bt, scaled_bt[-LOOK_BACK:, 0], BACKTEST_H)
pred_lstm_bt = scaler_bt.inverse_transform(pred_lstm_scaled_bt.reshape(-1, 1)).flatten()
results_bt.append(evaluate_split(y_true_bt, pred_lstm_bt, "LSTM"))

# --- Naif simple ---
pred_naive_bt = forecast_naive(train_bt.set_index("date")[TARGET], BACKTEST_H)
results_bt.append(evaluate_split(y_true_bt, pred_naive_bt, "Naif"))

# --- Naif saisonnier ---
hist_dict_bt = train_bt.set_index("date")[TARGET].to_dict()
pred_naive_saison_bt = forecast_naive_seasonal(hist_dict_bt, test_dates_bt)
results_bt.append(evaluate_split(y_true_bt, pred_naive_saison_bt, "Naif_saisonnier"))

results_bt_df = pd.DataFrame(results_bt)

# --- MASE (Mean Absolute Scaled Error) : MAE du modele / MAE du Naif ---
# Contrairement au MAPE, le MASE ne souffre pas du probleme de division par
# une valeur reelle proche de zero. Lecture directe : <1 = bat le naif,
# >1 = moins bon que le naif, =1 = equivalent.
naive_mae_global = results_bt_df.loc[results_bt_df["Modele"] == "Naif", "MAE_global"].values[0]
naive_mae_court = results_bt_df.loc[results_bt_df["Modele"] == "Naif", "MAE_court"].values[0]
naive_mae_long = results_bt_df.loc[results_bt_df["Modele"] == "Naif", "MAE_long"].values[0]
results_bt_df["MASE_global"] = results_bt_df["MAE_global"] / naive_mae_global
results_bt_df["MASE_court"] = results_bt_df["MAE_court"] / naive_mae_court
results_bt_df["MASE_long"] = results_bt_df["MAE_long"] / naive_mae_long

results_bt_df.to_csv(os.path.join(OUTPUT_DIR, "resultats_backtest_court_vs_long.csv"), index=False)
print("\n=== TABLEAU BACKTEST (court vs long) ===")
print(results_bt_df[["Modele", "RMSE_court", "RMSE_long", "MAE_court", "MAE_long"]]
      .to_string(index=False))
print("\n=== MASE (< 1 = bat le naif, > 1 = moins bon que le naif) ===")
print(results_bt_df[["Modele", "MASE_global", "MASE_court", "MASE_long"]]
      .to_string(index=False))

MEILLEUR_MODELE_BT = results_bt_df.sort_values("RMSE_global").iloc[0]["Modele"]
print(f"\n-> Meilleur modele sur le backtest (RMSE global le plus bas) : {MEILLEUR_MODELE_BT}")

# --- Verdict explicite : chaque modele bat-il le naif simple ? ---
naive_rmse = results_bt_df.loc[results_bt_df["Modele"] == "Naif", "RMSE_global"].values[0]
print(f"\n[BENCHMARK] RMSE global du Naif simple = {naive_rmse:.3f} "
      f"(reference : un modele en dessous de ce chiffre a une vraie valeur predictive)")
for _, row in results_bt_df.sort_values("RMSE_global").iterrows():
    if row["Modele"] in ("Naif", "Naif_saisonnier"):
        continue
    verdict = "bat le naif" if row["RMSE_global"] < naive_rmse else "NE BAT PAS le naif"
    print(f"   {row['Modele']:15s} RMSE={row['RMSE_global']:.3f} -> {verdict}")


# ============================================================
# 5. PREVISION REELLE (refit sur TOUTES les donnees disponibles)
# ============================================================
print(f"\n=== PREVISION REELLE : {BACKTEST_H} mois, {future_dates_real[0].date()} "
      f"-> {future_dates_real[-1].date()} ===\n")

forecasts = pd.DataFrame({"date": future_dates_real})

forecasts["ARIMA"] = forecast_arima(df_ml.set_index("date")[TARGET], BACKTEST_H, arima_order)
print("VAR (prevision finale) :")
forecasts["VAR"] = forecast_var(df_ml, var_cols, BACKTEST_H)

X_full, y_full = df_ml[FEATURES], df_ml[TARGET]
rf_full = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42).fit(X_full, y_full)
forecasts["Random Forest"] = forecast_tree_model(rf_full, df_ml, BACKTEST_H, future_dates_real)

xgb_full = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         random_state=42).fit(X_full, y_full)
forecasts["XGBoost"] = forecast_tree_model(xgb_full, df_ml, BACKTEST_H, future_dates_real)

full_series = df_ml.set_index("date")[TARGET].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(full_series)
lstm_full = train_lstm(scaled)
pred_lstm_scaled = forecast_lstm(lstm_full, scaled[-LOOK_BACK:, 0], BACKTEST_H)
forecasts["LSTM"] = scaler.inverse_transform(pred_lstm_scaled.reshape(-1, 1)).flatten()

forecasts["Naif"] = forecast_naive(df_ml.set_index("date")[TARGET], BACKTEST_H)
hist_dict_full = df_ml.set_index("date")[TARGET].to_dict()
forecasts["Naif_saisonnier"] = forecast_naive_seasonal(hist_dict_full, future_dates_real)

forecasts.to_csv(os.path.join(OUTPUT_DIR, "previsions_futures_tous_modeles.csv"), index=False)
print(forecasts.head(6).to_string(index=False))

# ------------------------------------------------------------------
# 5bis. DEUX LIVRABLES SEPARES (chacun independant, pour usage/lecture distincts)
#    - Court terme : les 6 premiers mois de la prevision (usage operationnel)
#    - Long terme  : l'annee civile complete 2027 (comparaison avec le FMI)
# ------------------------------------------------------------------
forecast_court_terme = forecasts.iloc[:SHORT_HORIZON].copy()
forecast_court_terme.to_csv(os.path.join(OUTPUT_DIR, "prevision_court_terme_6mois.csv"), index=False)

forecast_long_terme_2027 = forecasts[forecasts["date"].dt.year == 2027].reset_index(drop=True)
forecast_long_terme_2027.to_csv(os.path.join(OUTPUT_DIR, "prevision_longue_2027.csv"), index=False)

print(f"\n[COURT TERME] {len(forecast_court_terme)} mois : "
      f"{forecast_court_terme['date'].min().date()} -> {forecast_court_terme['date'].max().date()}")
print(forecast_court_terme.to_string(index=False))

print(f"\n[LONG TERME] {len(forecast_long_terme_2027)} mois (annee 2027) : "
      f"{forecast_long_terme_2027['date'].min().date()} -> {forecast_long_terme_2027['date'].max().date()}")
print(forecast_long_terme_2027.to_string(index=False))

# --- Graphique dedie : court terme (zoom sur les 12 derniers mois connus + 6 mois prevus) ---
plt.figure(figsize=(10, 5))
hist_recent = df_ml[df_ml["date"] >= df_ml["date"].max() - pd.DateOffset(months=12)]
plt.plot(hist_recent["date"], hist_recent[TARGET], label="Inflation observee (BKAM)",
         color="black", linewidth=2, marker="o")
for col in MODELES:
    plt.plot(forecast_court_terme["date"], forecast_court_terme[col],
              label=f"Prevision {col}", linestyle="--", marker="o")
plt.axvline(df_ml["date"].max(), color="gray", linestyle=":", alpha=0.6)
plt.title(f"Prevision COURT TERME - {SHORT_HORIZON} mois "
          f"({forecast_court_terme['date'].min().strftime('%b %Y')} - "
          f"{forecast_court_terme['date'].max().strftime('%b %Y')})")
plt.xlabel("Date")
plt.ylabel("Inflation (%)")
plt.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "graphique_court_terme_6mois.png"), dpi=150)
plt.close()

# --- Graphique dedie : long terme (annee 2027 complete vs FMI) ---
plt.figure(figsize=(10, 5))
for col in MODELES:
    plt.plot(forecast_long_terme_2027["date"], forecast_long_terme_2027[col],
              label=f"Prevision {col}", linestyle="--", marker="o")
plt.axhline(FMI_2027, color="red", linestyle=":", linewidth=2,
            label=f"FMI 2027 ({FMI_2027}%, moyenne annuelle)")
plt.title("Prevision LONG TERME - Annee 2027 complete vs FMI")
plt.xlabel("Date")
plt.ylabel("Inflation (%)")
plt.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "graphique_long_terme_2027.png"), dpi=150)
plt.close()


# ============================================================
# 6. COMPARAISON FMI - DEUX METHODES + VALIDATION CROISEE 2026
#    A) Moyenne simple des taux glissants mensuels prevus (approximatif)
#    B) Vraie methode FMI (PCPIPCH) : ratio des moyennes ANNUELLES du
#       NIVEAU d'IPC, reconstruit a partir des taux glissants prevus.
#       IPC_reconstruit(t) = IPC_connu(t-12) * (1 + taux_prevu(t)/100)
#    VALIDATION : on recalcule aussi l'annee 2026 (melange donnees
#    reelles jan-mai 2026 + prevu juin-dec 2026) et on la compare au
#    vrai chiffre FMI 2026 (connu et publie). Si l'ecart est faible,
#    cela confirme que la methode de reconstruction est correcte.
# ============================================================
# Methode A : moyenne simple
mask_2027 = forecasts["date"].dt.year == 2027
moyennes_A = forecasts.loc[mask_2027, MODELES].mean()

# Methode B : reconstruction du niveau d'IPC (utilise df_ipc_full, qui
# contient aussi 2017, pour securiser tous les lookbacks a 12 mois)
ipc_hist = df_ipc_full.set_index("date")["IPC_General"].to_dict()

comparaison_fmi = []
for modele in MODELES:
    ipc_recon = dict(ipc_hist)   # copie : on va y ajouter les mois reconstruits
    for d, taux in zip(forecasts["date"], forecasts[modele]):
        base_date = d - pd.DateOffset(months=12)
        base_ipc = ipc_recon.get(base_date)
        if base_ipc is not None:
            ipc_recon[d] = base_ipc * (1 + taux / 100)

    annee_2026 = [v for k, v in ipc_recon.items() if k.year == 2026]
    annee_2025 = [v for k, v in ipc_recon.items() if k.year == 2025]
    annee_2027 = [v for k, v in ipc_recon.items() if k.year == 2027]

    infl_2026_recalc = ((np.mean(annee_2026) / np.mean(annee_2025) - 1) * 100
                         if len(annee_2026) == 12 and len(annee_2025) == 12 else np.nan)
    infl_methode_B = ((np.mean(annee_2027) / np.mean(annee_2026) - 1) * 100
                       if len(annee_2026) == 12 and len(annee_2027) == 12 else np.nan)

    comparaison_fmi.append({
        "Modele": modele,
        "Methode_A_moyenne_simple_2027": moyennes_A[modele],
        "Methode_B_ratio_IPC_annuel_2027": infl_methode_B,
        "Ecart_A_vs_FMI_2027": moyennes_A[modele] - FMI_2027,
        "Ecart_B_vs_FMI_2027": infl_methode_B - FMI_2027 if not np.isnan(infl_methode_B) else np.nan,
        "2026_recalcule_validation": infl_2026_recalc,
        "Ecart_validation_vs_FMI_2026": (infl_2026_recalc - FMI_2026
                                          if not np.isnan(infl_2026_recalc) else np.nan),
    })

comparaison_fmi_df = pd.DataFrame(comparaison_fmi)
comparaison_fmi_df.to_csv(os.path.join(OUTPUT_DIR, "comparaison_FMI_2027.csv"), index=False)

print("\n=== COMPARAISON 2027 : NOS MODELES VS FMI ===")
print(comparaison_fmi_df.to_string(index=False))
print(f"\nFMI (WEO avril 2026, methode officielle PCPIPCH) : "
      f"2026 = {FMI_2026}%  |  2027 = {FMI_2027}%")
print("\nMethode A = moyenne simple des taux glissants mensuels (approximation)")
print("Methode B = ratio des moyennes annuelles du niveau d'IPC (definition FMI exacte)")
print("\n[VALIDATION] Colonne '2026_recalcule_validation' = notre reconstruction de")
print("l'inflation 2026 (melange reel jan-mai + prevu juin-dec), comparee au vrai")
print(f"chiffre FMI 2026 ({FMI_2026}%). Un faible ecart valide notre methode de")
print("reconstruction de l'indice des prix utilisee pour comparer 2027 au FMI.")


# ============================================================
# 7. GRAPHIQUE RECAPITULATIF
# ============================================================
plt.figure(figsize=(12, 6))
plt.plot(df_ml["date"], df_ml[TARGET], label="Inflation observee (BKAM)",
         color="black", linewidth=2)
for col in MODELES:
    plt.plot(forecasts["date"], forecasts[col], label=f"Prevision {col}", linestyle="--")
plt.axhline(FMI_2027, color="red", linestyle=":", label=f"FMI 2027 ({FMI_2027}%)")
plt.axvline(df_ml["date"].max(), color="gray", linestyle=":", alpha=0.5)
plt.axvline(df_ml["date"].max() + pd.DateOffset(months=SHORT_HORIZON),
            color="orange", linestyle=":", alpha=0.5, label="Limite horizon court (6 mois)")
plt.title("Inflation au Maroc : observee et previsions par modele (2018-2027)")
plt.xlabel("Date")
plt.ylabel("Inflation (%)")
plt.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "graphique_previsions_inflation.png"), dpi=150)

print("\n=== TERMINE ===")
print("Fichiers generes :")
print(" - resultats_backtest_court_vs_long.csv (precision reelle testee sur donnees connues)")
print(" - previsions_futures_tous_modeles.csv  (mois par mois jusqu'a fin 2027)")
print(" - comparaison_FMI_2027.csv             (2 methodes + validation 2026, vs FMI)")
print(" - graphique_previsions_inflation.png")
