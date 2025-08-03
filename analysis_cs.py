#!/usr/bin/env python
# coding: utf-8
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use('TkAgg')  
import pandas as pd
import numpy as np
import scipy
import matplotlib
import statsmodels
import sklearn
import seaborn

print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)
print("scipy version:", scipy.__version__)
print("matplotlib version:", matplotlib.__version__)
print("statsmodels version:", statsmodels.__version__)
print("scikit-learn version:", sklearn.__version__)
print("seaborn version:", seaborn.__version__)


# Importing required packgages
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np. random.seed(42)
from scipy.stats import zscore, normaltest
from sklearn.metrics import r2_score

# Creating output packgages
os.makedirs("output/figures", exist_ok=True)
os.makedirs("output/tables", exist_ok=True)

# Loading temperature data
import os
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
temp_path = os.path.join(base_dir, "data", "raw", "temp_korea.csv")
df_temp = pd.read_csv(temp_path)
df_temp.columns = df_temp.columns.str.strip()
df_temp = df_temp.sort_values("BP").reset_index(drop=True)

# Importing required packages
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np. random.seed(42)
from scipy.stats import zscore, normaltest
from sklearn.metrics import r2_score

# Creating output packgages
os.makedirs("output/figures", exist_ok=True)
os.makedirs("output/tables", exist_ok=True)

# Calculating resid_z
df_temp["resid_z"] = zscore(df_temp["Temp"])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading data using exact relative path-
df = pd.read_csv('data/raw/temp_korea.csv')

# Preprocessing
df = df.rename(columns=str.strip)
df = df.dropna()
df['BP'] = df['BP'].astype(int)
df = df.sort_values('BP')

# Cheking
df.head()

# 5.1 Climate Stress Analysis Pipeline – the Korea peninsula
import os
import numpy as np
np. random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import zscore

# Importing data
# e.g.: df = pd.read_csv("your_file.csv")

# Sorting data
df = df.sort_values('BP', ascending=False).reset_index(drop=True)
x = df['BP'].values
y = df['Temp'].values

# Fitting second-degree polynomial regression model -
coeffs = np.polyfit(x, y, 2)
model = np.poly1d(coeffs)
y_pred = model(x)
r2 = r2_score(y, y_pred)

# Equation text
eq_text = f"$y = {coeffs[0]:.2e}x^2 + {coeffs[1]:.2f}x + {coeffs[2]:.2f}$\n$R^2 = {r2:.4f}$"

# Calculating residuals and reidual Z-score
residuals = y - y_pred
resid_z = zscore(residuals)
df['residual'] = residuals
df['resid_z'] = resid_z

# Saving residuals data
os.makedirs('output/tables', exist_ok=True)
df_out = df[['BP', 'Temp', 'residual', 'resid_z']]
df_out.to_csv('output/tables/table1_residual_zscore_korea.csv', index=False, encoding='utf-8-sig')

# Visualizing regression 
os.makedirs('output/figures', exist_ok=True)
plt.figure(figsize=(10, 5))
plt.plot(x, y, 'o', markersize=3, color='black', label='Observed Temp')
plt.plot(x, y_pred, '-', color='gray', linewidth=1.5, label='2nd-degree Fit')
plt.xlabel('Years BP', fontsize=11)
plt.ylabel('Temperature (°C)', fontsize=11)
plt.text(x=int(x.mean()), y=min(y) + 0.2, s=eq_text, fontsize=10, ha='center', va='top')
plt.gca().invert_xaxis()
plt.grid(alpha=0.3, linestyle='--', linewidth=0.5)
plt.legend(frameon=False, fontsize=9)
plt.tight_layout()

# Saving plots
plt.savefig('output/figures/fig1_temperature_regression_korea.eps', format='eps', dpi=300)
plt.savefig('output/figures/fig1_temperature_regression_korea.png', format='png', dpi=300)
plt.show()

# Checking the normality of residuals

from scipy.stats import shapiro, normaltest
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
np. random.seed(42)

# 5.1.1 Detecting long-term climate stress which Z-scores exceeded ±1.5 for at least two consecutive bins : the Korea peninsula
import os
import numpy as np
np. random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt

# Basic steup
DATA_PATH = "data/raw/temp_korea.csv"
FIG_PATH_PNG = "output/figures/fig_s1_stress_clusters_1.5_korea.png"
FIG_PATH_EPS = "output/figures/fig_s1_stress_clusters_1.5_korea.eps"
TABLE_PATH = "output/tables/table_stress_clusters_1.5_korea.csv"

# Creating output packages
os.makedirs("output/figures", exist_ok=True)
os.makedirs("output/tables", exist_ok=True)

# Loading and sorting data
df = pd.read_csv(DATA_PATH)
df = df.rename(columns=str.strip)
df = df.sort_values("BP").reset_index(drop=True)

# calcuating residual_Z score
df["resid_z"] = (df["Temp"] - df["Temp"].mean()) / df["Temp"].std()

# Detecting cluster with Z-scores exceeding ±1.5 for at least two consecutive bins 
def detect_stress_clusters(series, threshold=1.5, min_duration=2):
    flags = series.abs() >= threshold
    group_id = 1
    group_list = []
    count = 0

    for flag in flags:
        if flag:
            count += 1
            group_list.append(group_id)
        else:
            if count >= min_duration:
                group_id += 1
            else:
                group_list[-count:] = [0] * count
                group_id += 1
            group_list.append(0)
            count = 0

    if count < min_duration:
        group_list[-count:] = [0] * count

    while len(group_list) < len(series):
        group_list.append(0)

    return pd.Series(group_list, index=series.index)

# Running Detection
df["cluster_all_1.5"] = detect_stress_clusters(df["resid_z"], threshold=1.5, min_duration=2)

# Checking for the existence of clusters and saving the results
df_cluster = df[df["cluster_all_1.5"] > 0][["BP", "Temp", "resid_z", "cluster_all_1.5"]]

if not df_cluster.empty:
    df_cluster.to_csv(TABLE_PATH, index=False, encoding="utf-8-sig")

# Visualizing clusters
plt.figure(figsize=(10, 4))
plt.plot(df["BP"], df["resid_z"], color="black", linewidth=1)

# Setting baseline
plt.axhline(1.5, linestyle="--", color="black", linewidth=0.8)
plt.axhline(-1.5, linestyle="--", color="black", linewidth=0.8)

# Shading emphasized clusters
for cid in df["cluster_all_1.5"].unique():
    if cid > 0:
        segment = df[df["cluster_all_1.5"] == cid]
        plt.axvspan(segment["BP"].max(), segment["BP"].min(), color="orange", alpha=0.3)

plt.gca().invert_xaxis()
plt.xlabel("BP")
plt.ylabel("Z-score")
plt.tight_layout()
plt.savefig(FIG_PATH_PNG, dpi=300)
plt.savefig(FIG_PATH_EPS, format="eps", dpi=300)
plt.show()

# 5.1.1 Detecting long-term climate stress which Z-scores exceeded ±1.0 for at least two consecutive bins (relaxed magnitide): the Korea peninsula
import pandas as pd
import numpy as np
np. random.seed(42)

def detect_stress_clusters_signed(series, threshold, min_duration=2, direction="both"):
    if direction == "positive":
        flags = series >= threshold
    elif direction == "negative":
        flags = series <= -threshold
    else:  # both
        flags = series.abs() >= threshold

    group_id = 1
    group_list = [0] * len(series)
    count = 0
    start_idx = None

    for i, flag in enumerate(flags):
        if flag:
            if count == 0:
                start_idx = i
            count += 1
        else:
            if count >= min_duration:
                for j in range(start_idx, i):
                    group_list[j] = group_id
                group_id += 1
            count = 0
            start_idx = None

    # Processing last cluster
    if count >= min_duration:
        for j in range(start_idx, len(series)):
            group_list[j] = group_id

    return pd.Series(group_list, index=series.index)

df["cluster_pos_1.5"] = detect_stress_clusters_signed(df["resid_z"], threshold=1.5, min_duration=2, direction="positive")
df["cluster_neg_1.5"] = detect_stress_clusters_signed(df["resid_z"], threshold=1.5, min_duration=2, direction="negative")
df["cluster_all_1.5"] = detect_stress_clusters_signed(df["resid_z"], threshold=1.5, min_duration=2, direction="both")

df["cluster_pos_1.0"] = detect_stress_clusters_signed(df["resid_z"], threshold=1.0, min_duration=2, direction="positive")
df["cluster_neg_1.0"] = detect_stress_clusters_signed(df["resid_z"], threshold=1.0, min_duration=2, direction="negative")
df["cluster_all_1.0"] = detect_stress_clusters_signed(df["resid_z"], threshold=1.0, min_duration=2, direction="both")

print("▶ cluster_all_1.0 distribution:\n", df["cluster_all_1.0"].value_counts())
print("▶ ΔZ ≥ ±1.0 & duration ≥ 2 cluster:")
print(df[df["cluster_all_1.0"] > 0][["BP", "resid_z", "cluster_all_1.0"]])

import os

# Importing output packages
os.makedirs("output/tables", exist_ok=True)

# Saving cluster_all_1.0 > 0
df[df["cluster_all_1.0"] > 0][["BP", "Temp", "resid_z", "cluster_all_1.0"]].to_csv(
    "output/tables/climate_stress_clusters_1.0_korea.csv",
    index=False,
    encoding="utf-8-sig"
)

import matplotlib.pyplot as plt

# Creating visualization directory
os.makedirs("output/figures", exist_ok=True)

# Plots
plt.figure(figsize=(10, 4))
plt.plot(df["BP"], df["resid_z"], color="black", linewidth=1)
plt.axhline(1.0, linestyle="--", color="black", linewidth=0.8)
plt.axhline(-1.0, linestyle="--", color="black", linewidth=0.8)

# Emphasizing cluster areas
for cid in df["cluster_all_1.0"].unique():
    if cid > 0:
        seg = df[df["cluster_all_1.0"] == cid]
        plt.axvspan(seg["BP"].max(), seg["BP"].min(), color="orange", alpha=0.3)

# Setting axis limits
plt.gca().invert_xaxis()
plt.xlabel("BP")
plt.ylabel("Z-score")
plt.tight_layout()

# Saving plots
plt.savefig("output/figures/fig_s2_zscore_stress_clusters_1.0_korea.png", dpi=300)
plt.savefig("output/figures/fig_s2_zscore_stress_clusters_1.0_korea.eps", format="eps")
plt.show()

# 5.1.2 Detecting Short-Term Shock Detection via ΔZ : the Korea peninsula
import pandas as pd
np. random.seed(42)
import matplotlib.pyplot as plt
import os
from scipy.stats import shapiro, normaltest

# Setting paths
INPUT_PATH = "output/tables/table1_residual_zscore_korea.csv"
OUTPUT_TABLE_SPOT = "output/tables/table_spot_delta_residz_korea.csv"
OUTPUT_TABLE_FULL = "output/tables/table2_residual_dz_zscore_korea.csv"
OUTPUT_FIG_PNG = "output/figures/fig_s3_spot_delta_residz_korea.png"
OUTPUT_FIG_EPS = "output/figures/fig_s3_spot_delta_residz_korea.eps"

# Roading and sorting data
df = pd.read_csv(INPUT_PATH)
df = df.sort_values("BP", ascending=True).reset_index(drop=True)

# Caculating Δresidual Z 
df["delta_residz"] = df["resid_z"].diff()

# Checking normality of Δresidual
delta_residz_clean = df["delta_residz"].dropna()

# Shapiro-Wilk
shapiro_stat, shapiro_p = shapiro(delta_residz_clean)

# D’Agostino K²
dagostino_stat, dagostino_p = normaltest(delta_residz_clean)

print(f"[Shapiro-Wilk] W = {shapiro_stat:.4f}, p = {shapiro_p:.4f}")
print(f"[D’Agostino K²] stat = {dagostino_stat:.4f}, p = {dagostino_p:.4f}")

# Δ Residual → Z-score transformation
mu = delta_residz_clean.mean()
sigma = delta_residz_clean.std()
df["delta_zscore"] = (df["delta_residz"] - mu) / sigma

# Extracting spots where exceeding Δresidual Z exceeds ±1.5
THRESHOLD = 1.5
df_spot = df[df["delta_zscore"].abs() >= THRESHOLD]
df_spot.to_csv(OUTPUT_TABLE_SPOT, index=False, encoding="utf-8-sig")

# Soring in descending order and Saving all
df_desc = df.sort_values("BP", ascending=False).reset_index(drop=True)
df_desc.to_csv(OUTPUT_TABLE_FULL, index=False, encoding="utf-8-sig")

# Visualizing
plt.figure(figsize=(10, 4))
plt.plot(df["BP"], df["delta_zscore"], color="gray", linewidth=1.2, label="Δ Residual Z-score")
plt.scatter(df_spot["BP"], df_spot["delta_zscore"], color="red", s=40, zorder=5, label="ΔZ ≥ ±1.5")
plt.axhline(THRESHOLD, linestyle="--", color="red", linewidth=0.8)
plt.axhline(-THRESHOLD, linestyle="--", color="red", linewidth=0.8)
plt.xlabel("BP")
plt.ylabel("Δ Residual Z-score")
plt.gca().invert_xaxis()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.legend(frameon=False)

# Saving plots
os.makedirs("output/figures", exist_ok=True)
plt.savefig(OUTPUT_FIG_PNG, dpi=300)
plt.savefig(OUTPUT_FIG_EPS, format="eps", dpi=300)
plt.show()

# 5.1.3 GeneratingSPD Data and Processing : the Korea peninsula
import pandas as pd
import numpy as np
np. random.seed(42)
import matplotlib.pyplot as plt
import os

# Setting paths
DATA_PATH = "data/raw/raw_spd_korea.csv"
OUTPUT_PATH_CSV = "data/processed/spd_korea_20yr.csv"
OUTPUT_PATH_PNG = "output/figures/fig_s4_spd_korea.png"
OUTPUT_PATH_EPS = "output/figures/fig_s4_spd_korea.eps"

N_SIM = 1000
BIN_WIDTH = 20
YEAR_MIN, YEAR_MAX = 2100, 6100  # Defining the reference range of BP 

# Creating output packages
os.makedirs("data/processed", exist_ok=True)
os.makedirs("output/figures", exist_ok=True)

# Loading data
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df["BP_central"] = df["BP"].str.extract(r"(\d+)\s*±\s*\d+").astype(float)
df["BP_error"] = df["BP"].str.extract(r"±\s*(\d+)").astype(float)
df = df.dropna(subset=["BP_central", "BP_error"])

# Calculating SPD(Performing probabilistic simulations, generating annual time series)
years = np.arange(YEAR_MIN, YEAR_MAX + 1)
spd_accumulate = np.zeros_like(years, dtype=float)

for _ in range(N_SIM):
    simulated_dates = np.random.normal(df["BP_central"], df["BP_error"])
    counts, _ = np.histogram(simulated_dates, bins=np.append(years, YEAR_MAX + 1))
    spd_accumulate += counts

spd = spd_accumulate / N_SIM  # average SPD

# Aggregating annual results into 20-year bins
df_spd = pd.DataFrame({"BP": years, "SPD": spd})
df_spd["CRA_bin"] = (df_spd["BP"] // BIN_WIDTH) * BIN_WIDTH
spd_bin = df_spd.groupby("CRA_bin")["SPD"].sum().reset_index()
spd_bin = spd_bin.sort_values("CRA_bin", ascending=False)

# Saving results
spd_bin.to_csv(OUTPUT_PATH_CSV, index=False, encoding="utf-8-sig")

# Visualizing
plt.figure(figsize=(10, 4))
plt.plot(spd_bin["CRA_bin"], spd_bin["SPD"], color="black", linewidth=1)
plt.xlabel("BP")
plt.ylabel("SPD")
plt.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig(OUTPUT_PATH_PNG, dpi=300)
plt.savefig(OUTPUT_PATH_EPS, format="eps", dpi=300)
plt.show()
print(f"Extracting the number of valid samples: {len(df)}")


import pandas as pd
np. random.seed(42)
import matplotlib.pyplot as plt
import os

# Preventing EPS save errors
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# Setting paths
INPUT_PATH = "data/processed/spd_korea_20yr.csv"
OUTPUT_CSV = "data/processed/spd_korea_z_m3.csv"
OUTPUT_PNG = "output/figures/fig_s5_spd_korea_z_m3.png"
OUTPUT_EPS = "output/figures/fig_s5_spd_korea_z_m3.eps"

# Loading data
df = pd.read_csv(INPUT_PATH)
df = df.sort_values("CRA_bin", ascending=False).reset_index(drop=True)

# Transforming SPD into SPD Z-score 
df["SPD_Z"] = (df["SPD"] - df["SPD"].mean()) / df["SPD"].std()

# Moving average M3  (3 bin = 60 years)
df["SPD_M3"] = df["SPD"].rolling(window=3, center=True).mean()

# Saving
os.makedirs("data/processed", exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

# Visualizing
plt.figure(figsize=(10, 4))
plt.plot(df["CRA_bin"], df["SPD_Z"], color="black", linewidth=1.2, label="SPD Z-score")
plt.plot(df["CRA_bin"], df["SPD_M3"], color="gray", linewidth=1.0, linestyle="--", label="SPD M3 (60yr)")
plt.axhline(0, color="black", linestyle=":", linewidth=0.6)
plt.xlabel("BP")
plt.ylabel("Z-score / M3")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(frameon=False)
plt.gca().invert_xaxis()
plt.tight_layout()

# Saving plots
plt.savefig(OUTPUT_PNG, dpi=300)
plt.savefig(OUTPUT_EPS, format='eps', dpi=300)
plt.show()

# 5.1.4 OverlayIing Climate Stress and SPD in the Korean Peninsula
import pandas as pd
np. random.seed(42)
import matplotlib.pyplot as plt
import os

# Setting paths
SPD_PATH = "data/processed/spd_korea_z_m3.csv"
CLUSTER_PATH = "output/tables/climate_stress_clusters_1.0_korea.csv"
DZ_PATH = "output/tables/table2_residual_dz_zscore_korea.csv"
OUTPUT_TABLE_SPOT = "output/tables/table_spot_delta_residz_korea.csv"
FIG_PATH_PNG = "output/figures/fig2_spd_z_climate_overlay_korea.png"
FIG_PATH_EPS = "output/figures/fig2_spd_z_climate_overlay_korea.eps"

# Loading data
df_spd = pd.read_csv(SPD_PATH)
df_clust = pd.read_csv(CLUSTER_PATH)
df_dz = pd.read_csv(DZ_PATH)

# Sorting in descending order
df_spd = df_spd.sort_values("CRA_bin", ascending=False)
df_clust = df_clust.sort_values("BP", ascending=False)
df_dz = df_dz.sort_values("BP", ascending=False)

# Visualizing 
plt.figure(figsize=(10, 5))

# SPD Z-score (Solid black line)
plt.plot(df_spd["CRA_bin"], df_spd["SPD_Z"], color="black", linewidth=1.2, label="SPD Z-score")

# Δresidual Z-score (solid grey line)
plt.plot(df_dz["BP"], df_dz["delta_zscore"], color="gray", linewidth=1.0, label="Δ Residual Z-score")

# ΔZ ≥ ±1.5 spots (red dots)
df_spot = df_dz[df_dz["delta_zscore"].abs() >= 1.5]
plt.scatter(df_spot["BP"], df_spot["delta_zscore"], color="red", s=30, zorder=5, label="ΔZ ≥ ±1.5")

# long-term stress(oragne band)
for cid in df_clust["cluster_all_1.0"].unique():
    if cid > 0:
        seg = df_clust[df_clust["cluster_all_1.0"] == cid]
        plt.axvspan(seg["BP"].max(), seg["BP"].min(), color="orange", alpha=0.3)

# Summarizing plot style
plt.axhline(0, linestyle=":", color="black", linewidth=0.7)
plt.axhline(1.5, linestyle="--", color="red", linewidth=0.6)
plt.axhline(-1.5, linestyle="--", color="red", linewidth=0.6)
plt.xlabel("BP")
plt.ylabel("Z-score")
plt.gca().invert_xaxis()
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(frameon=False)
plt.tight_layout()

# Saving
os.makedirs("output/figures", exist_ok=True)
plt.savefig(FIG_PATH_PNG, dpi=300)
plt.savefig(FIG_PATH_EPS, format="eps", dpi=300)
plt.show()

# 5.2 Climate Stress Analysis Pipeline – Southen Japan
import os
import numpy as np
np. random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import zscore

# Loading and sorting data
df_temp = pd.read_csv('data/raw/temp_japan.csv')  
df_temp = df_temp.sort_values("BP").reset_index(drop=True)

# Setting variables
x = df_temp['BP'].values
y = df_temp['Temp'].values

# Fitting a second-degree polynomial regression
coeffs = np.polyfit(x, y, 2)
model = np.poly1d(coeffs)
y_pred = model(x)
r2 = r2_score(y, y_pred)

# Equation test
eq_text = f"$y = {coeffs[0]:.2e}x^2 + {coeffs[1]:.2f}x + {coeffs[2]:.2f}$\n$R^2 = {r2:.4f}$"

# Calculating Residuals and residal Z-score
residuals = y - y_pred
resid_z = zscore(residuals)
df_temp['residual'] = residuals
df_temp['resid_z'] = resid_z

# Saving results
os.makedirs('output/tables', exist_ok=True)
df_temp[['BP', 'Temp', 'residual', 'resid_z']].to_csv('output/tables/table1_residual_zscore_japan.csv', index=False, encoding='utf-8-sig')

# Visualizing
os.makedirs('output/figures', exist_ok=True)
plt.figure(figsize=(10, 5))
plt.plot(x, y, 'o', markersize=3, color='black', label='Observed Temp')
plt.plot(x, y_pred, '-', color='gray', linewidth=1.5, label='2nd-degree Fit')
plt.xlabel('Years BP', fontsize=11)
plt.ylabel('Temperature (°C)', fontsize=11)
plt.text(int(x.mean()), min(y) + 0.2, eq_text, fontsize=10, ha='center', va='top')
plt.gca().invert_xaxis()
plt.grid(alpha=0.3, linestyle='--', linewidth=0.5)
plt.legend(frameon=False, fontsize=9)
plt.tight_layout()

#  Saving plots
plt.savefig('output/figures/fig3_temperature_regression_japan.eps', format='eps', dpi=300)
plt.savefig('output/figures/fig3_temperature_regression_japan.png', format='png', dpi=300)
plt.show()


# Checking the normality of residuals
from scipy.stats import shapiro, normaltest
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
np. random.seed(42)

# Shapiro-Wilk Test (n < 5000)
shapiro_stat, shapiro_p = shapiro(residuals)
print(f"Shapiro–Wilk p-value: {shapiro_p:.4f}")

# D'Agostino and Pearson test
dag_stat, dag_p = normaltest(residuals)
print(f"D’Agostino–Pearson p-value: {dag_p:.4f}")

# 5.2.1 Detecting long-term climate stress which Z-scores exceeded ±1.5 for at least two consecutive bins: southen Japan
import pandas as pd
np. random.seed(42)
import matplotlib.pyplot as plt
import os

# Setting path
DATA_PATH = "data/raw/temp_japan.csv"
FIG_PATH_PNG = "output/figures/fig_s6_stress_clusters_1.5_japan.png"
FIG_PATH_EPS = "output/figures/fig_s6_stress_clusters_1.5_japan.eps"
TABLE_PATH = "output/tables/table_stress_clusters_1.5_japan.csv"

# Creating outpackages
os.makedirs("output/figures", exist_ok=True)
os.makedirs("output/tables", exist_ok=True)

# Loading and sorting data
df = pd.read_csv(DATA_PATH)
df = df.rename(columns=str.strip)
df = df.sort_values("BP").reset_index(drop=True)

# calcuating residual Z-score
df["resid_z"] = (df["Temp"] - df["Temp"].mean()) / df["Temp"].std()

def detect_stress_clusters(series, threshold=1.5, min_duration=2):
    flags = series.abs() >= threshold
    group_id = 1
    group_list = []
    count = 0

    for flag in flags:
        if flag:
            count += 1
            group_list.append(group_id)
        else:
            if count >= min_duration:
                group_id += 1
            else:
                group_list[-count:] = [0] * count
                group_id += 1
            group_list.append(0)
            count = 0

    if count < min_duration:
        group_list[-count:] = [0] * count

    while len(group_list) < len(series):
        group_list.append(0)

    return pd.Series(group_list, index=series.index)

# Detecting cluster with Z-scores exceeding ±1.5 for at least two consecutive bins 
df["cluster_all_1.5"] = detect_stress_clusters(df["resid_z"], threshold=1.5, min_duration=2)

# Checking for the existence of clusters and saving the results
df_cluster = df[df["cluster_all_1.5"] > 0][["BP", "Temp", "resid_z", "cluster_all_1.5"]]

if not df_cluster.empty:
    df_cluster.to_csv(TABLE_PATH, index=False, encoding="utf-8-sig")

# Visualizing
plt.figure(figsize=(10, 4))
plt.plot(df["BP"], df["resid_z"], color="black", linewidth=1)

# Setting baseline
plt.axhline(1.5, linestyle="--", color="black", linewidth=0.8)
plt.axhline(-1.5, linestyle="--", color="black", linewidth=0.8)

# hading emphasized clusters
for cid in df["cluster_all_1.5"].unique():
    if cid > 0:
        segment = df[df["cluster_all_1.5"] == cid]
        plt.axvspan(segment["BP"].max(), segment["BP"].min(), color="orange", alpha=0.3)

plt.gca().invert_xaxis()
plt.xlabel("BP")
plt.ylabel("Z-score")
plt.tight_layout()
plt.savefig(FIG_PATH_PNG, dpi=300)
plt.savefig(FIG_PATH_EPS, format="eps", dpi=300)
plt.show()

# 5.2.1 Detecting long-term climate stress which Z-scores exceeded ±1.0 for at least two consecutive bins (relaxed magnitide): southen Japan

import pandas as pd
import numpy as np
np. random.seed(42)

def detect_stress_clusters_signed(series, threshold, min_duration=2, direction="both"):
    if direction == "positive":
        flags = series >= threshold
    elif direction == "negative":
        flags = series <= -threshold
    else:  # both
        flags = series.abs() >= threshold

    group_id = 1
    group_list = [0] * len(series)
    count = 0
    start_idx = None

    for i, flag in enumerate(flags):
        if flag:
            if count == 0:
                start_idx = i
            count += 1
        else:
            if count >= min_duration:
                for j in range(start_idx, i):
                    group_list[j] = group_id
                group_id += 1
            count = 0
            start_idx = None

    # Last cluster
    if count >= min_duration:
        for j in range(start_idx, len(series)):
            group_list[j] = group_id

    return pd.Series(group_list, index=series.index)

df["cluster_pos_1.5"] = detect_stress_clusters_signed(df["resid_z"], threshold=1.5, min_duration=2, direction="positive")
df["cluster_neg_1.5"] = detect_stress_clusters_signed(df["resid_z"], threshold=1.5, min_duration=2, direction="negative")
df["cluster_all_1.5"] = detect_stress_clusters_signed(df["resid_z"], threshold=1.5, min_duration=2, direction="both")

df["cluster_pos_1.0"] = detect_stress_clusters_signed(df["resid_z"], threshold=1.0, min_duration=2, direction="positive")
df["cluster_neg_1.0"] = detect_stress_clusters_signed(df["resid_z"], threshold=1.0, min_duration=2, direction="negative")
df["cluster_all_1.0"] = detect_stress_clusters_signed(df["resid_z"], threshold=1.0, min_duration=2, direction="both")

print("▶ cluster_all_1.0 distibution:\n", df["cluster_all_1.0"].value_counts())
print("▶ ΔZ ≥ ±1.0 & duration ≥ 2 cluster:")
print(df[df["cluster_all_1.0"] > 0][["BP", "resid_z", "cluster_all_1.0"]])

import os

# Creating Output packages
os.makedirs("output/tables", exist_ok=True)

# Saving cluster_all_1.0 > 0
df[df["cluster_all_1.0"] > 0][["BP", "Temp", "resid_z", "cluster_all_1.0"]].to_csv(
    "output/tables/climate_stress_clusters_1.0_japan.csv",
    index=False,
    encoding="utf-8-sig"
)

import matplotlib.pyplot as plt

# Creating visualization directory
os.makedirs("output/figures", exist_ok=True)

# Plots
plt.figure(figsize=(10, 4))
plt.plot(df["BP"], df["resid_z"], color="black", linewidth=1)
plt.axhline(1.0, linestyle="--", color="black", linewidth=0.8)
plt.axhline(-1.0, linestyle="--", color="black", linewidth=0.8)

# Emphasizing cluster areas
for cid in df["cluster_all_1.0"].unique():
    if cid > 0:
        seg = df[df["cluster_all_1.0"] == cid]
        plt.axvspan(seg["BP"].max(), seg["BP"].min(), color="orange", alpha=0.3)

# Setting axis limits
plt.gca().invert_xaxis()
plt.xlabel("BP")
plt.ylabel("Z-score")
plt.tight_layout()

# Saving plots
plt.savefig("output/figures/fig_s7_zscore_stress_clusters_1.0_japan.png", dpi=300)
plt.savefig("output/figures/fig_s7_zscore_stress_clusters_1.0_japan.eps", format="eps")
plt.show()

# 5.2.2 Detecting Short-Term Shock Detection (Δ residual Z ): southen Japan
import pandas as pd
np. random.seed(42)
import matplotlib.pyplot as plt
import os
from scipy.stats import shapiro, normaltest
np.random.seed(42)

# Setting path 
INPUT_PATH = "output/tables/table1_residual_zscore_japan.csv"
OUTPUT_TABLE_SPOT = "output/tables/table_spot_delta_residz_japan.csv"
OUTPUT_TABLE_FULL = "output/tables/table2_residual_dz_zscore_japan.csv"
OUTPUT_FIG_PNG = "output/figures/fig_s8_spot_delta_residz_japan.png"
OUTPUT_FIG_EPS = "output/figures/fig_s8_spot_delta_residz_japan.eps"

# Loading and sorting data
df = pd.read_csv(INPUT_PATH)
df = df.sort_values("BP", ascending=True).reset_index(drop=True)

# calcuating Δresiduals 
df["delta_residz"] = df["resid_z"].diff()

# Checking the normality of Δresiduals
delta_residz_clean = df["delta_residz"].dropna()

# Shapiro-Wilk
shapiro_stat, shapiro_p = shapiro(delta_residz_clean)

# D’Agostino K²
dagostino_stat, dagostino_p = normaltest(delta_residz_clean)

print(f"[Shapiro-Wilk] W = {shapiro_stat:.4f}, p = {shapiro_p:.4f}")
print(f"[D’Agostino K²] stat = {dagostino_stat:.4f}, p = {dagostino_p:.4f}")

# Transformating Δresiduals into Z-score
mu = delta_residz_clean.mean()
sigma = delta_residz_clean.std()
df["delta_zscore"] = (df["delta_residz"] - mu) / sigma

# Extracting spots where exceeding Δresidual Z exceeds ±1.5
THRESHOLD = 1.5
df_spot = df[df["delta_zscore"].abs() >= THRESHOLD]
df_spot.to_csv(OUTPUT_TABLE_SPOT, index=False, encoding="utf-8-sig")

# Soring in descending order and Saving all
df_desc = df.sort_values("BP", ascending=False).reset_index(drop=True)
df_desc.to_csv(OUTPUT_TABLE_FULL, index=False, encoding="utf-8-sig")

# Visualizing
plt.figure(figsize=(10, 4))
plt.plot(df["BP"], df["delta_zscore"], color="gray", linewidth=1.2, label="Δ Residual Z-score")
plt.scatter(df_spot["BP"], df_spot["delta_zscore"], color="red", s=40, zorder=5, label="ΔZ ≥ ±1.5")
plt.axhline(THRESHOLD, linestyle="--", color="red", linewidth=0.8)
plt.axhline(-THRESHOLD, linestyle="--", color="red", linewidth=0.8)
plt.xlabel("BP")
plt.ylabel("Δ Residual Z-score")
plt.gca().invert_xaxis()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.legend(frameon=False)

# Saving plots
os.makedirs("output/figures", exist_ok=True)
plt.savefig(OUTPUT_FIG_PNG, dpi=300)
plt.savefig(OUTPUT_FIG_EPS, format="eps", dpi=300)
plt.show()

# 5.2.3 GeneratingSPD Data and Processing : southen Japan
import pandas as pd
import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
import os

# Setting path
DATA_PATH = "data/raw/raw_spd_japan.csv"
OUTPUT_PATH_CSV = "data/processed/spd_japan_20yr.csv"
OUTPUT_PATH_PNG = "output/figures/fig_s9_spd_japan.png"
OUTPUT_PATH_EPS = "output/figures/fig_s9_spd_japan.eps"

# Setting
FILE_PATH = "data/raw/raw_spd_japan.csv"
OUTPUT_PATH = "data/processed/spd_japan_20yr.csv"

# Loading data
df = pd.read_csv(FILE_PATH, encoding='ISO-8859-1', low_memory=False)
df['CRA'] = pd.to_numeric(df['CRA'], errors='coerce')
df['CRAError'] = pd.to_numeric(df['CRAError'], errors='coerce')
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

BP_MIN = 2100
BP_MAX = 6000

# Fitering CRA Range
df = df[(df['CRA'] >= BP_MIN) & (df['CRA'] <= BP_MAX)]

# Supplementing latitude and longitude based on site names
coord_lookup = df.dropna(subset=['Latitude', 'Longitude']) \
    .groupby('SiteNameEn')[['Latitude', 'Longitude']].first()

df[['Latitude', 'Longitude']] = df.apply(
    lambda row: coord_lookup.loc[row['SiteNameEn']]
    if pd.isna(row['Latitude']) and row['SiteNameEn'] in coord_lookup.index
    else row[['Latitude', 'Longitude']],
    axis=1, result_type='expand'
)

# Filtering by latitude and removing missing values
df = df.dropna(subset=['Latitude', 'CRA', 'CRAError'])
df = df[(df['Latitude'] >= 29.0) & (df['Latitude'] <= 32.0)]

# Calculating SPD(Performing probabilistic simulations, generating annual time series)
years = np.arange(BP_MIN, BP_MAX + 1)
spd_accum = np.zeros_like(years, dtype=float)

for _ in range(N_SIM):
    simulated_dates = np.random.normal(loc=df['CRA'], scale=df['CRAError'])
    counts, _ = np.histogram(simulated_dates, bins=np.append(years, BP_MAX + 1))
    spd_accum += counts

spd_mean = spd_accum / N_SIM

# Aggregating annual results into 20-year bins
df_spd = pd.DataFrame({'BP': years, 'SPD': spd_mean})
df_spd['CRA_bin'] = (df_spd['BP'] // BIN_WIDTH) * BIN_WIDTH
spd_bin = df_spd.groupby('CRA_bin')['SPD'].sum().reset_index()
spd_bin = spd_bin.sort_values(by='CRA_bin', ascending=False)

# Saving results
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
spd_bin.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

# Visualizing
plt.figure(figsize=(10, 4))
plt.plot(spd_bin['CRA_bin'], spd_bin['SPD'], color='black', linewidth=1)
plt.xlabel("BP")
plt.ylabel("Simulated SPD")
plt.ylim(0, spd_bin['SPD'].max() * 1.1) 
plt.grid(True, linestyle='--', alpha=0.5, linewidth=0.4)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig(OUTPUT_PATH_PNG, dpi=300)
plt.savefig(OUTPUT_PATH_EPS, format='eps', dpi=300)
plt.show()

N_SIM = 1000
BIN_WIDTH = 20
YEAR_MIN, YEAR_MAX = 2100, 6000 
print(f"Extracting the number of valid samples: {len(df)}")

import pandas as pd
np.random.seed(42)
import matplotlib.pyplot as plt
import os

# Preventing EPS save errors
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# Setting path
INPUT_PATH = "data/processed/spd_japan_20yr.csv"
OUTPUT_CSV = "data/processed/spd_japan_z_m3.csv"
OUTPUT_PNG = "output/figures/fig_s10_spd_japan_z_m3.png"
OUTPUT_EPS = "output/figures/fig_s10_spd_japan_z_m3.eps"

# Loading data
df = pd.read_csv(INPUT_PATH)
df = df.sort_values("CRA_bin", ascending=False).reset_index(drop=True)

# Transforming SPD into SPD Z-score 
df["SPD_Z"] = (df["SPD"] - df["SPD"].mean()) / df["SPD"].std()

# Moving average M3 (3 bin = 60 years)
df["SPD_M3"] = df["SPD"].rolling(window=3, center=True).mean()

# Saving
os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_EPS), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

# Visualizing
plt.figure(figsize=(10, 4))
plt.plot(df["CRA_bin"], df["SPD_Z"], color="black", linewidth=1.2, label="SPD Z-score")
plt.plot(df["CRA_bin"], df["SPD_M3"], color="gray", linewidth=1.0, linestyle="--", label="SPD M3 (60yr)")
plt.axhline(0, color="black", linestyle=":", linewidth=0.6)
plt.xlabel("BP")
plt.ylabel("Z-score / M3")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(frameon=False)
plt.gca().invert_xaxis()
plt.tight_layout()

# Saving plots
plt.savefig(OUTPUT_PNG, dpi=300)
plt.savefig(OUTPUT_EPS, format='eps', dpi=300)
plt.show()

# 5.1.4 OverlayIing Climate Stress and SPD in southe Japan
import pandas as pd
np.random.seed(42)
import matplotlib.pyplot as plt
import os

# Setting Path
SPD_PATH = "data/processed/spd_japan_z_m3.csv"
CLUSTER_PATH = "output/tables/climate_stress_clusters_1.0_japan.csv"
DZ_PATH = "output/tables/table2_residual_dz_zscore_japan.csv"
OUTPUT_TABLE_SPOT = "output/tables/table_spot_delta_residz_japan.csv"
FIG_PATH_PNG = "output/figures/fig4_spd_z_climate_overlay_japan.png"
FIG_PATH_EPS = "output/figures/fig4_spd_z_climate_overlay_japan.eps"

# Loading data
df_spd = pd.read_csv(SPD_PATH)
df_clust = pd.read_csv(CLUSTER_PATH)
df_dz = pd.read_csv(DZ_PATH)

# # Sorting in descending order
df_spd = df_spd.sort_values("CRA_bin", ascending=False)
df_clust = df_clust.sort_values("BP", ascending=False)
df_dz = df_dz.sort_values("BP", ascending=False)

# Visualizing
plt.figure(figsize=(10, 5))

# SPD Z-score (Solid black line)
plt.plot(df_spd["CRA_bin"], df_spd["SPD_Z"], color="black", linewidth=1.2, label="SPD Z-score")

# Δresidual Z-score (solid grey line)
plt.plot(df_dz["BP"], df_dz["delta_zscore"], color="gray", linewidth=1.0, label="Δ Residual Z-score")

# ΔZ ≥ ±1.5 spots (red dots)
df_spot = df_dz[df_dz["delta_zscore"].abs() >= 1.5]
plt.scatter(df_spot["BP"], df_spot["delta_zscore"], color="red", s=30, zorder=5, label="ΔZ ≥ ±1.5")

# long-term stress(oragne band)
for cid in df_clust["cluster_all_1.0"].dropna().unique():
    if cid > 0:
        seg = df_clust[df_clust["cluster_all_1.0"] == cid]
        plt.axvspan(seg["BP"].max(), seg["BP"].min(), color="orange", alpha=0.3)

# Summarizing plot style
plt.axhline(0, linestyle=":", color="black", linewidth=0.7)
plt.axhline(1.5, linestyle="--", color="red", linewidth=0.6)
plt.axhline(-1.5, linestyle="--", color="red", linewidth=0.6)
plt.xlabel("BP")
plt.ylabel("Z-score")
plt.gca().invert_xaxis()
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(frameon=False)
plt.tight_layout()

# Saving
os.makedirs("output/figures", exist_ok=True)
plt.savefig(FIG_PATH_PNG, dpi=300)
plt.savefig(FIG_PATH_EPS, format="eps", dpi=300)
plt.show()

#6.4 Linkage and Causality Analysis Between the korean peninsula and southe Japan

from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib
matplotlib.use('TkAgg')    
np.random.seed(42)
from pathlib import Path
import os
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import grangercausalitytests

# Creating output package
project_dir = Path().resolve()
fig_dir     = project_dir/"output"/"figures"
tab_dir     = project_dir/"output"/"tables"
fig_dir.mkdir(parents=True, exist_ok=True)
tab_dir.mkdir(parents=True, exist_ok=True)

# Loading data
korea_df = pd.read_csv(project_dir/"data"/"processed"/"spd_korea_z_m3.csv") \
             [['CRA_bin','SPD_Z']].rename(columns={'SPD_Z':'SPD_Z_korea'})
japan_df = pd.read_csv(project_dir/"data"/"processed"/"spd_japan_z_m3.csv") \
             [['CRA_bin','SPD_Z']].rename(columns={'SPD_Z':'SPD_Z_japan'})
filtered = pd.merge(
    korea_df.query("2100 <= CRA_bin <= 2700"),
    japan_df.query("2100 <= CRA_bin <= 2700"),
    on='CRA_bin'
).sort_values('CRA_bin').reset_index(drop=True)

# calcuating lagged spearman 
results = []
for lag in range(-10,11):
    if lag<0:
        x = filtered.SPD_Z_korea.iloc[:lag]
        y = filtered.SPD_Z_japan.iloc[-lag:]
    elif lag>0:
        x = filtered.SPD_Z_korea.iloc[lag:]
        y = filtered.SPD_Z_japan.iloc[:-lag]
    else:
        x = filtered.SPD_Z_korea
        y = filtered.SPD_Z_japan
    r, p = spearmanr(x, y)
    results.append({'Lag_Years': lag*20, 'SpearmanR': r, 'p_value': p})

lagged_df = pd.DataFrame(results)
lagged_df.to_csv(tab_dir/"lagcorrelation_results_z_filtered.csv", index=False)

# Setting plot style
sns.set_style('white')
plt.rcParams.update({
    'axes.edgecolor':'black','xtick.color':'black','ytick.color':'black',
    'font.family':'serif','axes.grid':False
})

# Plots
plt.figure(figsize=(8,5))
plt.plot(lagged_df.Lag_Years, lagged_df.SpearmanR,
         marker='o', linestyle='-')
plt.axhline(0, linestyle='--', linewidth=0.7)
plt.xlabel('Lag (Years)')
plt.ylabel('Spearman Correlation (SPD_Z)')
plt.tight_layout()

# Saving plots
plt.savefig(fig_dir/"fig_s11_lagged_spearman_z_filtered.png", dpi=300)
plt.savefig(fig_dir/"fig_s11_lagged_spearman_z_filtered.eps", format='eps')
plt.show()  


import os
from pathlib import Path
import pandas as pd
import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import grangercausalitytests


import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests

# Creating output package
np.random.seed(42)

project_dir = Path().resolve()
data_dir    = project_dir / "data" / "processed"
out_tab     = project_dir / "output" / "tables"
out_fig     = project_dir / "output" / "figures"

out_tab.mkdir(parents=True, exist_ok=True)
out_fig.mkdir(parents=True, exist_ok=True)

# Loading data
k = (
    pd.read_csv(data_dir / "spd_korea_z_m3.csv")
      [['CRA_bin','SPD_Z']]
      .rename(columns={'SPD_Z':'SPD_Z_korea'})
)
j = (
    pd.read_csv(data_dir / "spd_japan_z_m3.csv")
      [['CRA_bin','SPD_Z']]
      .rename(columns={'SPD_Z':'SPD_Z_japan'})
)

df = (
    pd.merge(k, j, on="CRA_bin")
      .query("2100 <= CRA_bin <= 2700")
      .sort_values("CRA_bin")
      .reset_index(drop=True)
)

# Granger casuality analysis
maxlag = 9
results = []

for direction, (src, tgt) in {
    'Korea_to_Japan': ('SPD_Z_korea', 'SPD_Z_japan'),
    'Japan_to_Korea': ('SPD_Z_japan', 'SPD_Z_korea'),
}.items():
    gc = grangercausalitytests(df[[src, tgt]], maxlag=maxlag, verbose=False)
    for lag in range(1, maxlag+1):
        p_val = gc[lag][0]['ssr_ftest'][1]
        results.append({
            'Lag':       lag,
            'Direction': direction,
            'p_value':   p_val
        })

gr_df = pd.DataFrame(results)
gr_df.to_csv(out_tab / "granger_results_z_bidirectional.csv", index=False)

# Displyaing dataframe
print("\nBidirectional Granger causality results (BP 2700–2100, SPD_Z):\n")
print(gr_df.to_string(index=False))

# Visualizing
plt.figure(figsize=(8, 5))
colors = {'Korea_to_Japan':'black','Japan_to_Korea':'gray'}

for direction, color in colors.items():
    sub = gr_df[gr_df.Direction == direction]
    plt.plot(
        sub.Lag * 20,
        sub.p_value,
        marker='o',
        linestyle='-',
        color=color,
        label=direction.replace('_', ' → ')
    )

# dashed line at p = 0.05 (significance threshold)
plt.axhline(0.05, linestyle='--', linewidth=0.7, color='black')

plt.xlabel("Lag (Years)")
plt.ylabel("p-value")
plt.legend(frameon=False)
plt.tight_layout()

# Saving plots
plt.savefig(out_fig / "fig_s12_granger_bidirectional_spd_z.png", dpi=300)
plt.savefig(out_fig / "fig_s12_granger_bidirectional_spd_z.eps", format='eps')
plt.show()
