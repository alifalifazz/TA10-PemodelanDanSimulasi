# app.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# =========================
# 0) Konfigurasi Streamlit
# =========================
st.set_page_config(page_title="TA-10 RK4 SIR COVID-19 Indonesia", layout="centered")
st.title("TA-10 — Simulasi COVID-19 Indonesia (SIR + RK4 From Scratch)")

# =====================================
# 1) Load Data Kaggle (Confirmed Indo)
# =====================================
@st.cache_data(show_spinner=False)
def load_confirmed_indonesia(csv_path: str):
    df = pd.read_csv(csv_path)

    # Deteksi kolom negara (beda versi dataset)
    if "Country/Region" in df.columns:
        country_col = "Country/Region"
    elif "Country_Region" in df.columns:
        country_col = "Country_Region"
    else:
        raise ValueError("Kolom negara tidak ditemukan. Pastikan dataset time-series JHU (confirmed).")

    # Kolom metadata yang umum
    meta_cols = set([
        "Province/State", "Province_State",
        "Lat", "Long", "Latitute", "Longitude",
        country_col
    ])

    date_cols = [c for c in df.columns if c not in meta_cols]
    if len(date_cols) < 10:
        raise ValueError("Kolom tanggal terlalu sedikit. Pastikan file adalah time-series confirmed (global).")

    indo = df[df[country_col] == "Indonesia"]
    if indo.empty:
        raise ValueError("Indonesia tidak ditemukan pada dataset. Cek penulisan negara di kolom country.")

    series = indo[date_cols].sum(axis=0)
    I_data = series.to_numpy(dtype=float)          # confirmed kumulatif
    t_data = np.arange(len(I_data), dtype=float)   # hari ke-0..N-1

    return t_data, I_data, date_cols

# =====================================
# 2) Model SIR (ODE)
# =====================================
def sir_model(t, y, beta, gamma):
    S, I, R = y
    N = S + I + R
    if N <= 0:
        return np.array([0.0, 0.0, 0.0], dtype=float)

    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return np.array([dS, dI, dR], dtype=float)

# =====================================
# 3) RK4 Solver (FROM SCRATCH)
# =====================================
def rk4_solver(f, t0, y0, t_end, h, params):
    t = float(t0)
    y = np.array(y0, dtype=float)

    T = [t]
    Y = [y.copy()]

    # Loop diskrit
    while t < t_end - 1e-12:
        k1 = h * f(t, y, *params)
        k2 = h * f(t + h/2, y + k1/2, *params)
        k3 = h * f(t + h/2, y + k2/2, *params)
        k4 = h * f(t + h,   y + k3,   *params)

        y = y + (k1 + 2*k2 + 2*k3 + k4) / 6.0

        # jaga agar tidak negatif (kadang muncul karena error numerik)
        y = np.maximum(y, 0.0)

        t = t + h
        T.append(t)
        Y.append(y.copy())

    return np.array(T), np.array(Y)

# =====================================
# 4) RMSE
# =====================================
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# =====================================
# 5) UI Input
# =====================================
st.subheader("Input Simulasi")

default_file = "dataset/time_series_covid_19_confirmed.csv"
csv_path = st.text_input("Nama file CSV (confirmed time-series)", value=default_file)

colA, colB = st.columns(2)
with colA:
    beta = st.slider("β (beta) — laju penularan", min_value=0.01, max_value=2.50, value=0.60, step=0.01)
with colB:
    gamma = st.slider("γ (gamma) — laju pemulihan", min_value=0.001, max_value=1.00, value=0.15, step=0.001)

colC, colD = st.columns(2)
with colC:
    h = st.selectbox("Step size RK4 (h)", options=[1.0, 0.5, 0.25, 0.1], index=0)
with colD:
    max_days = st.number_input("Durasi simulasi (hari)", min_value=30, max_value=2000, value=400, step=10)

normalize = st.checkbox("Normalisasi data (disarankan)", value=True)
show_scale_real = st.checkbox("Tampilkan juga grafik skala asli (kasus)", value=True)

run_btn = st.button("Run Simulation")

# =====================================
# 6) Run + Output
# =====================================
if run_btn:
    if not os.path.exists(csv_path):
        st.error(f"File '{csv_path}' tidak ditemukan. Letakkan file CSV di folder yang sama dengan app.py atau isi path lengkap.")
        st.stop()

    try:
        t_data, I_data, date_cols = load_confirmed_indonesia(csv_path)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Batasi durasi sesuai input user
    max_days = int(min(max_days, len(I_data) - 1))
    t_data = t_data[:max_days + 1]
    I_data = I_data[:max_days + 1]

    # Normalisasi jika dipilih
    I_max = I_data.max() if I_data.max() > 0 else 1.0
    if normalize:
        I_target = I_data / I_max
    else:
        I_target = I_data.copy()

    # Kondisi awal: samakan dengan data hari pertama (sesuai modul)
    I0 = float(I_target[0])
    R0 = 0.0
    S0 = max(1.0 - I0, 0.0) if normalize else max((I_target.max() - I0), 0.0)
    y0 = [S0, I0, R0]

    t0 = 0.0
    t_end = float(t_data[-1])

    # Simulasi RK4
    T, Y = rk4_solver(sir_model, t0, y0, t_end, float(h), (float(beta), float(gamma)))
    I_sim = Y[:, 1]

    # Agar bisa dibandingkan dengan data harian, kita sampling hasil RK4 ke integer day
    # Kalau h=1, otomatis pas. Kalau h<1, ambil indeks terdekat untuk tiap hari.
    day_points = np.arange(0, int(t_end) + 1, 1)
    I_sim_day = np.interp(day_points, T, I_sim)
    I_target_day = I_target[:len(day_points)]

    # Hitung RMSE
    e = rmse(I_target_day, I_sim_day)

    st.subheader("Hasil")
    st.write(f"**Parameter:** β = `{beta:.4f}`, γ = `{gamma:.4f}`, h = `{h}`")
    st.write(f"**RMSE:** `{e:.6f}`")

    # Interpretasi sederhana
    recovery_days = (1.0 / gamma) if gamma > 0 else np.inf
    st.info(
        f"Interpretasi singkat:\n"
        f"- β lebih besar → penularan lebih cepat\n"
        f"- γ lebih besar → pemulihan lebih cepat\n"
        f"- Perkiraan rata-rata waktu pemulihan ≈ 1/γ = {recovery_days:.2f} hari"
    )

    # Plot overlay normalized (atau sesuai I_target)
    fig1 = plt.figure()
    plt.scatter(day_points, I_target_day, s=12, label="Data Asli (Indonesia)")
    plt.plot(day_points, I_sim_day, linewidth=2, label="Simulasi RK4 (SIR)")
    plt.xlabel("Hari (t)")
    plt.ylabel("Nilai" + (" (normalized)" if normalize else ""))
    plt.title("Overlay: Data vs Simulasi RK4 — Indonesia")
    plt.legend()
    st.pyplot(fig1)

    # Plot skala asli (kasus) jika user minta dan normalize aktif
    if show_scale_real and normalize:
        I_sim_real = I_sim_day * I_max
        fig2 = plt.figure()
        plt.scatter(day_points, I_data[:len(day_points)], s=12, label="Data Asli (Confirmed)")
        plt.plot(day_points, I_sim_real, linewidth=2, label="Simulasi RK4 (Scaled)")
        plt.xlabel("Hari (t)")
        plt.ylabel("Confirmed (kasus)")
        plt.title("Overlay Skala Asli: Data vs Simulasi RK4 — Indonesia")
        plt.legend()
        st.pyplot(fig2)
