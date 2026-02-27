import numpy as np
from scipy.optimize import curve_fit

# Shared data definitions (used by all appendices)

a_bao = 0.01 # Late low
b_bao = 0.9 # Scale
a_de = 0.01
b_de = 0.9
a_t1 = 0.01
b_t1 = 0.9

t_bao_hist = np.array([1963.0, 2005.0, 2010.0, 2015.0, 2020.0, 2023.0, 2023.0, 2023.0, 2023.0])
obs_bao_hist = np.array([0.905, 0.05, 0.03, 0.02, 0.015, 0.0029, 0.0251, 0.0114, 0.0089])
obs_bao_hist = obs_bao_hist / np.max(obs_bao_hist)
sig_bao_hist = np.array([0.01, 0.6, 0.5, 0.4, 0.3, 0.1, 0.1, 0.1, 0.1]) * 3

t_de_hist = np.array([2000.0, 2000.0, 2010.0, 2010.0, 2018.0, 2018.0, 2023.0, 2023.0])
obs_de_hist = np.array([0.5, 0.5, 0.4, 0.4, 0.03, 0.32, 0.1, 0.3])
obs_de_hist = obs_de_hist / np.max(obs_de_hist)
sig_de_hist = np.array([0.6, 0.6, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1]) * 3

t_t1_hist = np.array([2000.0, 2005.0, 2010.0, 2015.0, 2018.0, 2020.0, 2022.0, 2023.0, 2024.0, 2024.5, 2025.0])
obs_t1_hist_raw = np.array([0.000001, 0.00001, 0.001, 0.05, 0.1, 0.2, 0.4, 0.15, 0.25, 0.50, 1.6])
obs_t1_hist = 1 / (1 + obs_t1_hist_raw / 1.6)
sig_t1_hist = np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]) * 3

t_all_hist = np.concatenate([t_bao_hist, t_de_hist, t_t1_hist])
obs_all_hist = np.concatenate([obs_bao_hist, obs_de_hist, obs_t1_hist])
sig_all_hist = np.concatenate([sig_bao_hist, sig_de_hist, sig_t1_hist])

len_bao = len(t_bao_hist)
len_de = len(t_de_hist)
len_t1 = len(t_t1_hist)

def model_func(t_all, t0, lam, psi0):
    psi = psi0 * np.exp(-lam * (t_all - t0))
    model_bao = a_bao + b_bao * psi[:len_bao]
    model_de = a_de + b_de * psi[len_bao:len_bao + len_de]
    model_t1 = a_t1 + b_t1 * psi[len_bao + len_de:]
    return np.concatenate([model_bao, model_de, model_t1])

# Common bounds
bounds = ([1900, 0.001, 0.5], [2030, 100, 1.5])
