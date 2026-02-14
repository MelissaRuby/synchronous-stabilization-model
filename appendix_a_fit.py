import numpy as np
from scipy.optimize import curve_fit

a_bao = 0.01 # Late low
b_bao = 0.9 # Scale
a_de = 0.01
b_de = 0.9
a_t2 = 0.01
b_t2 = 0.9

t_bao_hist = np.array([1963.0, 2005.0, 2010.0, 2015.0, 2020.0, 2023.0, 2023.0, 2023.0, 2023.0])
obs_bao_hist = np.array([0.905, 0.05, 0.03, 0.02, 0.015, 0.0029, 0.0251, 0.0114, 0.0089])
obs_bao_hist = obs_bao_hist / np.max(obs_bao_hist)
sig_bao_hist = np.array([0.01, 0.6, 0.5, 0.4, 0.3, 0.1, 0.1, 0.1, 0.1]) * 3

t_de_hist = np.array([2000.0, 2000.0, 2010.0, 2010.0, 2018.0, 2018.0, 2023.0, 2023.0])
obs_de_hist = np.array([0.5, 0.5, 0.4, 0.4, 0.03, 0.32, 0.1, 0.3])
obs_de_hist = obs_de_hist / np.max(obs_de_hist)
sig_de_hist = np.array([0.6, 0.6, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1]) * 3

t_t2_hist = np.array([2000.0, 2005.0, 2010.0, 2015.0, 2018.0, 2020.0, 2022.0, 2023.0, 2024.0, 2024.5, 2025.0])
obs_t2_hist_raw = np.array([0.000001, 0.00001, 0.001, 0.05, 0.1, 0.2, 0.4, 0.15, 0.25, 0.50, 1.6])
obs_t2_hist = 1 / (1 + obs_t2_hist_raw / 1.6)
sig_t2_hist = np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]) * 3

t_all_hist = np.concatenate([t_bao_hist, t_de_hist, t_t2_hist])
obs_all_hist = np.concatenate([obs_bao_hist, obs_de_hist, obs_t2_hist])
sig_all_hist = np.concatenate([sig_bao_hist, sig_de_hist, sig_t2_hist])

len_bao = len(t_bao_hist)
len_de = len(t_de_hist)
len_t2 = len(t_t2_hist)

def model_func(t_all, t0, lam, psi0):
    psi = psi0 * np.exp(-lam * (t_all - t0))
    model_bao = a_bao + b_bao * psi[:len_bao]
    model_de = a_de + b_de * psi[len_bao:len_bao + len_de]
    model_t2 = a_t2 + b_t2 * psi[len_bao + len_de:]
    return np.concatenate([model_bao, model_de, model_t2])

# Updated bounds to allow slow-decay solution (λ >= 0.001)
bounds = ([1900, 0.001, 0.5], [2030, 100, 1.5])

# For slow/unanchored: seed near 1975
# p0 = [1975, 0.015, 0.916]

# For sharp/near-anchored: seed near 1963 with large λ
p0 = [1963, 31.46, 0.905]

popt, pcov = curve_fit(model_func, t_all_hist, obs_all_hist, p0=p0, sigma=sig_all_hist, bounds=bounds)

print('Best fit params:', popt)
chi2 = np.sum(((obs_all_hist - model_func(t_all_hist, *popt)) / sig_all_hist)**2)
dof = len(obs_all_hist) - 3
reduced_chi2 = chi2 / dof
print('Model chi2:', chi2)
print('Reduced chi2:', reduced_chi2)

mean_bao = np.mean(obs_bao_hist)
chi2_bao_null = np.sum(((obs_bao_hist - mean_bao) / sig_bao_hist)**2)
mean_de = np.mean(obs_de_hist)
chi2_de_null = np.sum(((obs_de_hist - mean_de) / sig_de_hist)**2)
mean_t2 = np.mean(obs_t2_hist)
chi2_t2_null = np.sum(((obs_t2_hist - mean_t2) / sig_t2_hist)**2)
null_chi2_hist = chi2_bao_null + chi2_de_null + chi2_t2_null
print('Null chi2:', null_chi2_hist)
print('Delta chi2:', null_chi2_hist - chi2)
