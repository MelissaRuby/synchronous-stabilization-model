import numpy as np
from scipy.optimize import curve_fit
from model_common import * # imports all data and model_func

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
mean_t1 = np.mean(obs_t1_hist)
chi2_t1_null = np.sum(((obs_t1_hist - mean_t1) / sig_t1_hist)**2)
null_chi2_hist = chi2_bao_null + chi2_de_null + chi2_t1_null
print('Null chi2:', null_chi2_hist)
print('Delta chi2:', null_chi2_hist - chi2)
