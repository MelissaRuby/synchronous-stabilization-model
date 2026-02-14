import numpy as np
from scipy.optimize import curve_fit

# IMPORTANT: This script depends on variables from appendix_a_fit.py
# You must run appendix_a_fit.py first (or define the required variables):
# - model_func
# - t_all_hist
# - sig_all_hist
# - bounds
# - p0
# - len_bao, len_de, len_t2
# - obs_bao_hist, obs_de_hist, obs_t2_hist, sig_bao_hist, sig_de_hist, sig_t2_hist

mean_bao = np.mean(obs_bao_hist)
mean_de = np.mean(obs_de_hist)
mean_t2 = np.mean(obs_t2_hist)

def compute_chi2_null(obs_all):
    obs_bao = obs_all[:len_bao]
    obs_de = obs_all[len_bao:len_bao + len_de]
    obs_t2 = obs_all[len_bao + len_de:]
    chi2_bao = np.sum(((obs_bao - mean_bao) / sig_bao_hist)**2)
    chi2_de = np.sum(((obs_de - mean_de) / sig_de_hist)**2)
    chi2_t2 = np.sum(((obs_t2 - mean_t2) / sig_t2_hist)**2)
    return chi2_bao + chi2_de + chi2_t2

num_trials = 1000
deltas = []

for i in range(num_trials):
    fake_bao = mean_bao + np.random.randn(len_bao) * sig_bao_hist
    fake_de = mean_de + np.random.randn(len_de) * sig_de_hist
    fake_t2 = mean_t2 + np.random.randn(len_t2) * sig_t2_hist
    fake_all = np.concatenate([fake_bao, fake_de, fake_t2])

    chi2_null = compute_chi2_null(fake_all)    
    
    try:    
        popt, pcov = curve_fit(model_func, t_all_hist, fake_all, p0=p0, sigma=sig_all_hist, bounds=bounds, maxfev=10000)    
        model_pred = model_func(t_all_hist, *popt)    
        chi2_model = np.sum(((fake_all - model_pred) / sig_all_hist)**2)    
        delta = chi2_null - chi2_model    
        deltas.append(delta)    
    except:    
        pass

observed_delta = 802 # or 827 for unanchored
frac = np.sum(np.array(deltas) >= observed_delta) / len(deltas) if deltas else 0
print(f'Fraction of trials with Delta chi2 >= {observed_delta}: {frac}')
