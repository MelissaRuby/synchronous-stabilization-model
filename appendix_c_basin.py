import numpy as np
from scipy.optimize import curve_fit
from model_common import * # imports all data and model_func

# Seed sweep (100 random + 10 targeted)
num_random_seeds = 100
num_targeted_seeds = 10
random_seeds = np.random.uniform([1900, 0.001, 0.5], [2030, 100, 1.5], size=(num_random_seeds, 3))
targeted_seeds = np.array([[1963, 31, 0.905]] * num_targeted_seeds) + np.random.uniform(-5, 5, size=(num_targeted_seeds, 3)) # Near 1963 sharp

all_seeds = np.vstack((random_seeds, targeted_seeds))

# Run sweeps
converged = []
for seed in all_seeds:
    try:
        popt, pcov = curve_fit(model_func, t_all_hist, obs_all_hist, p0=seed, sigma=sig_all_hist, bounds=bounds, maxfev=10000)
        converged.append(popt)
    except:
        pass

# Cluster converged t0
converged_t0 = [p[0] for p in converged]
# Simple binning or clustering (e.g., sharp < 1970, slow > 1970)
sharp = [p for p in converged if p[0] < 1970]
slow = [p for p in converged if p[0] >= 1970]

print(f"Sharp basin: \~{len(sharp)} seeds, mean t0 \~{np.mean([p[0] for p in sharp]):.1f}")
print(f"Slow basin: \~{len(slow)} seeds, mean t0 \~{np.mean([p[0] for p in slow]):.1f}")
print(f"Non-converged: {len(all_seeds) - len(converged)} seeds")

# Fixed-t0 scan example (optimize lambda, psi0)
t0_range = np.linspace(1968, 1992, 25)
chi2_scan = []
for fixed_t0 in t0_range:
    def fixed_model(t_all, lam, psi0):
        return model_func(t_all, fixed_t0, lam, psi0)
    try:
        popt, pcov = curve_fit(fixed_model, t_all_hist, obs_all_hist, p0=[0.015, 0.917], sigma=sig_all_hist, bounds=([0.001, 0.5], [100, 1.5]))
        chi2 = np.sum(((obs_all_hist - fixed_model(t_all_hist, *popt)) / sig_all_hist)**2)
        chi2_scan.append(chi2)
    except:
        chi2_scan.append(np.nan)

print(f"Fixed-t0 scan min chi2: {min(chi2_scan):.2f} at t0 \~{t0_range[np.argmin(chi2_scan)]:.1f}")




