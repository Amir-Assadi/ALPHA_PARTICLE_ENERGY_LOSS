import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.interpolate import UnivariateSpline

# Set font for plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# Enforce monotonic decrease in E(x) before fitting
ENFORCE_MONOTONIC = True

# Smoothing factor for the spline (higher = smoother, rougher Bragg curve)
# Tune this if needed; s=None uses automatic GCV smoothing
SPLINE_SMOOTHING = 0.01

# Dense grid for derivative/plot
N_GRID = 4000


def load_peaks_by_thickness(path):
	data = np.loadtxt(path, comments='#')
	if data.ndim == 1:
		data = data.reshape(1, -1)
	if data.shape[1] < 7:
		raise ValueError('peaks_by_thickness.txt must include energy columns (avg_energy_MeV and uncertainties).')
	thickness_um = data[:, 0]
	energy_mev = data[:, 6]
	energy_total_unc = data[:, 10] if data.shape[1] > 10 else None
	return thickness_um, energy_mev, energy_total_unc


def fit_spline(x, y, yerr=None):
	"""
	Fit smoothing spline using energy uncertainties as weights.
	Returns (spline_func, derivative_func).
	"""
	# Use inverse variance weighting from uncertainties
	# Set w=None to ignore uncertainties and see raw smoothing effect
	w = None
	
	# UnivariateSpline with weights and automatic smoothing
	spline = UnivariateSpline(x, y, w=w, s=SPLINE_SMOOTHING, k=3, ext='extrapolate')
	
	# Derivative is built-in
	dspline = spline.derivative()
	
	return spline, dspline


def isotonic_regression_decreasing(y, w=None):
	# Pool-adjacent-violators algorithm for non-increasing sequence.
	n = len(y)
	if n == 0:
		return y
	if w is None:
		w = np.ones(n, dtype=float)
	values = y.astype(float).copy()
	weights = w.astype(float).copy()
	idx_start = np.arange(n)
	idx_end = np.arange(n)
	i = 0
	while i < n - 1:
		if values[i] >= values[i + 1]:
			i += 1
			continue
		# merge blocks i and i+1
		new_w = weights[i] + weights[i + 1]
		new_v = (values[i] * weights[i] + values[i + 1] * weights[i + 1]) / new_w
		values[i] = new_v
		weights[i] = new_w
		idx_end[i] = idx_end[i + 1]
		# remove block i+1 by shifting left
		values = np.delete(values, i + 1)
		weights = np.delete(weights, i + 1)
		idx_start = np.delete(idx_start, i + 1)
		idx_end = np.delete(idx_end, i + 1)
		n -= 1
		if i > 0:
			i -= 1
	# expand to full length
	out = np.empty(len(y), dtype=float)
	for v, s, e in zip(values, idx_start, idx_end):
		out[s:e + 1] = v
	return out


def main():
	base_dir = os.path.dirname(os.path.abspath(__file__))
	peaks_path = os.path.join(base_dir, 'peaks_by_thickness.txt')
	if not os.path.isfile(peaks_path):
		print(f'File not found: {peaks_path}')
		return

	thickness_um, energy_mev, energy_total_unc = load_peaks_by_thickness(peaks_path)

	# Sort by thickness (x = distance, y = energy)
	order = np.argsort(thickness_um)
	x = thickness_um[order]  # Thickness in μm (x-axis)
	y = energy_mev[order]    # Energy in MeV (y-axis)
	yerr = energy_total_unc[order] if energy_total_unc is not None else None

	# Enforce monotonic decrease in energy to reduce multi-peak artifacts
	if ENFORCE_MONOTONIC:
		w = None if yerr is None else 1.0 / np.maximum(yerr, 1e-12)**2
		y = isotonic_regression_decreasing(y, w=w)

	# Evaluate on a dense grid
	x_fit = np.linspace(np.min(x), np.max(x), N_GRID)

	# Fit smoothing spline with energy uncertainties as weights
	fit_func, dfit_func = fit_spline(x, y, yerr=yerr)
	y_fit = fit_func(x_fit)
	dE_dx_fit = dfit_func(x_fit)
	
	# Fit upper and lower bounds for uncertainty bands
	if yerr is not None:
		y_upper = y + yerr
		y_lower = y - yerr
		fit_upper, dfit_upper = fit_spline(x, y_upper, yerr=yerr)
		fit_lower, dfit_lower = fit_spline(x, y_lower, yerr=yerr)
		y_upper_fit = fit_upper(x_fit)
		y_lower_fit = fit_lower(x_fit)
		dE_dx_upper = dfit_upper(x_fit)
		dE_dx_lower = dfit_lower(x_fit)
	else:
		y_upper_fit = None
		y_lower_fit = None
		dE_dx_upper = None
		dE_dx_lower = None

	# Plot E(x) with spline fit and uncertainty band
	plt.figure(figsize=(8, 5))
	if yerr is not None:
		plt.errorbar(x, y, yerr=yerr, fmt='o', label='Data (E vs x)', capsize=5, alpha=0.7)
	else:
		plt.plot(x, y, 'o', label='Data (E vs x)')
	plt.plot(x_fit, y_fit, 'r-', linewidth=2, label='Spline fit')
	if y_upper_fit is not None:
		plt.fill_between(x_fit, y_lower_fit, y_upper_fit, alpha=0.3, color='red', label='Uncertainty band')
	plt.xlabel('Thickness x (μm)')
	plt.ylabel('Energy E (MeV)')
	plt.title('E(x) fitted with spline (with uncertainty band)')
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.show()

	# Plot Bragg curve: stopping power vs thickness
	plt.figure(figsize=(8, 5))
	# Stopping power = -dE/dx
	stopping_power = -dE_dx_fit
	plt.plot(x_fit, stopping_power, 'r-', linewidth=2, label='-dE/dx (Bragg curve)')
	if dE_dx_upper is not None:
		plt.fill_between(x_fit, -dE_dx_upper, -dE_dx_lower, alpha=0.3, color='red', label='Uncertainty band')
	plt.xlabel('Thickness x (μm)')
	plt.ylabel('Stopping power -dE/dx (MeV/μm)')
	plt.title('Bragg curve from spline fit (with uncertainty band)')
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.show()

	# Write out results
	out_path = os.path.join(base_dir, 'bragg_curve.txt')
	header = 'x_um E_MeV dEdx_MeV_per_um'
	np.savetxt(out_path, np.column_stack([x_fit, y_fit, -dE_dx_fit]), header=header)
	print(f'Wrote Bragg curve (-dE/dx vs x) to: {out_path}')


if __name__ == '__main__':
	main()
