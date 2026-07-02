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
SPLINE_SMOOTHING = 0.012

# Dense grid for derivative/plot
N_GRID = 4000

# Color map for sources
SOURCE_COLORS = {
	'pu': 'blue',
	'am': 'green',
	'cm': 'red',
	'th': 'purple',
}


def load_peaks_by_thickness(path):
	"""
	Load peaks by thickness file with source column.
	Returns dict: {source: (thickness_um, energy_mev, energy_total_unc)}
	"""
	# Use genfromtxt to handle mixed types (string source column)
	data = np.genfromtxt(path, comments='#', dtype=None, encoding=None)
	
	if data.ndim == 0:
		data = data.reshape(1)
	
	# Column indices:
	# 2: thickness_um, 3: source, 9: avg_energy_MeV, 13: energy_total_unc
	sources_data = {}
	for row in data:
		source = row[3]
		thickness_um = float(row[2])
		energy_mev = float(row[9])
		energy_total_unc = float(row[13]) if len(row) > 13 else None
		
		if source not in sources_data:
			sources_data[source] = {'thickness': [], 'energy': [], 'error': []}
		
		sources_data[source]['thickness'].append(thickness_um)
		sources_data[source]['energy'].append(energy_mev)
		sources_data[source]['error'].append(energy_total_unc)
	
	# Convert lists to arrays
	for source in sources_data:
		sources_data[source]['thickness'] = np.array(sources_data[source]['thickness'])
		sources_data[source]['energy'] = np.array(sources_data[source]['energy'])
		sources_data[source]['error'] = np.array(sources_data[source]['error'])
	
	return sources_data


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


def process_station(base_dir, station_tag):
	"""Process one station: load data, fit splines, plot, and save results."""
	peaks_path = os.path.join(base_dir, f'peaks_by_thickness_{station_tag}.txt')
	
	if not os.path.isfile(peaks_path):
		print(f'File not found: {peaks_path}. Skipping {station_tag.upper()}.')
		return
	
	print(f'\nProcessing {station_tag.upper()}...')
	sources_data = load_peaks_by_thickness(peaks_path)
	
	if not sources_data:
		print(f'No data found in {peaks_path}')
		return
	
	# Store fit results for each source
	fit_results = {}
	
	for source, data in sources_data.items():
		x = data['thickness']
		y = data['energy']
		yerr = data['error']
		
		# Sort by thickness
		order = np.argsort(x)
		x = x[order]
		y = y[order]
		yerr = yerr[order]
		
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
		
		fit_results[source] = {
			'x': x,
			'y': y,
			'yerr': yerr,
			'x_fit': x_fit,
			'y_fit': y_fit,
			'dE_dx_fit': dE_dx_fit,
			'y_upper_fit': y_upper_fit,
			'y_lower_fit': y_lower_fit,
			'dE_dx_upper': dE_dx_upper,
			'dE_dx_lower': dE_dx_lower,
		}
	
	# Plot E(x) for each source separately
	for source, res in fit_results.items():
		color = SOURCE_COLORS.get(source, 'black')
		plt.figure(figsize=(8, 5))
		if res['yerr'] is not None:
			plt.errorbar(res['x'], res['y'], yerr=res['yerr'], fmt='o', 
			            color=color, label='Data', capsize=5, alpha=0.7)
		else:
			plt.plot(res['x'], res['y'], 'o', color=color, label='Data', alpha=0.7)
		plt.plot(res['x_fit'], res['y_fit'], '-', color=color, linewidth=2, label='Spline fit')
		if res['y_upper_fit'] is not None:
			plt.fill_between(res['x_fit'], res['y_lower_fit'], res['y_upper_fit'], 
			                alpha=0.3, color=color, label='Uncertainty band')
		plt.xlabel('Thickness x (μm)')
		plt.ylabel('Energy E (MeV)')
		plt.title(f'{station_tag.upper()} {source.upper()} - E(x) fitted with spline')
		plt.grid(True, alpha=0.3)
		plt.legend()
		plt.tight_layout()
		plt.show()
	
	# Plot Bragg curve: stopping power vs thickness for each source separately
	for source, res in fit_results.items():
		color = SOURCE_COLORS.get(source, 'black')
		plt.figure(figsize=(8, 5))
		stopping_power = -res['dE_dx_fit']
		plt.plot(res['x_fit'], stopping_power, '-', color=color, linewidth=2, label='Bragg curve')
		if res['dE_dx_upper'] is not None:
			plt.fill_between(res['x_fit'], -res['dE_dx_upper'], -res['dE_dx_lower'], 
			                alpha=0.3, color=color, label='Uncertainty band')
		plt.xlabel('Thickness x (μm)')
		plt.ylabel('Stopping power -dE/dx (MeV/μm)')
		plt.title(f'{station_tag.upper()} {source.upper()} - Bragg curve from spline fit')
		plt.grid(True, alpha=0.3)
		plt.legend()
		plt.tight_layout()
		plt.show()
	
	# Write out results for each source
	for source, res in fit_results.items():
		out_path = os.path.join(base_dir, f'bragg_curve_{station_tag}_{source}.txt')
		header = 'x_um E_MeV dEdx_MeV_per_um'
		np.savetxt(out_path, np.column_stack([res['x_fit'], res['y_fit'], -res['dE_dx_fit']]), 
		          header=header, fmt='%.6f')
		print(f'Wrote {source.upper()} Bragg curve to: {out_path}')


def main():
	base_dir = os.path.dirname(os.path.abspath(__file__))
	
	# Process station 3 (ar3)
	process_station(base_dir, 'ar3')
	
	# Process station 4 (ar4) - will skip gracefully if not available
	process_station(base_dir, 'ar4')


if __name__ == '__main__':
	main()
