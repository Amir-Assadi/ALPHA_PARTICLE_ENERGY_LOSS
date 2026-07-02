import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt

try:
	from scipy.optimize import curve_fit
	from scipy.special import erfc
except Exception:
	curve_fit = None
	erfc = None

# Set font for plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# --- User configuration: provide peak channels per file and optional per-file gaps ---
# Example: PEAK_CHANNELS = {'T0one.txt': 1607, 'T1_5two.txt': 802}
PEAK_CHANNELS = {
	'T0one.txt': 1588,
	'T0two.txt': 1598,
	'T0three.txt': 1600,
	'T1_5one.txt': 1519,
	'T1_5two.txt': 1528,
	'T1_5three.txt': 1530,
	'T4_5one.txt': 1397,
	'T4_5two.txt': 1408,
	'T4_5three.txt': 1407,
	'T6one.txt': 1307,
	'T6two.txt': 1319,
	'T6three.txt': 1316,
	'T7_5one.txt': 1225,
	'T7_5two.txt': 1228,
	'T7_5three.txt': 1227,
	'T10_5one.txt': 1082,
	'T10_5two.txt': 1084,
	'T10_5three.txt': 1082,
	'T12one.txt': 986,
	'T12two.txt': 986,
	'T12three.txt': 986,
	'T18one.txt': 786,
	'T18two.txt': 792,
	'T18three.txt': 792,
	'T19_5one.txt': 674,
	'T19_5two.txt': 670,
	'T19_5three.txt': 680,
	'T22_5one.txt': 456,
	'T22_5two.txt': 462,
	'T22_5three.txt': 467,
	'T24one.txt': 284,
	'T24two.txt': 286,
	'T24three.txt': 287,
	'T25_5one.txt': 118,
	'T25_5two.txt': 118,
	'T25_5three.txt': 118,
}

# Optional per-file fit gap (channels from peak to fit edge). If missing, GAP_DISTANCE is used.
PEAK_GAPS = {
    'T0one.txt': 9,
	'T0two.txt': 10,
	'T0three.txt': 10,
	'T1_5one.txt': 11,
	'T1_5two.txt': 11,
	'T1_5three.txt': 11,
	'T4_5one.txt': 22,
	'T4_5two.txt': 22,
	'T4_5three.txt': 20,
	'T6one.txt': 20,
	'T6two.txt': 20,
	'T6three.txt': 20,
	'T7_5one.txt': 20,
	'T7_5two.txt': 20,
	'T7_5three.txt': 20,
	'T10_5one.txt': 24,
	'T10_5two.txt': 24,
	'T10_5three.txt': 24,
	'T12one.txt': 28,
	'T12two.txt': 28,
	'T12three.txt': 28,
	'T18one.txt': 40,
	'T18two.txt': 40,
	'T18three.txt': 40,
	'T19_5one.txt': 50,
	'T19_5two.txt': 50,
	'T19_5three.txt': 50,
	'T22_5one.txt': 71,
	'T22_5two.txt': 71,
	'T22_5three.txt': 71,
	'T24one.txt': 75,
	'T24two.txt': 75,
	'T24three.txt': 75,
	'T25_5one.txt': 52,
	'T25_5two.txt': 52,
	'T25_5three.txt': 52,
}

# default gap distance from peak for fit range edges
GAP_DISTANCE = 8
# base constant for search span calculation: search_span = SEARCH_SPAN_BASE + peak_gap
SEARCH_SPAN_BASE = 12
# reduced-chi2 window: accept fits within center ± window
# e.g., RED_CHI_CENTER=1.0, RED_CHI_WINDOW=0.1 gives [0.9, 1.1]
RED_CHI_CENTER = 1.0
RED_CHI_WINDOW = 0.1


def parse_counts(filepath):
	counts = []
	start = False
	with open(filepath, 'r', errors='ignore') as f:
		for line in f:
			s = line.strip()
			if not start:
				if re.match(r'^0\s+2047\b', s):
					start = True
				continue
			if s.startswith('$'):
				break
			if not s:
				continue
			for tok in s.split():
				try:
					counts.append(int(tok))
				except ValueError:
					pass
	return counts


def model_gaussian(x, A, mu, sigma, c):
	return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c


def model_generalized_normal(x, A, mu, sigma, beta, c):
	z = np.abs((x - mu) / sigma)
	return A * np.exp(- (z ** beta)) + c


def model_crystalball(x, A, mu, sigma, alpha, n, c):
	x = np.array(x, dtype=float)
	t = (x - mu) / sigma
	out = np.empty_like(t)
	abs_alpha = np.abs(alpha)
	with np.errstate(divide='ignore', invalid='ignore'):
		logA_cb = n * (np.log(n) - np.log(abs_alpha + 1e-30)) - 0.5 * (abs_alpha ** 2)
		logA_cb = np.clip(logA_cb, -700, 700)
		A_cb = np.exp(logA_cb)
	B = n / (abs_alpha + 1e-30) - abs_alpha
	mask_gauss = t > -abs_alpha
	out[mask_gauss] = np.exp(-0.5 * t[mask_gauss] ** 2)
	tt = B - t[~mask_gauss]
	tt_safe = np.maximum(tt, 1e-12)
	with np.errstate(over='ignore', invalid='ignore'):
		out[~mask_gauss] = A_cb * (tt_safe ** (-n))
	return A * out + c

def model_gauss_one_sided_exp(x, A, mu, sigma, lam, c):
	x = np.array(x, dtype=float)
	gauss = A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
	exp_arg = -lam * np.maximum(0, x - mu)
	exp_arg = np.clip(exp_arg, -700, 700)
	with np.errstate(over='ignore', invalid='ignore'):
		tail = gauss * np.exp(exp_arg)
	return tail + c


def model_polynomial4(x, A, mu, a2, a3, a4, c):
	"""4th degree polynomial peak model centered at mu"""
	x = np.array(x, dtype=float)
	dx = x - mu
	return A + a2 * (dx**2) + a3 * (dx**3) + a4 * (dx**4) + c


def model_poly_expo(x, A, mu, sigma, lam, offset, c):
	"""Polynomial-exponential hybrid: polynomial left and at peak, exponential tail starting at mu + offset"""
	x = np.array(x, dtype=float)
	out = np.zeros_like(x)
	
	# transition point (offset is a fit parameter)
	x_trans = mu + offset
	
	# polynomial part (inverted parabola centered at mu): x <= x_trans
	mask_poly = x <= x_trans
	dx_poly = x[mask_poly] - mu
	poly_val = A * (1.0 - (dx_poly / sigma) ** 2) + c
	out[mask_poly] = poly_val
	
	# value at transition for continuity
	amplitude_trans = A * (1.0 - (offset / sigma) ** 2)
	
	# exponential part: x > x_trans, decaying from amplitude_trans
	mask_expo = x > x_trans
	exp_arg = -lam * (x[mask_expo] - x_trans)
	exp_arg = np.clip(exp_arg, -700, 700)
	out[mask_expo] = amplitude_trans * np.exp(exp_arg) + c
	
	return out


MODEL_FUNCS = {
	'Gaussian': (model_gaussian, 4),
	'GeneralizedNormal': (model_generalized_normal, 5),
	'CrystalBall': (model_crystalball, 6),
	'GaussOneSidedExp': (model_gauss_one_sided_exp, 5),
	'Polynomial4': (model_polynomial4, 6),
	'PolyExpo': (model_poly_expo, 6),
}


def extract_thickness(filename):
	"""Extract thickness from filename like 'T0one.txt' -> '0 μm' or 'T1_5two.txt' -> '1.5 μm'"""
	match = re.match(r'^T(?P<th>\d+(?:_\d+)?)', filename)
	if match:
		th_raw = match.group('th')
		th = float(th_raw.replace('_', '.'))
		return f"{th:.1f} μm" if th % 1 != 0 else f"{int(th)} μm"
	return filename


def find_all_fits(counts, peak_ch, gap=8, search_span=20, red_chi_center=1.0, red_chi_window=0.1):
	"""Collect all fits within reduced-chi2 window, grouped by model."""
	n_channels = len(counts)
	start_min = max(0, int(peak_ch - search_span))
	start_max = max(0, int(peak_ch - gap - 1))
	end_min = min(n_channels - 1, int(peak_ch + gap + 1))
	end_max = min(n_channels - 1, int(peak_ch + search_span))

	accepted_by_model = {name: [] for name in MODEL_FUNCS.keys()}
	accepted_errs_by_model = {name: [] for name in MODEL_FUNCS.keys()}
	best_fits_info = []
	red_chi_min = red_chi_center - red_chi_window
	red_chi_max = red_chi_center + red_chi_window

	for start in range(start_min, start_max + 1):
		for end in range(end_min, end_max + 1):
			if end - start < 6:
				continue
			x_idx = np.arange(start, end + 1)
			y_vals = np.array(counts[start:end + 1], dtype=float)
			sigma_y = np.sqrt(y_vals)
			sigma_y[sigma_y == 0] = 1.0

			for name, (func, nargs) in MODEL_FUNCS.items():
				try:
					if name == 'Gaussian':
						p0 = [max(y_vals) - min(y_vals), peak_ch, max(1.0, (end - start) / 10.0), min(y_vals)]
						bounds = ([0, peak_ch - 10, 0.1, -np.inf], [np.inf, peak_ch + 10, (end - start), np.inf])
					elif name == 'GeneralizedNormal':
						p0 = [max(y_vals) - min(y_vals), peak_ch, max(1.0, (end - start) / 10.0), 2.0, min(y_vals)]
						bounds = ([0, peak_ch - 10, 0.01, 0.1, -np.inf], [np.inf, peak_ch + 10, (end - start), 10.0, np.inf])

					elif name == 'CrystalBall':
						p0 = [max(y_vals) - min(y_vals), peak_ch, max(0.5, (end - start) / 20.0), 1.5, 3.0, min(y_vals)]
						bounds = ([0, peak_ch - 10, 0.1, 0.1, 1.1, -np.inf], [np.inf, peak_ch + 10, (end - start), 5.0, 200.0, np.inf])
					elif name == 'GaussOneSidedExp':
						p0 = [max(y_vals) - min(y_vals), peak_ch, max(0.5, (end - start) / 20.0), 0.05, min(y_vals)]
						bounds = ([0, peak_ch - 10, 0.1, 1e-4, -np.inf], [np.inf, peak_ch + 10, (end - start), 5.0, np.inf])
					elif name == 'Polynomial4':
						p0 = [max(y_vals), peak_ch, -0.01, 0.0, 0.0, min(y_vals)]
						bounds = ([-np.inf, peak_ch - 10, -np.inf, -np.inf, -np.inf, -np.inf], [np.inf, peak_ch + 10, np.inf, np.inf, np.inf, np.inf])
					elif name == 'PolyExpo':
						p0 = [max(y_vals) - min(y_vals), peak_ch, max(1.0, (end - start) / 10.0), 0.05, 0.0, min(y_vals)]
						bounds = ([0, peak_ch - 10, 0.1, 1e-4, -gap, -np.inf], [np.inf, peak_ch + 10, (end - start), 5.0, gap, np.inf])
					else:
						continue

					popt, pcov = curve_fit(func, x_idx, y_vals, p0=p0, sigma=sigma_y, absolute_sigma=True, bounds=bounds, maxfev=20000)
					model_y = func(x_idx, *popt)
					chi2 = np.sum(((y_vals - model_y) / sigma_y) ** 2)
					dof = max(1, len(x_idx) - len(popt))
					red_chi2 = chi2 / dof

					if red_chi_min <= red_chi2 <= red_chi_max:
						try:
							perr = np.sqrt(np.diag(pcov)) if pcov is not None else None
						except Exception:
							perr = None
						try:
							mu_val = float(popt[1])
						except Exception:
							mu_val = float(peak_ch)
						if perr is not None and len(perr) > 1 and np.isfinite(perr[1]):
							mu_err = float(perr[1])
						else:
							mu_err = 1.0
						accepted_by_model[name].append(mu_val)
						accepted_errs_by_model[name].append(mu_err)
						best_fits_info.append({
							'name': name,
							'x_idx': x_idx,
							'y_vals': y_vals,
							'popt': popt,
							'pcov': pcov,
							'red_chi2': red_chi2,
							'func': func,
						})
				except Exception:
					continue

	# flatten and apply IQR outlier removal
	all_vals = []
	all_errs = []
	for name in accepted_by_model:
		all_vals.extend(accepted_by_model[name])
		all_errs.extend(accepted_errs_by_model[name])

	if len(all_vals) == 0:
		return None, None, None, None

	# IQR outlier removal
	all_vals_arr = np.array(all_vals, dtype=float)
	Q1 = np.percentile(all_vals_arr, 25)
	Q3 = np.percentile(all_vals_arr, 75)
	IQR = Q3 - Q1
	lower_bound = Q1 - 1.5 * IQR
	upper_bound = Q3 + 1.5 * IQR
	mask = (all_vals_arr >= lower_bound) & (all_vals_arr <= upper_bound)

	all_vals_filtered = [v for i, v in enumerate(all_vals) if mask[i]]
	all_errs_filtered = [e for i, e in enumerate(all_errs) if mask[i]]

	if len(all_vals_filtered) == 0:
		return None, None, None, None

	return accepted_by_model, best_fits_info, all_vals_filtered, all_errs_filtered


def main():
	base_dir = os.path.dirname(os.path.abspath(__file__))
	files = os.listdir(base_dir)

	# Try to load calibration parameters
	calib_params = None
	calib_path = os.path.join(base_dir, 'calibration_params.txt')
	if os.path.isfile(calib_path):
		try:
			calib_params = {}
			with open(calib_path, 'r') as f:
				for line in f:
					line = line.strip()
					if not line or line.startswith('#'):
						continue
					parts = line.split('=')
					if len(parts) == 2:
						key = parts[0].strip()
						val = float(parts[1].strip())
						calib_params[key] = val
			print(f'Loaded calibration parameters: a={calib_params.get("a", "N/A")}, c={calib_params.get("c", "N/A")}')
		except Exception as e:
			print(f'Warning: Could not load calibration parameters: {e}')
			calib_params = None
	else:
		print(f'Note: calibration_params.txt not found. Energy calibration will not be applied.')
		print(f'      Run Calibration_Al.py first to generate it.')

	# match filenames like T0one.txt, T1_5two.txt, T12three.txt etc.
	m = re.compile(r'^T(?P<th>\d+(?:_\d+)?)(?P<trial>one|two|three)\.txt$', re.IGNORECASE)

	file_map = []
	for fn in files:
		mm = m.match(fn)
		if mm:
			th_raw = mm.group('th')
			th = float(th_raw.replace('_', '.'))
			trial = mm.group('trial').lower()
			file_map.append((th, trial, fn))

	if not file_map:
		print('No T* trial files found in directory')
		return

	results = {}  # fn -> (thickness, trial, mu, mu_err, model_name)

	for th, trial, fn in sorted(file_map):
		path = os.path.join(base_dir, fn)
		counts = parse_counts(path)
		if not counts:
			print(f'No data in {fn}')
			continue
		# allow user-specified peak channel per file, otherwise use argmax
		peak_ch = PEAK_CHANNELS.get(fn, None)
		if peak_ch is None:
			peak_ch = int(np.argmax(counts))
		else:
			peak_ch = int(peak_ch)

		# plot raw spectrum (spectrum alone) for quick inspection
		channels = np.arange(len(counts))
		thickness_label = extract_thickness(fn)
		plt.figure(figsize=(10,4))
		plt.bar(channels, counts, width=1.0, edgecolor='black', alpha=0.9)
		plt.xlabel('Channel')
		plt.ylabel('Counts')
		plt.title(f"{thickness_label} — Raw spectrum")
		plt.xlim(max(0, peak_ch - 500), min(len(counts) - 1, peak_ch + 500))
		plt.tight_layout()
		plt.show()

		if curve_fit is None:
			print('scipy not available; cannot fit. Using argmax fallback.')
			mu = float(peak_ch)
			mu_sys = 0.0
			mu_stat = 1.0
			model_name = 'argmax'
		else:
			gap = PEAK_GAPS.get(fn, GAP_DISTANCE)
			search_span = SEARCH_SPAN_BASE + gap
			accepted_by_model, best_fits_info, all_vals, all_errs = find_all_fits(counts, peak_ch, gap=gap, search_span=search_span, red_chi_center=RED_CHI_CENTER, red_chi_window=RED_CHI_WINDOW)
			if all_vals is None or len(all_vals) == 0:
				mu = float(peak_ch)
				mu_sys = 0.0
				mu_stat = 1.0
				model_name = 'fallback'
			else:
				mus_arr = np.array(all_vals, dtype=float)
				mus_errs_arr = np.array(all_errs, dtype=float)
				mu = float(np.mean(mus_arr))
				mu_sys = float(np.std(mus_arr, ddof=0))  # systematic: std of distribution
				mu_stat = float(np.mean(mus_errs_arr))   # statistical: avg of fit errors
				mu_err = mu_stat  # for compatibility with existing code
				model_name = 'ensemble'
				
				# stacked histogram per model (filtered data)
				thickness_label = extract_thickness(fn)
				plt.figure(figsize=(8,4))
				model_names = [name for name in MODEL_FUNCS.keys() if len(accepted_by_model[name]) > 0]
				# only use filtered data for histogram
				data_by_model = {}
				for name in model_names:
					all_model_vals = np.array(accepted_by_model[name], dtype=float)
					# filter by the same mask that was applied overall
					all_model_filtered = [v for v in all_model_vals if v in all_vals]
					if len(all_model_filtered) > 0:
						data_by_model[name] = np.array(all_model_filtered)
				
				data = [data_by_model[name] for name in model_names if name in data_by_model]
				color_map = {
					'Gaussian': 'C0',
					'GeneralizedNormal': 'C1',
					'CrystalBall': 'C2',
					'GaussOneSidedExp': 'C3',
					'Polynomial4': 'C4',
					'PolyExpo': 'C5',
				}
				colors = [color_map.get(name, None) for name in model_names if name in data_by_model]
				n_bins = max(6, int(min(50, len(mus_arr)//1)))
				plt.hist(data, bins=n_bins, stacked=True, color=colors, label=model_names, edgecolor='black', alpha=0.8)
				plt.xlabel('Fitted peak (μ)')
				plt.ylabel('Count')
				plt.title(thickness_label + ' — Distribution of accepted peak values (stacked by model)')
				plt.axvline(mu, color='r', linestyle='--', linewidth=2, label=f'mean = {mu:.2f}')
				annot = f'mean = {mu:.2f}\nsys = {np.std(mus_arr, ddof=0):.3f}\nstat = {mu_err:.3f}\nN = {len(mus_arr)}'
				plt.legend(loc='upper left')
				plt.text(0.98, 0.95, annot, transform=plt.gca().transAxes, va='top', ha='right', bbox=dict(facecolor='white', alpha=0.8))
				plt.tight_layout()
				plt.show()
				
				# Plot top 4 best fits in 2x2 subplots
				if len(best_fits_info) > 0:
					thickness_label = extract_thickness(fn)
					n_channels = len(counts)
					VIEW_MARGIN = 20  # margin to extend the view beyond the fit region (in channels)
					
					best_fits_sorted = sorted(best_fits_info, key=lambda x: abs(x['red_chi2'] - 1.0))
					top_4 = best_fits_sorted[:min(4, len(best_fits_sorted))]
					
					fig, axes = plt.subplots(2, 2, figsize=(12, 10))
					axes = axes.flatten()
					
					for idx, fit_info in enumerate(top_4):
						ax = axes[idx]
						x_idx = fit_info['x_idx']
						y_vals = fit_info['y_vals']
						popt = fit_info['popt']
						pcov = fit_info['pcov']
						func = fit_info['func']
						name = fit_info['name']
						red_chi2 = fit_info['red_chi2']
						
						# extract mu and its uncertainty
						try:
							perr = np.sqrt(np.diag(pcov)) if pcov is not None else None
						except Exception:
							perr = None
						try:
							mu_val = float(popt[1])
						except Exception:
							mu_val = np.nan
						if perr is not None and len(perr) > 1 and np.isfinite(perr[1]):
							mu_err_fit = float(perr[1])
						else:
							mu_err_fit = np.nan
						
						# extend the displayed range to show context
						x_min_fit = int(x_idx[0])
						x_max_fit = int(x_idx[-1])
						x_min_view = max(0, x_min_fit - VIEW_MARGIN)
						x_max_view = min(n_channels - 1, x_max_fit + VIEW_MARGIN)
						
						# plot full histogram for extended range
						x_extended = np.arange(x_min_view, x_max_view + 1)
						y_extended = np.array(counts[x_min_view:x_max_view + 1], dtype=float)
						ax.bar(x_extended, y_extended, width=1.0, edgecolor='black', alpha=0.6, label='Data')
						
						# plot fit only over the fitted region
						y_model = func(x_idx, *popt)
						ax.plot(x_idx, y_model, 'r-', linewidth=2, label='Fit')
						
						ax.set_xlabel('Channel')
						ax.set_ylabel('Counts')
						if np.isfinite(mu_val) and np.isfinite(mu_err_fit):
							ax.set_title(f'{name}\n$\mu$ = {mu_val:.2f} ± {mu_err_fit:.2f}\n$\chi^2_{{red}}$ = {red_chi2:.3f}')
						else:
							ax.set_title(f'{name}\n$\chi^2_{{red}}$ = {red_chi2:.3f}')
						ax.legend()
					
					# hide unused subplots
					for idx in range(len(top_4), 4):
						axes[idx].axis('off')
					
					fig.suptitle(thickness_label + ' — Top 4 Best Fits', fontsize=14, fontweight='bold')
					plt.tight_layout()
					plt.show()

		results[fn] = (th, trial, mu, mu_sys, mu_stat, model_name)

	# write per-file (no averaging) peaks file
	out_all_path = os.path.join(base_dir, 'peaks_all_trials.txt')
	with open(out_all_path, 'w') as outf:
		if calib_params:
			outf.write('# filename thickness_um trial peak_channel sys_uncertainty stat_uncertainty total_uncertainty model energy_MeV energy_sys_unc energy_stat_unc energy_calib_unc energy_total_unc\n')
		else:
			outf.write('# filename thickness_um trial peak_channel sys_uncertainty stat_uncertainty total_uncertainty model\n')
		for fn in sorted(results.keys()):
			th, trial, mu, mu_sys, mu_stat, model_name = results[fn]
			mu_total = np.sqrt(mu_sys**2 + mu_stat**2)
			line = f"{fn} {th:.3f} {trial} {mu:.3f} {mu_sys:.3f} {mu_stat:.3f} {mu_total:.3f} {model_name}"
			
			# Add calibrated energy if parameters available
			if calib_params:
				a = calib_params.get('a', 1.0)
				a_err = calib_params.get('a_err', 0.0)
				c = calib_params.get('c', 0.0)
				c_err = calib_params.get('c_err', 0.0)
				
				# E = (mu - c) / a
				E = (mu - c) / a
				
				# Uncertainty propagation: dE/dmu = 1/a, dE/da = -(mu-c)/a^2, dE/dc = -1/a
				E_err_a = abs((mu - c) / (a**2)) * a_err if a != 0 else 0
				E_err_c = c_err / a if a != 0 else 0
				E_err_calib = np.sqrt(E_err_a**2 + E_err_c**2)
				E_err_mu = mu_total / a
				
				# Total energy uncertainty (add in quadrature)
				E_err_total = np.sqrt(E_err_mu**2 + E_err_calib**2)
				
				# Separate systematic and statistical energy uncertainties from peak fitting
				E_err_sys = mu_sys / a
				E_err_stat = mu_stat / a
				
				line += f" {E:.3f} {E_err_sys:.3f} {E_err_stat:.3f} {E_err_calib:.3f} {E_err_total:.3f}"
			
			outf.write(line + '\n')
	print(f'Wrote per-trial peaks to: {out_all_path}')

	# group by thickness and average three trials
	by_th = {}
	for fn, (th, trial, mu, mu_sys, mu_stat, model_name) in results.items():
		by_th.setdefault(th, []).append((mu, mu_sys, mu_stat))

	out_lines = []
	for th in sorted(by_th.keys()):
		items = by_th[th]
		mus = np.array([it[0] for it in items], dtype=float)
		sys_errs = np.array([it[1] for it in items], dtype=float)
		stat_errs = np.array([it[2] for it in items], dtype=float)
		
		# total uncertainty per trial = sqrt(sys^2 + stat^2)
		total_errs = np.sqrt(sys_errs**2 + stat_errs**2)
		total_errs[total_errs <= 0] = 1.0
		
		# weighted average by inverse square of total uncertainty
		weights = 1.0 / (total_errs ** 2)
		avg = float(np.sum(mus * weights) / np.sum(weights))
		
		# combined systematic: average in quadrature
		avg_sys = float(np.sqrt(np.mean(sys_errs**2)))
		
		# combined statistical: from weighted averaging formula
		avg_stat = float(math.sqrt(1.0 / np.sum(weights)))
		
		# combined total uncertainty
		avg_total = float(np.sqrt(avg_sys**2 + avg_stat**2))
		
		out_lines.append((th, avg, avg_sys, avg_stat, avg_total, len(items)))

	out_path = os.path.join(base_dir, 'peaks_by_thickness.txt')
	with open(out_path, 'w') as outf:
		if calib_params:
			outf.write('# thickness_um  avg_peak_channel  avg_sys_uncertainty  avg_stat_uncertainty  avg_total_uncertainty  n_trials  avg_energy_MeV  energy_sys_unc  energy_stat_unc  energy_calib_unc  energy_total_unc\n')
		else:
			outf.write('# thickness_um  avg_peak_channel  avg_sys_uncertainty  avg_stat_uncertainty  avg_total_uncertainty  n_trials\n')
		
		for th, avg, avg_sys, avg_stat, avg_total, n in sorted(out_lines):
			line = f"{th:.3f} {avg:.3f} {avg_sys:.3f} {avg_stat:.3f} {avg_total:.3f} {n}"
			
			# Add calibrated energy if parameters available
			if calib_params:
				a = calib_params.get('a', 1.0)
				a_err = calib_params.get('a_err', 0.0)
				c = calib_params.get('c', 0.0)
				c_err = calib_params.get('c_err', 0.0)
				
				# E = (avg - c) / a
				E_avg = (avg - c) / a
				
				# Uncertainty propagation from calibration
				E_err_a = abs((avg - c) / (a**2)) * a_err if a != 0 else 0
				E_err_c = c_err / a if a != 0 else 0
				E_err_calib = np.sqrt(E_err_a**2 + E_err_c**2)
				E_err_avg = avg_total / a
				
				# Total energy uncertainty
				E_err_total = np.sqrt(E_err_avg**2 + E_err_calib**2)
				
				# Separate systematic and statistical from peak fitting
				E_err_sys = avg_sys / a
				E_err_stat = avg_stat / a
				
				line += f" {E_avg:.3f} {E_err_sys:.3f} {E_err_stat:.3f} {E_err_calib:.3f} {E_err_total:.3f}"
			
			outf.write(line + '\n')

	print(f'Wrote averaged peaks to: {out_path}')

	# summary plot
	if out_lines:
		ths = [o[0] for o in out_lines]
		avgs = [o[1] for o in out_lines]
		total_errs = [o[4] for o in out_lines]  # combined total uncertainty
		plt.figure(figsize=(7,5))
		plt.errorbar(ths, avgs, yerr=total_errs, fmt='o-')
		plt.xlabel('Thickness (μm)')
		plt.ylabel('Average peak channel')
		plt.title('Peak channel vs thickness (averaged trials)')
		plt.grid(True)
		plt.tight_layout()
		plt.show()
		
		# If calibration available, also plot energy vs thickness
		if calib_params:
			a = calib_params.get('a', 1.0)
			a_err = calib_params.get('a_err', 0.0)
			c = calib_params.get('c', 0.0)
			c_err = calib_params.get('c_err', 0.0)
			
			Es = [(ch - c) / a for ch in avgs]
			E_errs_peak = [err/a for err in total_errs]
			E_errs_calib = [np.sqrt(((ch-c)/(a**2) * a_err)**2 + (c_err/a)**2) for ch in avgs]
			E_errs_total = [np.sqrt(ep**2 + ec**2) for ep, ec in zip(E_errs_peak, E_errs_calib)]
			
			fig, ax = plt.subplots(figsize=(8,6))
			ax.errorbar(ths, Es, yerr=E_errs_total, fmt='o-', color='green', label='Total uncertainty', capsize=5)
			
			# Add a text box showing the breakdown of uncertainties for the first point
			if len(ths) > 0:
				text_info = f'Calibration uncertainty:\na = {a:.4f} ± {a_err:.4f}\nc = {c:.2f} ± {c_err:.2f}\n\nCalib effect at E≈{Es[0]:.2f} MeV:\n{E_errs_calib[0]:.4f} MeV'
				ax.text(0.98, 0.97, text_info, transform=ax.transAxes, va='top', ha='right', 
						bbox=dict(facecolor='wheat', alpha=0.8), fontsize=9)
			
			ax.set_xlabel('Thickness (μm)')
			ax.set_ylabel('Average peak energy (MeV)')
			ax.set_title('Calibrated energy vs thickness (with peak fitting + calibration uncertainties)')
			ax.grid(True)
			plt.tight_layout()
			plt.show()


if __name__ == '__main__':
	main()

