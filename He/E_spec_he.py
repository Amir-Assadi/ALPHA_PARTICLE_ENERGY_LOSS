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

# --- Plot controls ---
# Set SHOW_PLOTS to False to skip all plot pop-ups.
# Set SHOW_DISTRIBUTION_PLOT to True to always show the model distribution plot.
SHOW_PLOTS = False
SHOW_DISTRIBUTION_PLOT = False

# --- User configuration (He stations) ---
# Geometry + gas reference values (editable)
X0_MM = 144.5
P0_MBAR = 1013.25
RHO0_G_CM3 = 0.0001785
GAS_LABEL = 'Helium gas'

# Station 3 has 3 sources (Pu, Am, Cm) in the same spectrum.
STATION3_SOURCES = [
	'pu',
	'am',
	'cm',
]

# Station 4 has 1 source (Th).
STATION4_SOURCES = [
	'th',
]

PEAK_CHANNELS_HE3 = {
	# Fill these with your manual peak channels for HE3 files.
	#  'T0_0156he3.txt': {'pu': 877, 'am': 934, 'cm': 990},
}

# Optional: specify fit gaps per source (if not specified, GAP_DISTANCE=8 is used)
PEAK_GAPS_HE3 = {
    # Optional: per-source fit gaps for HE3 files.
    # 'T0_0156he3.txt': {'pu': 4, 'am': 4, 'cm': 1},
}

# Per-file peak channels for HE4 (single peak per file)
PEAK_CHANNELS_HE4 = {
	'T0_00353he4.txt': {'th': 1496},
	'T140he4.txt': {'th': 1378},
	'T215he4.txt': {'th': 1313},
	'T276he4.txt': {'th': 1259},
	'T344he4.txt': {'th': 1200},
	'T418he4.txt': {'th': 1125},
	'T506he4.txt': {'th': 1040},
	'T568he4.txt': {'th': 975},
	'T617he4.txt': {'th': 918},
	'T660he4.txt': {'th': 860},
	'T700he4.txt': {'th': 816},
	'T752he4.txt': {'th': 741},
	'T800he4.txt': {'th': 686},
	'T849he4.txt': {'th': 607},
	'T903he4.txt': {'th': 513},
	'T960he4.txt': {'th': None},
	'T1005he4.txt': {'th': 346},
}

# Optional per-file fit gaps for HE4 (single peak per file)
PEAK_GAPS_HE4 = {
	'T0_00353he4.txt': {'th': 8},
	'T140he4.txt': {'th': 20},
	'T215he4.txt': {'th': 20},
	'T276he4.txt': {'th': 23},
	'T344he4.txt': {'th': 25},
	'T418he4.txt': {'th': 24},
	'T506he4.txt': {'th': 25},
	'T568he4.txt': {'th': 25},
	'T617he4.txt': {'th': 25},
	'T660he4.txt': {'th': 25},
	'T700he4.txt': {'th': 25},
	'T752he4.txt': {'th':40},
	'T800he4.txt': {'th': 30},
	'T849he4.txt': {'th':45},
	'T903he4.txt': {'th': 40},
	'T960he4.txt': {'th': None},
	'T1005he4.txt': {'th': 47},
}

# default gap distance from peak for fit range edges
GAP_DISTANCE = 8
# base constant for search span calculation: search_span = SEARCH_SPAN_BASE + peak_gap
SEARCH_SPAN_BASE = 12
# reduced-chi2 window: accept fits within center ± window
# e.g., RED_CHI_CENTER=1.0, RED_CHI_WINDOW=0.1 gives [0.9, 1.1]
RED_CHI_CENTER = 1.0
RED_CHI_WINDOW = 0.2


def parse_counts(filepath):
	counts = []
	start = False
	with open(filepath, 'r', errors='ignore') as f:
		for line in f:
			s = line.strip()
			if not start:
				if re.match(r'^0\s+\d+\b', s):
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


def parse_pressure_mbar(filename, station_tag):
	pattern = rf'^T(?P<p>\d+(?:_\d+)?){re.escape(station_tag)}\.txt$'
	m = re.match(pattern, filename, re.IGNORECASE)
	if not m:
		return None
	p_raw = m.group('p')
	try:
		return float(p_raw.replace('_', '.'))
	except ValueError:
		return None


def pressure_to_thickness_mm(pressure_mbar):
	return (pressure_mbar / P0_MBAR) * X0_MM


def format_thickness(thickness_mm):
	"""Convert thickness in mm to appropriate units and return (value, unit_str)"""
	if thickness_mm < 1.0:
		return thickness_mm * 1000.0, 'μm'
	elif thickness_mm < 100.0:
		return thickness_mm, 'mm'
	elif thickness_mm < 10000.0:
		return thickness_mm / 10.0, 'cm'
	else:
		return thickness_mm / 1000.0, 'm'


def load_calibration_params(calib_path):
	if not os.path.isfile(calib_path):
		return None
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
		return calib_params
	except Exception as e:
		print(f'Warning: Could not load calibration parameters from {calib_path}: {e}')
		return None


def apply_calibration(mu, mu_sys, mu_stat, calib_params):
	if not calib_params:
		return None
	a = calib_params.get('a', 1.0)
	a_err = calib_params.get('a_err', 0.0)
	c = calib_params.get('c', 0.0)
	c_err = calib_params.get('c_err', 0.0)
	if a == 0:
		return None

	E = (mu - c) / a
	mu_total = np.sqrt(mu_sys**2 + mu_stat**2)
	E_err_mu = mu_total / a
	E_err_a = abs((mu - c) / (a**2)) * a_err
	E_err_c = c_err / a
	E_err_calib = np.sqrt(E_err_a**2 + E_err_c**2)
	E_err_total = np.sqrt(E_err_mu**2 + E_err_calib**2)
	E_err_sys = mu_sys / a
	E_err_stat = mu_stat / a
	return {
		'E': E,
		'E_err_sys': E_err_sys,
		'E_err_stat': E_err_stat,
		'E_err_calib': E_err_calib,
		'E_err_total': E_err_total,
	}


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

	all_vals = []
	all_errs = []
	for name in accepted_by_model:
		all_vals.extend(accepted_by_model[name])
		all_errs.extend(accepted_errs_by_model[name])

	if len(all_vals) == 0:
		return None, None, None, None

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


def process_station(base_dir, station_tag, sources, peak_channels, peak_gaps, calib_params):
	files = os.listdir(base_dir)
	station_files = []
	for fn in files:
		pressure = parse_pressure_mbar(fn, station_tag)
		if pressure is not None:
			station_files.append((pressure, fn))

	if not station_files:
		print(f'No {station_tag.upper()} files found.')
		return

	results = []
	for pressure_mbar, fn in sorted(station_files):
		path = os.path.join(base_dir, fn)
		counts = parse_counts(path)
		if not counts:
			print(f'No data in {fn}')
			continue

		thickness_mm = pressure_to_thickness_mm(pressure_mbar)
		thickness_um = thickness_mm * 1000.0
		thickness_val, thickness_unit = format_thickness(thickness_mm)

		# Plot raw spectrum once per file (before looping through sources)
		channels = np.arange(len(counts))
		plt.figure(figsize=(10,4))
		plt.bar(channels, counts, width=1.0, edgecolor='black', alpha=0.9)
		plt.xlabel('Channel')
		plt.ylabel('Counts')
		plt.title(f'{GAS_LABEL} {station_tag.upper()} - Thickness = {thickness_val:.3f} {thickness_unit}')
		plt.tight_layout()
		if SHOW_PLOTS:
			plt.show()
		plt.close()

		for source in sources:
			peak_info = peak_channels.get(fn, None)
			peak_ch = None
			if isinstance(peak_info, dict):
				peak_ch = peak_info.get(source, None)
			else:
				peak_ch = peak_info

			if peak_ch is None:
				print(f'No peak channel for {fn} source {source}. Skipping.')
				continue
			peak_ch = int(peak_ch)

			if curve_fit is None:
				print('scipy not available; cannot fit. Using argmax fallback.')
				mu = float(peak_ch)
				mu_sys = 0.0
				mu_stat = 1.0
				model_name = 'argmax'
				best_fits_info = []
				accepted_by_model = {}
				all_vals = []
				all_errs = []
			else:
				gap_info = peak_gaps.get(fn, None)
				gap = GAP_DISTANCE
				if isinstance(gap_info, dict):
					gap = gap_info.get(source, GAP_DISTANCE)
				elif gap_info is not None:
					gap = gap_info
				# Ensure gap is not None (default to GAP_DISTANCE)
				if gap is None:
					gap = GAP_DISTANCE
				search_span = SEARCH_SPAN_BASE + gap
				accepted_by_model, best_fits_info, all_vals, all_errs = find_all_fits(
					counts,
					peak_ch,
					gap=gap,
					search_span=search_span,
					red_chi_center=RED_CHI_CENTER,
					red_chi_window=RED_CHI_WINDOW,
				)
				if all_vals is None or len(all_vals) == 0:
					mu = float(peak_ch)
					mu_sys = 0.0
					mu_stat = 1.0
					model_name = 'fallback'
				else:
					mus_arr = np.array(all_vals, dtype=float)
					mus_errs_arr = np.array(all_errs, dtype=float)
					mu = float(np.mean(mus_arr))
					mu_sys = float(np.std(mus_arr, ddof=0))
					mu_stat = float(np.mean(mus_errs_arr))
					model_name = 'ensemble'

					plt.figure(figsize=(8,4))
					model_names = [name for name in MODEL_FUNCS.keys() if len(accepted_by_model[name]) > 0]
					data_by_model = {}
					for name in model_names:
						all_model_vals = np.array(accepted_by_model[name], dtype=float)
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
					n_bins = max(6, int(min(50, len(mus_arr) // 1)))
					plt.hist(data, bins=n_bins, stacked=True, color=colors, label=model_names, edgecolor='black', alpha=0.8)
					plt.xlabel('Fitted peak (μ)')
					plt.ylabel('Count')
					plt.title(f'{GAS_LABEL} {station_tag.upper()} {source.upper()} - Peak distribution')
					plt.axvline(mu, color='r', linestyle='--', linewidth=2, label=f'mean = {mu:.2f}')
					annot = f'mean = {mu:.2f}\nsys = {np.std(mus_arr, ddof=0):.3f}\nstat = {mu_stat:.3f}\nN = {len(mus_arr)}'
					plt.legend(loc='upper left')
					plt.text(0.98, 0.95, annot, transform=plt.gca().transAxes, va='top', ha='right', bbox=dict(facecolor='white', alpha=0.8))
					plt.tight_layout()
					if SHOW_PLOTS or SHOW_DISTRIBUTION_PLOT:
						plt.show()
					plt.close()

					if len(best_fits_info) > 0:
						n_channels = len(counts)
						view_margin = 20
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

							x_min_fit = int(x_idx[0])
							x_max_fit = int(x_idx[-1])
							x_min_view = max(0, x_min_fit - view_margin)
							x_max_view = min(n_channels - 1, x_max_fit + view_margin)

							x_extended = np.arange(x_min_view, x_max_view + 1)
							y_extended = np.array(counts[x_min_view:x_max_view + 1], dtype=float)
							ax.bar(x_extended, y_extended, width=1.0, edgecolor='black', alpha=0.6, label='Data')
							y_model = func(x_idx, *popt)
							ax.plot(x_idx, y_model, 'r-', linewidth=2, label='Fit')
							ax.set_xlabel('Channel')
							ax.set_ylabel('Counts')
							if np.isfinite(mu_val) and np.isfinite(mu_err_fit):
								ax.set_title(f'{name}\nμ = {mu_val:.2f} ± {mu_err_fit:.2f}\nχ²_red = {red_chi2:.3f}')
							else:
								ax.set_title(f'{name}\nχ²_red = {red_chi2:.3f}')
							ax.legend()

						for idx in range(len(top_4), 4):
							axes[idx].axis('off')
						fig.suptitle(f'{GAS_LABEL} {station_tag.upper()} {source.upper()} - Top 4 Best Fits', fontsize=14, fontweight='bold')
						plt.tight_layout()
						if SHOW_PLOTS:
							plt.show()
						plt.close(fig)

			calib_info = apply_calibration(mu, mu_sys, mu_stat, calib_params)
			results.append({
				'filename': fn,
				'pressure_mbar': pressure_mbar,
				'thickness_mm': thickness_mm,
				'thickness_um': thickness_um,
				'source': source,
				'mu': mu,
				'mu_sys': mu_sys,
				'mu_stat': mu_stat,
				'model': model_name,
				'calib': calib_info,
			})

	if not results:
		print(f'No results for {station_tag.upper()}')
		return

	all_out_path = os.path.join(base_dir, f'peaks_all_trials_{station_tag}.txt')
	with open(all_out_path, 'w') as outf:
		if calib_params:
			outf.write('# filename pressure_mbar thickness_mm thickness_um source peak_channel sys_uncertainty stat_uncertainty total_uncertainty model energy_MeV energy_sys_unc energy_stat_unc energy_calib_unc energy_total_unc\n')
		else:
			outf.write('# filename pressure_mbar thickness_mm thickness_um source peak_channel sys_uncertainty stat_uncertainty total_uncertainty model\n')
		for row in results:
			mu_total = np.sqrt(row['mu_sys']**2 + row['mu_stat']**2)
			line = (
				f"{row['filename']} {row['pressure_mbar']:.4f} "
				f"{row['thickness_mm']:.6f} {row['thickness_um']:.3f} "
				f"{row['source']} {row['mu']:.3f} {row['mu_sys']:.3f} {row['mu_stat']:.3f} {mu_total:.3f} {row['model']}"
			)
			if row['calib']:
				c = row['calib']
				line += f" {c['E']:.3f} {c['E_err_sys']:.3f} {c['E_err_stat']:.3f} {c['E_err_calib']:.3f} {c['E_err_total']:.3f}"
			outf.write(line + '\n')
	print(f'Wrote per-file peaks to: {all_out_path}')

	by_key = {}
	for row in results:
		key = (row['pressure_mbar'], row['source'])
		by_key.setdefault(key, []).append(row)

	out_lines = []
	for (pressure_mbar, source), items in sorted(by_key.items()):
		mus = np.array([it['mu'] for it in items], dtype=float)
		sys_errs = np.array([it['mu_sys'] for it in items], dtype=float)
		stat_errs = np.array([it['mu_stat'] for it in items], dtype=float)
		total_errs = np.sqrt(sys_errs**2 + stat_errs**2)
		total_errs[total_errs <= 0] = 1.0
		weights = 1.0 / (total_errs ** 2)
		avg = float(np.sum(mus * weights) / np.sum(weights))
		avg_sys = float(np.sqrt(np.mean(sys_errs**2)))
		avg_stat = float(math.sqrt(1.0 / np.sum(weights)))
		avg_total = float(np.sqrt(avg_sys**2 + avg_stat**2))
		thickness_mm = pressure_to_thickness_mm(pressure_mbar)
		thickness_um = thickness_mm * 1000.0
		out_lines.append((pressure_mbar, thickness_mm, thickness_um, source, avg, avg_sys, avg_stat, avg_total, len(items)))

	avg_out_path = os.path.join(base_dir, f'peaks_by_thickness_{station_tag}.txt')
	with open(avg_out_path, 'w') as outf:
		if calib_params:
			outf.write('# pressure_mbar thickness_mm thickness_um source avg_peak_channel avg_sys_unc avg_stat_unc avg_total_unc n_trials avg_energy_MeV energy_sys_unc energy_stat_unc energy_calib_unc energy_total_unc\n')
		else:
			outf.write('# pressure_mbar thickness_mm thickness_um source avg_peak_channel avg_sys_unc avg_stat_unc avg_total_unc n_trials\n')
		for pressure_mbar, thickness_mm, thickness_um, source, avg, avg_sys, avg_stat, avg_total, n in out_lines:
			line = f"{pressure_mbar:.4f} {thickness_mm:.6f} {thickness_um:.3f} {source} {avg:.3f} {avg_sys:.3f} {avg_stat:.3f} {avg_total:.3f} {n}"
			if calib_params:
				calib_info = apply_calibration(avg, avg_sys, avg_stat, calib_params)
				if calib_info:
					line += f" {calib_info['E']:.3f} {calib_info['E_err_sys']:.3f} {calib_info['E_err_stat']:.3f} {calib_info['E_err_calib']:.3f} {calib_info['E_err_total']:.3f}"
			outf.write(line + '\n')
	print(f'Wrote averaged peaks to: {avg_out_path}')

	if out_lines:
		plt.figure(figsize=(7,5))
		# Determine appropriate unit for x-axis (use max thickness)
		max_thickness_mm = max([o[1] for o in out_lines])
		_, x_unit = format_thickness(max_thickness_mm)
		for source in sources:
			thicknesses_mm = [o[1] for o in out_lines if o[3] == source]
			if x_unit == 'μm':
				xs = [t * 1000.0 for t in thicknesses_mm]
			elif x_unit == 'mm':
				xs = thicknesses_mm
			elif x_unit == 'cm':
				xs = [t / 10.0 for t in thicknesses_mm]
			else:  # meters
				xs = [t / 1000.0 for t in thicknesses_mm]
			ys = [o[4] for o in out_lines if o[3] == source]
			errs = [o[7] for o in out_lines if o[3] == source]
			if xs:
				plt.errorbar(xs, ys, yerr=errs, fmt='o-', label=source.upper())
		plt.xlabel(f'Thickness ({x_unit})')
		plt.ylabel('Average peak channel')
		plt.title(f'{GAS_LABEL} {station_tag.upper()} - Peak channel vs thickness')
		plt.grid(True)
		plt.legend()
		plt.tight_layout()
		if SHOW_PLOTS:
			plt.show()
		plt.close()

		if calib_params:
			plt.figure(figsize=(7,5))
			# Determine appropriate unit for x-axis (use max thickness)
			max_thickness_mm = max([o[1] for o in out_lines])
			_, x_unit = format_thickness(max_thickness_mm)
			for source in sources:
				thicknesses_mm = [o[1] for o in out_lines if o[3] == source]
				if x_unit == 'μm':
					xs = [t * 1000.0 for t in thicknesses_mm]
				elif x_unit == 'mm':
					xs = thicknesses_mm
				elif x_unit == 'cm':
					xs = [t / 10.0 for t in thicknesses_mm]
				else:  # meters
					xs = [t / 1000.0 for t in thicknesses_mm]
				avgs = [o[4] for o in out_lines if o[3] == source]
				avg_sys = [o[5] for o in out_lines if o[3] == source]
				avg_stat = [o[6] for o in out_lines if o[3] == source]
				if not xs:
					continue
				calib_es = []
				calib_errs = []
				for avg, s_err, st_err in zip(avgs, avg_sys, avg_stat):
					calib_info = apply_calibration(avg, s_err, st_err, calib_params)
					if calib_info:
						calib_es.append(calib_info['E'])
						calib_errs.append(calib_info['E_err_total'])
				plt.errorbar(xs, calib_es, yerr=calib_errs, fmt='o-', label=source.upper())
			plt.xlabel(f'Thickness ({x_unit})')
			plt.ylabel('Average energy (MeV)')
			plt.title(f'{GAS_LABEL} {station_tag.upper()} - Energy vs thickness')
			plt.grid(True)
			plt.legend()
			plt.tight_layout()
			if SHOW_PLOTS:
				plt.show()
			plt.close()


def main():
	base_dir = os.path.dirname(os.path.abspath(__file__))

	calib_he3 = load_calibration_params(os.path.join(base_dir, 'calibration_params_he3.txt'))
	if calib_he3:
		print(f"Loaded HE3 calibration: a={calib_he3.get('a', 'N/A')}, c={calib_he3.get('c', 'N/A')}")
	else:
		print('Note: calibration_params_he3.txt not found. HE3 energy calibration will not be applied.')

	calib_he4 = load_calibration_params(os.path.join(base_dir, 'calibration_params_he4.txt'))
	if calib_he4:
		print(f"Loaded HE4 calibration: a={calib_he4.get('a', 'N/A')}, c={calib_he4.get('c', 'N/A')}")
	else:
		print('Note: calibration_params_he4.txt not found. HE4 energy calibration will not be applied.')

	process_station(
		base_dir,
		'he3',
		STATION3_SOURCES,
		PEAK_CHANNELS_HE3,
		PEAK_GAPS_HE3,
		calib_he3,
	)
	process_station(
		base_dir,
		'he4',
		STATION4_SOURCES,
		PEAK_CHANNELS_HE4,
		PEAK_GAPS_HE4,
		calib_he4,
	)


if __name__ == '__main__':
	main()

