import os
import re
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.optimize import curve_fit
except Exception:
    curve_fit = None

# Set font for plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

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
            # stop if we reach a new section marker like $ROI or any $KEY
            if s.startswith('$'):
                break
            if not s:
                continue
            # ignore comment markers used in the attachments preview
            if s.startswith('/*') or s.startswith('Lines'):
                continue
            # split tokens and parse integers
            for tok in s.split():
                try:
                    counts.append(int(tok))
                except ValueError:
                    # skip non-integer tokens
                    pass
    return counts

def plot_counts(counts, title):
    channels = list(range(len(counts)))
    plt.figure(figsize=(10,6))
    plt.bar(channels, counts, width=1.0, edgecolor='black')
    plt.xlabel('Channel', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.title(title, fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.show()


def generalized_normal(x, A, mu, sigma, beta):
    x = np.array(x, dtype=float)
    z = np.abs((x - mu) / sigma)
    return A * np.exp(- (z ** beta))

def fit_generalized_normal(counts, fit_range):
    """
    Fit A, mu, sigma, beta for the generalized normal over fit_range.
    Returns x, y, model, popt, perr, chi2, red_chi2
    """
    if curve_fit is None:
        raise RuntimeError('scipy.optimize.curve_fit not available; install scipy to use fitting')

    start, end = fit_range
    x = np.arange(start, end + 1)
    y = np.array(counts[start:end + 1], dtype=float)
    if len(x) < 4:
        raise ValueError('fit range too small')

    sigma_y = np.sqrt(y)
    sigma_y[sigma_y == 0] = 1.0

    # initial guesses
    A0 = max(y) if len(y) else 1.0
    mu0 = start + np.argmax(y)
    sigma0 = max(1.0, (end - start) / 10.0)
    beta0 = 2.0

    p0 = [A0, mu0, sigma0, beta0]
    bounds_lower = [0.0, start, 0.01, 0.1]
    bounds_upper = [np.inf, end, (end - start), 10.0]

    popt, pcov = curve_fit(
        generalized_normal, x, y, p0=p0, sigma=sigma_y, absolute_sigma=True, bounds=(bounds_lower, bounds_upper)
    )

    model = generalized_normal(x, *popt)
    chi2 = np.sum(((y - model) / sigma_y) ** 2)
    dof = max(1, len(x) - len(popt))
    red_chi2 = chi2 / dof

    perr = np.sqrt(np.diag(pcov))
    return x, y, model, popt, perr, chi2, red_chi2

def main():
    # files to process 
    filenames = [
        'alpha 1x.txt',
        'alpha 0.5x.txt',
        'alpha 0.2x.txt',
        'alpha 0.1x.txt',
    ]

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Set the isotope energy (E0) for the `1x` file here (in MeV).
    E0_VALUE = 5.806

    # Provide peak channel per file here (edit these values to assign the peak channel)
    # Example: PEAK_CHANNELS = {'alpha 1x.txt': 1600, 'alpha 0.5x.txt': 800, ...}
    PEAK_CHANNELS = {
        'alpha 1x.txt': 1602,
        'alpha 0.5x.txt': 802,
        'alpha 0.2x.txt': 318,
        'alpha 0.1x.txt': 157,
    }

    # Optional per-file gap overrides (distance in channels from peak to fit edge).
    # If not present for a file, `GAP_DISTANCE` below is used.
    PEAK_GAPS = {
         'alpha 1x.txt': 7,
    }

    # collect peak channel (mu) and its uncertainties for each file (used later)
    peak_mus = []
    peak_mus_sys = []  # systematic uncertainties
    peak_mus_stat = []  # statistical uncertainties
    valid_energies_labels = []

    # user-tunable gap distance from peak for fit range edges
    GAP_DISTANCE = 8

    # search span (how far from peak to look for candidate fit range edges)
    SEARCH_SPAN = 20

    # reduced-chi2 window: accept fits within center ± window
    # e.g., RED_CHI_CENTER=1.0, RED_CHI_WINDOW=0.1 gives [0.9, 1.1]
    RED_CHI_CENTER = 1.0
    RED_CHI_WINDOW = 0.3

    # --- model definitions for multi-model fitting ---
    try:
        from scipy.special import erfc
    except Exception:
        erfc = None

    def model_gaussian(x, A, mu, sigma, c):
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c

    def model_generalized_normal(x, A, mu, sigma, beta, c):
        z = np.abs((x - mu) / sigma)
        return A * np.exp(- (z ** beta)) + c

    def model_crystalball(x, A, mu, sigma, alpha, n, c):
        x = np.array(x, dtype=float)
        t = (x - mu) / sigma
        out = np.empty_like(t)
        # constants
        abs_alpha = np.abs(alpha)
        # compute A_cb in log-space to avoid overflow
        with np.errstate(divide='ignore', invalid='ignore'):
            logA_cb = n * (np.log(n) - np.log(abs_alpha + 1e-30)) - 0.5 * (abs_alpha ** 2)
            logA_cb = np.clip(logA_cb, -700, 700)
            A_cb = np.exp(logA_cb)
        B = n / (abs_alpha + 1e-30) - abs_alpha
        mask_gauss = t > -abs_alpha
        out[mask_gauss] = np.exp(-0.5 * t[mask_gauss] ** 2)
        # safe tail calculation: avoid negative base for power
        tt = B - t[~mask_gauss]
        tt_safe = np.maximum(tt, 1e-12)
        with np.errstate(over='ignore', invalid='ignore'):
            out[~mask_gauss] = A_cb * (tt_safe ** (-n))
        return A * out + c

    def model_gauss_one_sided_exp(x, A, mu, sigma, lam, c):
        x = np.array(x, dtype=float)
        gauss = A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        # exponential tail starts at mu
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

    model_funcs = {
        'Gaussian': (model_gaussian, 4),
        'GeneralizedNormal': (model_generalized_normal, 5),
        'CrystalBall': (model_crystalball, 6),
        'GaussOneSidedExp': (model_gauss_one_sided_exp, 5),
        'Polynomial4': (model_polynomial4, 6),
        'PolyExpo': (model_poly_expo, 6),
    }

    for idx, fn in enumerate(filenames):
        path = os.path.join(base_dir, fn)
        if not os.path.isfile(path):
            print(f"Warning: file not found: {path}")
            continue
        counts = parse_counts(path)
        if not counts:
            print(f"No data parsed from {fn}")
            continue
        
        # extract scale factor from filename for cleaner titles
        scale_label = fn.replace('alpha ', '').replace('.txt', '')
        title = f"Counts vs Channel — {scale_label}"
        
        # plot the histogram first
        plot_counts(counts, title)
        # use configured peak channels (edit PEAK_CHANNELS at top of script)
        peak_ch = PEAK_CHANNELS.get(fn, None)
        if peak_ch is None:
            print(f'No peak channel provided for {fn}; set PEAK_CHANNELS in the script to enable fitting for this file.')
            continue
        gap = PEAK_GAPS.get(fn, GAP_DISTANCE)

        # candidate start/end ranges (must be at least `gap` away from peak)
        n_channels = len(counts)
        start_min = max(0, peak_ch - SEARCH_SPAN)
        start_max = max(0, peak_ch - gap - 1)
        end_min = min(n_channels - 1, peak_ch + gap + 1)
        end_max = min(n_channels - 1, peak_ch + SEARCH_SPAN)

        # Collect accepted fits grouped by model
        accepted_by_model = {name: [] for name in model_funcs.keys()}
        accepted_errs_by_model = {name: [] for name in model_funcs.keys()}
        best_fits_info = []  # track all accepted fits for top 4 selection
        red_chi_min = RED_CHI_CENTER - RED_CHI_WINDOW
        red_chi_max = RED_CHI_CENTER + RED_CHI_WINDOW

        for start in range(start_min, start_max + 1):
            for end in range(end_min, end_max + 1):
                if end - start < 6:
                    continue
                x_idx = np.arange(start, end + 1)
                y_vals = np.array(counts[start:end + 1], dtype=float)
                sigma_y = np.sqrt(y_vals)
                sigma_y[sigma_y == 0] = 1.0

                for name, (func, nargs) in model_funcs.items():
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

                        # accept fits within reduced-chi2 window
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
                            # track fit for top 4 selection
                            best_fits_info.append({
                                'name': name,
                                'x_idx': x_idx,
                                'y_vals': y_vals,
                                'popt': popt,
                                'pcov': pcov,
                                'chi2': chi2,
                                'red_chi2': red_chi2,
                                'func': func,
                            })
                    except Exception:
                        continue

        # flatten all accepted values across models for overall statistics
        all_vals = []
        all_errs = []
        for name in accepted_by_model:
            all_vals.extend(accepted_by_model[name])
            all_errs.extend(accepted_errs_by_model[name])

        if len(all_vals) == 0:
            print(f'No fits within reduced-chi2 window for {fn} (need between {red_chi_min:.2f} and {red_chi_max:.2f}).')
            continue

        # remove outliers using IQR method
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
            print(f'All fits were outliers for {fn}; no valid data remains.')
            continue
        
        n_removed = len(all_vals) - len(all_vals_filtered)
        if n_removed > 0:
            print(f'Removed {n_removed} outliers from {fn} distribution.')
        
        all_vals = all_vals_filtered
        all_errs = all_errs_filtered

        # compute distribution statistics
        mus_arr = np.array(all_vals, dtype=float)
        mus_errs_arr = np.array(all_errs, dtype=float)
        mu_mean = float(np.mean(mus_arr))
        mu_std = float(np.std(mus_arr, ddof=0))
        mu_stat_avg = float(np.mean(mus_errs_arr))

        # stacked histogram per model
        plt.figure(figsize=(10,6))
        model_names = [name for name in model_funcs.keys() if len(accepted_by_model[name]) > 0]
        data = [np.array(accepted_by_model[name], dtype=float) for name in model_names]
        # color mapping for models
        color_map = {
            'Gaussian': 'C0',
            'GeneralizedNormal': 'C1',
            'CrystalBall': 'C2',
            'GaussOneSidedExp': 'C3',
            'Polynomial4': 'C4',
            'PolyExpo': 'C5',
        }
        colors = [color_map.get(name, None) for name in model_names]
        n_bins = max(6, int(min(50, len(mus_arr)//1)))
        plt.hist(data, bins=n_bins, stacked=True, color=colors, label=model_names, edgecolor='black', alpha=0.8)
        plt.xlabel('Fitted Peak (Channel)', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.title(title + ' — Distribution of Accepted Peak Values', fontsize=16)
        plt.axvline(mu_mean, color='r', linestyle='--', linewidth=2, label=f'mean = {mu_mean:.2f}')
        annot = f'mean = {mu_mean:.2f}\nsys = {mu_std:.3f}\nstat = {mu_stat_avg:.3f}\nN = {len(mus_arr)}'
        plt.legend(loc='upper left', fontsize=11)
        plt.text(0.98, 0.95, annot, transform=plt.gca().transAxes, va='top', ha='right', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()
        plt.show()

        # Plot top 4 best fits in 2x2 subplots
        if len(best_fits_info) > 0:
            # sort by reduced chi2 distance from 1.0
            best_fits_info_sorted = sorted(best_fits_info, key=lambda x: abs(x['red_chi2'] - 1.0))
            top_4 = best_fits_info_sorted[:min(4, len(best_fits_info_sorted))]
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            # margin to extend the view beyond the fit region (in channels)
            VIEW_MARGIN = 20
            
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
                    mu_err = float(perr[1])
                else:
                    mu_err = np.nan
                
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
                
                ax.set_xlabel('Channel', fontsize=12)
                ax.set_ylabel('Counts', fontsize=12)
                if np.isfinite(mu_val) and np.isfinite(mu_err):
                    ax.set_title(f'{name}\nμ = {mu_val:.2f} ± {mu_err:.2f}\nχ²_red = {red_chi2:.3f}', fontsize=12)
                else:
                    ax.set_title(f'{name}\nχ²_red = {red_chi2:.3f}', fontsize=12)
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.legend(fontsize=10)
            
            # hide unused subplots
            for idx in range(len(top_4), 4):
                axes[idx].axis('off')
            
            fig.suptitle(title + ' — Top 4 Best Fits', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()

        # store mean (final peak value) and both uncertainties for later linear fit
        peak_mus.append(mu_mean)
        peak_mus_sys.append(mu_std)  # systematic: std of distribution
        peak_mus_stat.append(mu_stat_avg)  # statistical: avg fit error
        valid_energies_labels.append(fn)

    # After all per-file fits, perform linear fit of channel vs energy (E -> channel)
    if len(peak_mus) < 2:
        print('Not enough fitted peaks to perform linear fit (need >=2).')
        return

    # Use the hardcoded E0_VALUE above. Abort if not set.
    if E0_VALUE is None:
        print('E0_VALUE is not set in the script. Please edit the file and set E0_VALUE = <numeric MeV>.')
        return
    E0 = float(E0_VALUE)

    # compute energies matching filenames order: 1x, 0.5x, 0.2x, 0.1x
    scale_map = {
        'alpha 1x.txt': 1.0,
        'alpha 0.5x.txt': 0.5,
        'alpha 0.2x.txt': 0.2,
        'alpha 0.1x.txt': 0.1,
    }

    energies = []
    for fn in valid_energies_labels:
        scale = scale_map.get(fn, None)
        if scale is None:
            # attempt to parse a number like '1x' from filename
            if '1x' in fn:
                scale = 1.0
            elif '0.5x' in fn:
                scale = 0.5
            elif '0.2x' in fn:
                scale = 0.2
            elif '0.1x' in fn:
                scale = 0.1
            else:
                scale = 1.0
        energies.append(E0 * scale)

    # perform linear fit channel = a * E + c
    def linear_model(x, a, c):
        return a * x + c

    if curve_fit is None:
        print('scipy.optimize.curve_fit not available; cannot perform linear fit.')
        return

    x = np.array(energies, dtype=float)
    y = np.array(peak_mus, dtype=float)
    sys_errs = np.array(peak_mus_sys, dtype=float)
    stat_errs = np.array(peak_mus_stat, dtype=float)
    
    # total uncertainty = sqrt(sys^2 + stat^2)
    sigma_y = np.sqrt(sys_errs**2 + stat_errs**2)
    sigma_y[sigma_y == 0] = 1.0

    p0 = [ (y[-1]-y[0])/(x[-1]-x[0]) if x[-1]!=x[0] else 1.0, np.mean(y) ]

    popt_lin, pcov_lin = curve_fit(linear_model, x, y, p0=p0, sigma=sigma_y, absolute_sigma=True)
    perr_lin = np.sqrt(np.diag(pcov_lin))

    y_model = linear_model(x, *popt_lin)
    chi2_lin = np.sum(((y - y_model) / sigma_y) ** 2)
    dof_lin = max(1, len(x) - len(popt_lin))
    red_chi2_lin = chi2_lin / dof_lin

    a, c = popt_lin
    a_err, c_err = perr_lin

    # Plot final result
    plt.figure(figsize=(10,6))
    plt.errorbar(x, y, yerr=sigma_y, fmt='o', markersize=8, capsize=5, capthick=2, label='Measured peaks (total uncertainty)')
    xs = np.linspace(min(x)*0.9, max(x)*1.1, 200)
    plt.plot(xs, linear_model(xs, *popt_lin), 'r-', linewidth=2, label=f'Fit: channel = a*E + c')
    plt.xlabel('Energy (MeV)', fontsize=14)
    plt.ylabel('Peak Channel', fontsize=14)
    plt.title(f'Linear Fit of Peak Channel vs Energy ($E_0$={E0} MeV)', fontsize=16)
    txt2 = (
        f'a = {a:.4f} ± {a_err:.4f}\n'
        f'c = {c:.2f} ± {c_err:.2f}\n'
        f'χ² = {chi2_lin:.2f}\nχ²_red = {red_chi2_lin:.3f}'
    )
    plt.legend(fontsize=12)
    plt.text(0.02, 0.95, txt2, transform=plt.gca().transAxes, va='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Save calibration parameters to file for use in E_spec_Al.py
    calib_path = os.path.join(base_dir, 'calibration_params.txt')
    with open(calib_path, 'w') as f:
        f.write('# Calibration parameters: channel = a * E + c, or E = (channel - c) / a\n')
        f.write(f'a = {a}\n')
        f.write(f'a_err = {a_err}\n')
        f.write(f'c = {c}\n')
        f.write(f'c_err = {c_err}\n')
    print(f'Wrote calibration parameters to: {calib_path}')


if __name__ == '__main__':
    main()
