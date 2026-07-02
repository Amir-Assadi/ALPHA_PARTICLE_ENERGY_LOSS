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
                # Look for either "0 2047" or "0 1023" or "0 <any_number>" pattern
                if re.match(r'^0\s+\d+\b', s):
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




def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # ===== STATION 3 (N3) CONFIGURATION =====
    station3_config = {
        'station_name': 'nit3',
        'filenames': [
            'alpha 1xnit3.txt',
            'alpha 0.5xnit3.txt',
            'alpha 0.2xnit3.txt',
            'alpha 0.1xnit3.txt',
        ],
        'E0_VALUE': 5.16,  # Set the isotope energy (E0) for the `1x` file here (in MeV)
    }

    # ===== STATION 4 (N4) CONFIGURATION =====
    station4_config = {
        'station_name': 'nit4',
        'filenames': [
            'alpha 1xnit4.txt',
            'alpha 0.5xnit4.txt',
            'alpha 0.2xnit4.txt',
            'alpha 0.1xnit4.txt',
        ],
        'E0_VALUE': 4.69,  # Set the isotope energy (E0) for the `1x` file here (in MeV)
    }

    # Process both stations
    for station_config in [station3_config, station4_config]:
        process_station(base_dir, station_config)


def process_station(base_dir, config):
    """Process calibration for a single station - simplified peak detection using histogram maximum."""
    filenames = config['filenames']
    E0_VALUE = config['E0_VALUE']
    station_name = config['station_name']
    station_label = station_name.upper().replace('NIT', 'N')

    # collect peak channel (mu) and its uncertainties for each file (used later)
    peak_mus = []
    peak_mus_sys = []  # systematic uncertainties
    peak_mus_stat = []  # statistical uncertainties
    valid_energies_labels = []

    for fn in filenames:
        path = os.path.join(base_dir, fn)
        if not os.path.isfile(path):
            print(f"Warning: file not found: {path}")
            continue
        counts = parse_counts(path)
        if not counts:
            print(f"No data parsed from {fn}")
            continue
        
        # extract scale factor from filename for cleaner titles
        scale_label = fn.replace('alpha ', '').replace('Alpha ', '').replace('.txt', '')
        scale_label = re.sub(r'(?i)nit', 'n', scale_label)
        title = f"Counts vs Channel — {scale_label}"
        
        # plot the histogram first
        plot_counts(counts, title)
        
        # Find the channel with the highest count (the peak)
        counts_arr = np.array(counts, dtype=float)
        peak_ch = int(np.argmax(counts_arr))
        peak_count = float(counts_arr[peak_ch])
        
        print(f'Peak found in {fn} at channel {peak_ch} with {peak_count:.0f} counts')
        
        # store peak with zero uncertainty (direct measurement)
        peak_mus.append(float(peak_ch))
        peak_mus_sys.append(0.0)  # systematic: 0
        peak_mus_stat.append(0.0)  # statistical: 0
        valid_energies_labels.append(fn)

    # After all per-file peaks, perform linear fit of channel vs energy
    if len(peak_mus) < 1:
        print('No peaks found for calibration.')
        return

    # Handle single peak case (no linear fit possible)
    if len(peak_mus) == 1:
        mu_mean = peak_mus[0]
        # Determine scale: check which energy scale this single peak corresponds to
        fn = valid_energies_labels[0]
        if f'1x{station_name}' in fn:
            scale = 1.0
        elif f'0.5x{station_name}' in fn:
            scale = 0.5
        elif f'0.2x{station_name}' in fn:
            scale = 0.2
        elif f'0.1x{station_name}' in fn:
            scale = 0.1
        else:
            scale = 1.0
        E0 = float(E0_VALUE) * scale
        
        print(f'\nSingle peak detected for {station_label}:')
        print(f'  Peak channel: {mu_mean:.2f}')
        print(f'  Energy: {E0:.2f} MeV')
        print(f'  No linear fit possible with single point.')
        
        # Save single peak as calibration point
        calib_path = os.path.join(base_dir, f'calibration_params_{station_name}.txt')
        with open(calib_path, 'w') as f:
            f.write(f'# Single peak calibration point for {station_label}\n')
            f.write(f'# Peak channel: {mu_mean:.2f}\n')
            f.write(f'# Energy: {E0:.2f} MeV\n')
            f.write(f'peak_channel = {mu_mean:.2f}\n')
            f.write(f'peak_energy = {E0:.2f}\n')
            f.write(f'a = nan\n')
            f.write(f'a_err = nan\n')
            f.write(f'c = nan\n')
            f.write(f'c_err = nan\n')
        print(f'Wrote single peak calibration to: {calib_path}')
        return

    # Use the hardcoded E0_VALUE
    if E0_VALUE is None:
        print(f'E0_VALUE is not set in the script for {station_label}. Please edit the file and set E0_VALUE = <numeric MeV>.')
        return
    E0 = float(E0_VALUE)

    # compute energies matching filenames order: 1x, 0.5x, 0.2x, 0.1x
    scale_map = {
        f'alpha 1x{station_name}.txt': 1.0,
        f'alpha 0.5x{station_name}.txt': 0.5,
        f'alpha 0.2x{station_name}.txt': 0.2,
        f'alpha 0.1x{station_name}.txt': 0.1,
    }

    energies = []
    for fn in valid_energies_labels:
        scale = scale_map.get(fn, None)
        if scale is None:
            # attempt to parse scale from station-tagged filename
            if f'1x{station_name}' in fn:
                scale = 1.0
            elif f'0.5x{station_name}' in fn:
                scale = 0.5
            elif f'0.2x{station_name}' in fn:
                scale = 0.2
            elif f'0.1x{station_name}' in fn:
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
    # For simplicity, use a small default value where uncertainty is zero
    sigma_y[sigma_y == 0] = 0.5

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

    # Save calibration parameters to file
    calib_path = os.path.join(base_dir, f'calibration_params_{station_name}.txt')
    with open(calib_path, 'w') as f:
        f.write(f'# Calibration parameters for {station_label}: channel = a * E + c, or E = (channel - c) / a\n')
        f.write(f'a = {a}\n')
        f.write(f'a_err = {a_err}\n')
        f.write(f'c = {c}\n')
        f.write(f'c_err = {c_err}\n')
    print(f'Wrote calibration parameters to: {calib_path}')


if __name__ == '__main__':
    main()
