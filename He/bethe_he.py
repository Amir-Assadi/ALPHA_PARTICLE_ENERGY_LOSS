# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.optimize import minimize_scalar
import math
import sys

# Set font for plots (consistent with E_spec_he.py)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================================
# HELIUM GAS PROPERTIES (for energy loss calculation)
# ============================================================================
Z_HE = 2  # atomic number
A_HE = 4.002602  # atomic mass (g/mol)

# Reference geometry (from E_spec_he.py)
X0_MM = 144.5  # reference thickness in mm
P0_MBAR = 1013.25  # reference pressure in mbar (1 atm)

# Temperature for gas density calculation (K)
TEMP_K = 300.15  # room temperature (~27°C)

# Reference density of Helium at standard conditions (g/cm³)
# This is used for all range calculations since thickness is defined at P0_MBAR
RHO_REF_G_CM3 = 0.0001785 # g/cm³ at P0_MBAR and TEMP_K

# ============================================================================
# PRESSURE UNCERTAINTY (mbar) - ONE VALUE PER PRESSURE POINT
# ============================================================================
# Edit this dictionary: {pressure_mbar: uncertainty_mbar}
PRESSURE_UNC_MBAR = {
    # Station 4 (HE4, TH)
    0.0035: 0.0005,
    100.5000: 1,
    140.0000: 1,
    215.0000: 1,
    276.0000: 1,
    344.0000: 1,
    418.0000: 1,
    506.0000: 1,
    568.0000: 1,
    617.0000: 1,
    660.0000: 1,
    700.0000: 1,
    752.0000: 1,
    800.0000: 1,
    849.0000: 1,
    903.0000: 1,
    960.0000: 1,

    # Station 3 (HE3, PU/AM/CM)
    # Keep this section empty for now; fill when HE3 pressure points are available.
}

# Sources to process (Station 3 + Station 4)
SOURCES_TO_PROCESS = ['pu', 'am', 'cm', 'th']

# Map each source to its corresponding peaks-by-thickness file
SOURCE_PEAKS_FILE = {
    'pu': 'peaks_by_thickness_he3.txt',
    'am': 'peaks_by_thickness_he3.txt',
    'cm': 'peaks_by_thickness_he3.txt',
    'th': 'peaks_by_thickness_he4.txt',
}


def bethe_bloch_nonrelativistic(E, I, Z, A, rho):
    """
    Classical Bethe formula (non-relativistic).
    
    For a gas, density depends on pressure: ρ = ρ_ref × (P / P_ref)
    
    -dE/dx = 4π z² [e²/(4πε₀)]² (NZ)/(mv²) ln(2mv²/I)
    
    Returns: dE/dx in MeV/cm (positive value = energy loss)
    
    Parameters:
    -----------
    E : float
        Kinetic energy in MeV
    I : float
        Mean excitation energy in eV
    Z : int
        Atomic number of absorber
    A : float
        Atomic mass of absorber (g/mol)
    rho : float
        Density in g/cm^3 (for gas, calculated from pressure)
    """
    if E <= 0:
        return 0.0
    
    # Physical constants
    N_A = 6.02214076e23  # Avogadro's number (mol^-1)
    m_e = 0.510998950  # electron mass (MeV/c²)
    m_alpha = 3727.379  # alpha particle mass (MeV/c²)
    z = 2  # charge of alpha particle
    
    # Constant: K = 4π N_A r_e² m_e c² = 0.307075 MeV cm²/mol
    K = 0.307075  # MeV cm²/mol
    
    # Number density N in cm^-3
    N = (rho * N_A) / A
    
    # Convert I from eV to MeV
    I_MeV = I / 1e6
    
    # Logarithm term: ln(4 m_e E / (m_alpha * I))
    ln_arg = (4.0 * m_e * E) / (m_alpha * I_MeV)
    ln_term = np.log(ln_arg)
    
    # Stopping power: K * z² * (Z/A) * rho * (m_alpha / (2E)) * ln_term
    dEdx = K * (z**2) * (Z / A) * rho * (m_alpha / (2.0 * E)) * ln_term
    
    return dEdx


def pressure_to_density_g_cm3(pressure_mbar):
    """
    Convert pressure to Helium gas density using ideal gas law.
    ρ = (P × M) / (R × T)
    
    Parameters:
    -----------
    pressure_mbar : float
        Pressure in mbar
    
    Returns:
    --------
    rho : float
        Density in g/cm³
    """
    # Constants
    R = 8.314  # J/(mol·K) = 8.314 Pa·m³/(mol·K)
    M_HE = A_HE / 1000.0  # molar mass in kg/mol
    
    # Convert pressure
    P_Pa = pressure_mbar * 100.0  # 1 mbar = 100 Pa
    
    # Ideal gas law: ρ = (P × M) / (R × T)
    rho = (P_Pa * M_HE) / (R * TEMP_K)  # kg/m³
    rho_g_cm3 = rho / 1000.0  # convert to g/cm³
    
    return rho_g_cm3


def range_simpson(E1, E2, I, Z, A, rho, n_points=500):
    """
    Calculate range (thickness) using Simpson's rule integration.
    
    Range = integral from E2 to E1 of [1 / (dE/dx)] dE
    
    Parameters:
    -----------
    E1 : float
        Initial energy (MeV)
    E2 : float
        Final energy (MeV)
    I : float
        Mean excitation energy (eV)
    Z, A, rho : material parameters
    n_points : int
        Number of integration points
    
    Returns:
    --------
    range_cm : float
        Range in cm
    """
    if E2 >= E1 or E2 < 0:
        return 0.0
    
    # Energy array from E2 to E1
    E_array = np.linspace(E2, E1, n_points)
    
    # Compute 1/(dE/dx) for each energy
    integrand = np.zeros(n_points)
    for i, E in enumerate(E_array):
        dEdx = bethe_bloch_nonrelativistic(E, I, Z, A, rho)
        if dEdx > 0:
            integrand[i] = 1.0 / dEdx
        else:
            integrand[i] = 0.0
    
    # Simpson's rule integration
    range_cm = simpson(integrand, x=E_array)
    
    return range_cm


def range_simpson_with_error(E1, E2, I, Z, A, rho, n_points=500):
    if n_points < 8:
        n_points = 8
    # Ensure even number of intervals for Simpson
    if n_points % 2 == 1:
        n_points += 1

    R1 = range_simpson(E1, E2, I, Z, A, rho, n_points=n_points)
    R2 = range_simpson(E1, E2, I, Z, A, rho, n_points=n_points * 2)
    # Simpson error estimate ~ |R2 - R1| / 15
    sigma_R = abs(R2 - R1) / 15.0
    return R2, sigma_R


def find_mean_excitation_energy_for_thickness(E1, E2, thickness_cm, Z, A, rho, I_min=0, I_max=700):
    """
    Find mean excitation energy I that gives the measured thickness.
    
    Parameters:
    -----------
    E1, E2 : float
        Initial and final energies (MeV)
    thickness_cm : float
        Measured thickness (cm)
    Z, A, rho : material parameters
    I_min, I_max : float
        Search range for I (eV)
    
    Returns:
    --------
    I_opt : float
        Optimal mean excitation energy (eV)
    """
    def objective(I):
        predicted_range = range_simpson(E1, E2, I, Z, A, rho)
        return abs(predicted_range - thickness_cm)
    
    result = minimize_scalar(objective, bounds=(I_min, I_max), method='bounded')
    return result.x


def propagate_uncertainty(E1, E2, thickness_cm, E1_unc, E2_unc, pressure_unc_mbar, Z, A, I_nominal):
    """
    Propagate uncertainties to mean excitation energy using finite differences.
    
    Returns:
    --------
    dict with uncertainty contributions from each source
    """
    uncertainties = {}
    
    # Convert pressure uncertainty to thickness uncertainty
    # thickness_mm = (P_mbar / P0_mbar) * X0_mm
    # ∂x/∂P = X0_mm / P0_mbar
    thickness_unc_cm = (X0_MM / P0_MBAR) * pressure_unc_mbar / 10.0  # convert mm to cm
    
    # Use reference density for all calculations (thickness is defined at P0_MBAR)
    # Nominal mean excitation energy
    I_nominal_calc = find_mean_excitation_energy_for_thickness(E1, E2, thickness_cm, Z, A, RHO_REF_G_CM3)
    
    # E1 uncertainty contribution
    I_E1_plus = find_mean_excitation_energy_for_thickness(E1 + E1_unc, E2, thickness_cm, Z, A, RHO_REF_G_CM3)
    I_E1_minus = find_mean_excitation_energy_for_thickness(E1 - E1_unc, E2, thickness_cm, Z, A, RHO_REF_G_CM3)
    uncertainties['E1'] = abs(I_E1_plus - I_E1_minus) / 2.0
    
    # E2 uncertainty contribution
    I_E2_plus = find_mean_excitation_energy_for_thickness(E1, E2 + E2_unc, thickness_cm, Z, A, RHO_REF_G_CM3)
    I_E2_minus = find_mean_excitation_energy_for_thickness(E1, E2 - E2_unc, thickness_cm, Z, A, RHO_REF_G_CM3)
    uncertainties['E2'] = abs(I_E2_plus - I_E2_minus) / 2.0
    
    # Thickness (pressure) uncertainty contribution
    I_thick_plus = find_mean_excitation_energy_for_thickness(E1, E2, thickness_cm + thickness_unc_cm, Z, A, RHO_REF_G_CM3)
    I_thick_minus = find_mean_excitation_energy_for_thickness(E1, E2, thickness_cm - thickness_unc_cm, Z, A, RHO_REF_G_CM3)
    uncertainties['thickness'] = abs(I_thick_plus - I_thick_minus) / 2.0
    
    # Simpson integration uncertainty contribution (numerical model)
    R_nominal, sigma_R = range_simpson_with_error(E1, E2, I_nominal, Z, A, RHO_REF_G_CM3)
    I_simpson_plus = find_mean_excitation_energy_for_thickness(E1, E2, thickness_cm + sigma_R, Z, A, RHO_REF_G_CM3)
    I_simpson_minus = find_mean_excitation_energy_for_thickness(E1, E2, thickness_cm - sigma_R, Z, A, RHO_REF_G_CM3)
    uncertainties['simpson'] = abs(I_simpson_plus - I_simpson_minus) / 2.0
    
    # Total uncertainty (add in quadrature)
    total_unc = np.sqrt(sum(v**2 for v in uncertainties.values()))
    
    # Apply minimum uncertainty floor (0.01 eV) to prevent numerical issues
    if total_unc < 0.01:
        total_unc = 0.01
    
    uncertainties['total'] = total_unc
    
    return uncertainties


def load_data_for_source(peaks_path, source):
    """
    Load data from peaks_by_thickness file for a specific source.
    
    Returns:
    --------
    dict with keys: pressure_mbar, thickness_um, energy_MeV, energy_total_unc
    """
    data_dict = {
        'pressure_mbar': [],
        'thickness_um': [],
        'energy_MeV': [],
        'energy_total_unc': [],
    }
    
    if not os.path.isfile(peaks_path):
        print(f'File not found: {peaks_path}')
        return None
    
    try:
        with open(peaks_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) < 14:
                    continue
                
                # Parse columns (0-indexed):
                # 0=pressure, 1=thickness_mm, 2=thickness_um, 3=source, 
                # 9=avg_energy_MeV, 13=energy_total_unc
                source_col = parts[3]
                if source_col.lower() != source.lower():
                    continue
                
                try:
                    pressure = float(parts[0])
                    thickness_um = float(parts[2])
                    energy = float(parts[9])
                    energy_unc = float(parts[13])
                    
                    data_dict['pressure_mbar'].append(pressure)
                    data_dict['thickness_um'].append(thickness_um)
                    data_dict['energy_MeV'].append(energy)
                    data_dict['energy_total_unc'].append(energy_unc)
                except (ValueError, IndexError):
                    continue
        
        if len(data_dict['pressure_mbar']) > 0:
            # Convert to numpy arrays and sort by thickness
            indices = np.argsort(data_dict['thickness_um'])
            for key in data_dict.keys():
                data_dict[key] = np.array(data_dict[key])[indices]
            return data_dict
        else:
            print(f'No data found for source: {source}')
            return None
    except Exception as e:
        print(f'Error loading data: {e}')
        return None


def analyze_source(source, data_dict, base_dir, output_prefix):
    """
    Analyze a single source: find mean excitation energy, plot distributions, etc.
    
    Returns:
    --------
    results_list : list of dicts with I values and uncertainties
    """
    if data_dict is None:
        return []
    
    # Reference energy (at minimum thickness)
    E1 = data_dict['energy_MeV'][0]
    E1_unc = data_dict['energy_total_unc'][0]
    
    print(f"\n{'='*70}")
    print(f"ANALYZING SOURCE (HE): {source.upper()}")
    print(f"{'='*70}")
    print(f"Reference energy E1 (at minimum thickness): {E1:.4f} ± {E1_unc:.4f} MeV\n")
    
    results = []
    
    # Process each thickness (skip the first one which is reference)
    for i in range(1, len(data_dict['pressure_mbar'])):
        pressure_mbar = data_dict['pressure_mbar'][i]
        thickness_um = data_dict['thickness_um'][i]
        thickness_cm = thickness_um / 1e4  # convert μm to cm
        
        E2 = data_dict['energy_MeV'][i]
        E2_unc = data_dict['energy_total_unc'][i]
        
        # Get pressure uncertainty for this pressure point
        pressure_unc = PRESSURE_UNC_MBAR.get(round(pressure_mbar, 4), 0.2)
        
        print(f"Thickness {thickness_um:.2f} um (P={pressure_mbar:.4f} +/- {pressure_unc:.4f} mbar):")
        print(f"  E2 = {E2:.4f} ± {E2_unc:.4f} MeV")
        
        # Find nominal mean excitation energy (use reference density)
        I_nominal = find_mean_excitation_energy_for_thickness(E1, E2, thickness_cm, Z_HE, A_HE, RHO_REF_G_CM3)
        
        print(f"  Nominal I = {I_nominal:.2f} eV")
        
        # Propagate uncertainties
        unc_dict = propagate_uncertainty(E1, E2, thickness_cm, E1_unc, E2_unc, pressure_unc, Z_HE, A_HE, I_nominal)
        
        print(f"  Uncertainty contributions:")
        print(f"    from E1:        {unc_dict['E1']:.2f} eV")
        print(f"    from E2:        {unc_dict['E2']:.2f} eV")
        print(f"    from pressure:  {unc_dict['thickness']:.2f} eV")
        print(f"    from simpson:   {unc_dict['simpson']:.2f} eV")
        print(f"    Total:          {unc_dict['total']:.2f} eV")
        print()
        
        results.append({
            'thickness_um': thickness_um,
            'pressure_mbar': pressure_mbar,
            'E2': E2,
            'I': I_nominal,
            'unc_E1': unc_dict['E1'],
            'unc_E2': unc_dict['E2'],
            'unc_thickness': unc_dict['thickness'],
            'unc_simpson': unc_dict['simpson'],
            'unc_total': unc_dict['total']
        })
    
    if len(results) == 0:
        print(f"No results for source {source}")
        return []
    
    # Extract arrays for analysis
    I_values = np.array([r['I'] for r in results])
    I_uncs = np.array([r['unc_total'] for r in results])
    thicknesses = np.array([r['thickness_um'] for r in results])
    
    # Add minimum uncertainty floor
    I_uncs = np.maximum(I_uncs, 0.1)
    
    # Statistical analysis
    I_mean = np.mean(I_values)
    I_std = np.std(I_values, ddof=1)
    I_sem = I_std / np.sqrt(len(I_values))
    
    # Weighted mean
    weights = 1.0 / (I_uncs ** 2)
    I_weighted_mean = np.sum(I_values * weights) / np.sum(weights)
    I_weighted_unc = np.sqrt(1.0 / np.sum(weights))
    
    # Extract individual uncertainty components
    unc_E1_array = np.array([r['unc_E1'] for r in results])
    unc_E2_array = np.array([r['unc_E2'] for r in results])
    unc_thickness_array = np.array([r['unc_thickness'] for r in results])
    unc_simpson_array = np.array([r['unc_simpson'] for r in results])
    
    # Average systematic uncertainties
    avg_unc_E1 = np.mean(unc_E1_array)
    avg_unc_E2 = np.mean(unc_E2_array)
    avg_unc_thickness = np.mean(unc_thickness_array)
    avg_unc_simpson = np.mean(unc_simpson_array)
    
    # Combine systematic uncertainties
    I_sys_unc = np.sqrt(
        avg_unc_E1**2 + avg_unc_E2**2 + avg_unc_thickness**2 + avg_unc_simpson**2
    )
    
    # Total uncertainty
    I_total_unc = np.sqrt(I_sys_unc**2 + I_std**2)
    
    # Print results
    print("=" * 70)
    print(f"RESULTS FOR SOURCE (HE): {source.upper()}")
    print("=" * 70)
    print(f"\nMean Excitation Energy: I = {I_mean:.2f} eV")
    print(f"\nSystematic uncertainties:")
    print(f"  σ(E1):         {avg_unc_E1:.2f} eV")
    print(f"  σ(E2):         {avg_unc_E2:.2f} eV")
    print(f"  σ(pressure):   {avg_unc_thickness:.2f} eV")
    print(f"  σ(simpson):    {avg_unc_simpson:.2f} eV")
    print(f"  Systematic:    {I_sys_unc:.2f} eV")
    print(f"\nStatistical uncertainty:")
    print(f"  σ(spread):     {I_std:.2f} eV ({len(I_values)} measurements)")
    print(f"\nTotal uncertainty: {I_total_unc:.2f} eV")
    print(f"\nRESULT: I = {I_mean:.2f} ± {I_total_unc:.2f} eV")
    print("=" * 70)
    
    # Create individual plots
    create_source_plots(source, thicknesses, I_values, I_uncs, I_mean, I_total_unc, 
                       results, avg_unc_E1, avg_unc_E2, avg_unc_thickness, avg_unc_simpson,
                       base_dir, output_prefix)
    
    return results


def create_source_plots(source, thicknesses, I_values, I_uncs, I_mean, I_total_unc,
                       results, avg_unc_E1, avg_unc_E2, avg_unc_thickness, avg_unc_simpson,
                       base_dir, output_prefix):
    """Create individual plots for a source."""
    
    # Plot 1: I vs thickness
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.errorbar(thicknesses, I_values, yerr=I_uncs, fmt='o', capsize=5, label='Individual fits')
    ax1.axhline(I_mean, color='r', linestyle='--', linewidth=2, label=f'$I$ = {I_mean:.1f} eV')
    ax1.axhline(I_mean + I_total_unc, color='r', linestyle=':', alpha=0.5)
    ax1.axhline(I_mean - I_total_unc, color='r', linestyle=':', alpha=0.5)
    ax1.fill_between([thicknesses.min(), thicknesses.max()], 
                     I_mean - I_total_unc, I_mean + I_total_unc,
                     color='r', alpha=0.2, label=f'±{I_total_unc:.1f} eV')
    ax1.set_xlabel('Thickness (um)')
    ax1.set_ylabel('Mean Excitation Energy I (eV)')
    ax1.set_title(f'HELIUM GAS (HE) {source.upper()} - Mean Excitation Energy vs Thickness')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Histogram
    ax2.hist(I_values, bins=max(3, len(I_values)//2), edgecolor='black', alpha=0.7)
    ax2.axvline(I_mean, color='r', linestyle='--', linewidth=2, label=f'I = {I_mean:.1f} eV')
    ax2.set_xlabel('Mean Excitation Energy I (eV)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'HELIUM GAS (HE) {source.upper()} - Distribution of $I$ Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_name = os.path.join(base_dir, f'{output_prefix}_{source}_excitation_dist.png')
    plt.savefig(fig_name, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {fig_name}")
    plt.close()
    
    # Uncertainty table
    col_labels = [
        'Thickness (μm)',
        'I (eV)',
        'σ(E₂) (eV)',
        'σ(P) (eV)',
        'σ(num) (eV)',
        'σ(total) (eV)',
    ]
    cell_text = []
    for r in results:
        cell_text.append([
            f"{r['thickness_um']:.2f}",
            f"{r['I']:.2f}",
            f"{r['unc_E2']:.2f}",
            f"{r['unc_thickness']:.2f}",
            f"{r['unc_simpson']:.2f}",
            f"{r['unc_total']:.2f}",
        ])
    
    cell_text.append([
        'FINAL',
        f"{I_mean:.2f}",
        f"{avg_unc_E2:.2f}",
        f"{avg_unc_thickness:.2f}",
        f"{avg_unc_simpson:.2f}",
        f"√({I_total_unc:.2f}²)",
    ])
    
    fig_tbl, ax_tbl = plt.subplots(figsize=(12, 0.4 * (len(cell_text) + 1)))
    ax_tbl.axis('off')
    table = ax_tbl.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.2)
    ax_tbl.set_title(f'HELIUM GAS (HE) {source.upper()} - Mean Excitation Energy and Uncertainties', pad=12)
    plt.tight_layout()
    fig_tbl_name = os.path.join(base_dir, f'{output_prefix}_{source}_uncertainty_table.png')
    plt.savefig(fig_tbl_name, dpi=150, bbox_inches='tight')
    print(f"Saved table: {fig_tbl_name}")
    plt.close()


def create_combined_plots(all_results_by_source, base_dir, output_prefix, overall_mean, overall_total_unc):
    """Create combined plots for all sources with overall mean and uncertainty."""
    
    # Collect all individual uncertainties for quadrature combination
    all_individual_uncs = []
    
    # Plot 1: Combined thickness vs excitation energy
    fig, ax = plt.subplots(figsize=(11, 6))
    
    colors = {'pu': 'blue', 'am': 'red', 'cm': 'green'}
    
    for source in all_results_by_source.keys():
        if len(all_results_by_source[source]) == 0:
            continue
        
        results = all_results_by_source[source]
        thicknesses = np.array([r['thickness_um'] for r in results])
        I_values = np.array([r['I'] for r in results])
        I_uncs = np.array([r['unc_total'] for r in results])
        
        # Collect uncertainties for quadrature
        all_individual_uncs.extend(I_uncs)
        
        color = colors.get(source, 'gray')
        ax.errorbar(thicknesses, I_values, yerr=I_uncs, fmt='o', capsize=5,
                   label=source.upper(), color=color, markersize=8, linewidth=2, alpha=0.7)
    
    # Show overall mean and uncertainty band
    y_min, y_max = ax.get_ylim()
    ax.axhline(overall_mean, color='black', linestyle='--', linewidth=2.5, 
               label=f'Combined Mean: $I$ = {overall_mean:.1f} eV')
    ax.fill_between([thicknesses.min() if 'thicknesses' in locals() else 0, 
                      thicknesses.max() if 'thicknesses' in locals() else 1000],
                     overall_mean - overall_total_unc, overall_mean + overall_total_unc,
                     color='black', alpha=0.15, label=f'Uncertainty: ±{overall_total_unc:.1f} eV')
    
    ax.set_xlabel('Thickness (μm)', fontsize=12)
    ax.set_ylabel('Mean Excitation Energy I (eV)', fontsize=12)
    ax.set_title('HELIUM GAS (HE) - Combined Mean Excitation Energy vs Thickness (All Sources)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_name = os.path.join(base_dir, f'{output_prefix}_combined_excitation_vs_thickness.png')
    plt.savefig(fig_name, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {fig_name}")
    plt.close()
    
    # Plot 2: Combined distribution of I values
    fig, ax = plt.subplots(figsize=(11, 6))
    
    hist_data = []
    hist_labels = []
    hist_colors = []
    all_I_combined = []

    for source in all_results_by_source.keys():
        if len(all_results_by_source[source]) == 0:
            continue

        results = all_results_by_source[source]
        I_values = np.array([r['I'] for r in results])
        all_I_combined.extend(I_values)
        hist_data.append(I_values)
        hist_labels.append(source.upper())
        hist_colors.append(colors.get(source, 'gray'))

    if len(all_I_combined) > 0:
        n_bins = max(5, len(all_I_combined) // 4)
        ax.hist(
            hist_data,
            bins=n_bins,
            stacked=True,
            color=hist_colors,
            label=hist_labels,
            edgecolor='black',
            alpha=0.75
        )
    
    # Show overall mean and uncertainty band
    ax.axvline(overall_mean, color='black', linestyle='--', linewidth=2.5, 
               label=f'Combined Mean: {overall_mean:.1f} eV')
    y_min, y_max = ax.get_ylim()
    ax.fill_betweenx([y_min, y_max], overall_mean - overall_total_unc, overall_mean + overall_total_unc,
                      color='black', alpha=0.15, label=f'Uncertainty: ±{overall_total_unc:.1f} eV')
    
    ax.set_xlabel('Mean Excitation Energy I (eV)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('HELIUM GAS (HE) - Combined Distribution of $I$ Values (All Sources)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig_name = os.path.join(base_dir, f'{output_prefix}_combined_excitation_dist.png')
    plt.savefig(fig_name, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {fig_name}")
    plt.close()




def main():
    """Main analysis routine for Helium gas energy loss study."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n" + "="*70)
    print("HELIUM GAS - MEAN EXCITATION ENERGY ANALYSIS")
    print("="*70)
    print(f"Material: Helium (Z={Z_HE}, A={A_HE} g/mol)")
    print(f"Temperature: {TEMP_K} K")
    print(f"Reference conditions: P0={P0_MBAR} mbar, X0={X0_MM} mm")
    print()
    
    # Process each source
    all_results_by_source = {}
    all_I_values = []
    all_I_uncertainties = []
    
    for source in SOURCES_TO_PROCESS:
        peaks_filename = SOURCE_PEAKS_FILE.get(source, 'peaks_by_thickness_he3.txt')
        peaks_path = os.path.join(base_dir, peaks_filename)
        data_dict = load_data_for_source(peaks_path, source)
        if data_dict is not None:
            results = analyze_source(source, data_dict, base_dir, 'he_combined')
            all_results_by_source[source] = results
            
            # Collect all I values
            for r in results:
                all_I_values.append(r['I'])
                all_I_uncertainties.append(r['unc_total'])
    
    # Create combined analysis
    if len(all_I_values) > 0:
        print("\n" + "="*70)
        print("COMBINED ANALYSIS (ALL SOURCES)")
        print("="*70)
        
        all_I_arr = np.array(all_I_values)
        all_unc_arr = np.array(all_I_uncertainties)
        all_unc_arr = np.maximum(all_unc_arr, 0.1)
        
        # Overall statistics
        overall_mean = np.mean(all_I_arr)
        overall_std = np.std(all_I_arr, ddof=1)
        
        # Weighted mean
        weights = 1.0 / (all_unc_arr ** 2)
        overall_weighted_mean = np.sum(all_I_arr * weights) / np.sum(weights)
        overall_weighted_unc = np.sqrt(1.0 / np.sum(weights))
        
        # Total uncertainty
        overall_total_unc = np.sqrt((overall_std**2) + (overall_weighted_unc**2))
        
        print(f"\nOverall Mean Excitation Energy: I = {overall_mean:.2f} eV")
        print(f"Statistical spread: σ = {overall_std:.2f} eV ({len(all_I_values)} measurements)")
        print(f"Weighted uncertainty: σ_w = {overall_weighted_unc:.2f} eV")
        print(f"Total uncertainty: σ_total = {overall_total_unc:.2f} eV")
        print(f"\nFINAL RESULT FOR HELIUM: I = {overall_mean:.2f} ± {overall_total_unc:.2f} eV")
        print("="*70)
        
        # Create combined plots
        create_combined_plots(all_results_by_source, base_dir, 'he_combined', overall_mean, overall_total_unc)
        
        # Save combined results
        save_combined_results(all_results_by_source, base_dir, overall_mean, overall_total_unc)


def save_combined_results(all_results_by_source, base_dir, overall_mean, overall_total_unc):
    """Save combined analysis results to file."""
    out_path = os.path.join(base_dir, 'ionization_results_he_combined.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('# Mean Excitation Energy Analysis Results for Helium\n')
        f.write(f'# Material: Helium Gas (Z={Z_HE}, A={A_HE} g/mol)\n')
        f.write(f'# Temperature: {TEMP_K} K\n')
        f.write('#\n')
        f.write(f'# COMBINED RESULTS (all sources):\n')
        f.write(f'# Overall I = {overall_mean:.2f} ± {overall_total_unc:.2f} eV\n')
        f.write('#\n')
        f.write('# Individual source results:\n')
        for source in all_results_by_source.keys():
            if len(all_results_by_source[source]) > 0:
                I_vals = np.array([r['I'] for r in all_results_by_source[source]])
                I_mean_src = np.mean(I_vals)
                I_std_src = np.std(I_vals, ddof=1)
                f.write(f'#   {source.upper()}: I = {I_mean_src:.2f} ± {I_std_src:.2f} eV ({len(I_vals)} points)\n')
    
    print(f"Results saved to: {out_path}")


if __name__ == '__main__':
    main()
