import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.optimize import minimize_scalar

# Nickel properties
Z_NI = 28  # atomic number
A_NI = 58.6934  # atomic mass (g/mol)

# User inputs - EDIT THESE VALUES
DENSITY_NI = 8.908  # g/cm^3 - material density
DENSITY_UNC = 0.001  # uncertainty on density (g/cm^3)
THICKNESS_SYS_UNC = 0.05  # systematic thickness uncertainty (μm)


def bethe_bloch_nonrelativistic(E, I, Z, A, rho):
    """
    Classical Bethe formula (non-relativistic).
    
    -dE/dx = 4π z² [e²/(4πε₀)]² (NZ)/(mv²) ln(2mv²/I)
    
    where:
    - z = charge of incident particle (2 for alpha)
    - N = number density = (rho * N_A) / A
    - m = electron mass
    - v² = 2E/m_alpha (non-relativistic)
    
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
        Density in g/cm^3
    """
    if E <= 0:
        return 0.0
    
    # Physical constants
    N_A = 6.02214076e23  # Avogadro's number (mol^-1)
    m_e = 0.510998950  # electron mass (MeV/c²)
    m_alpha = 3727.379  # alpha particle mass (MeV/c²)
    z = 2  # charge of alpha particle
    
    # Classical electron radius and related constant
    # K = 4π N_A r_e² m_e c² = 0.307075 MeV cm²/mol
    K = 0.307075  # MeV cm²/mol
    
    # Number density N in cm^-3
    N = (rho * N_A) / A
    
    # Non-relativistic velocity relation: E = (1/2) m_alpha v²
    # So: m_e v² = m_e * (2E/m_alpha) = 2 m_e E / m_alpha
    # For the logarithm term: 2 m_e v² = 4 m_e E / m_alpha
    
    # Convert I from eV to MeV
    I_MeV = I / 1e6
    
    # Logarithm term: ln(2 m_e v² / I) = ln(4 m_e E / (m_alpha * I))
    ln_arg = (4.0 * m_e * E) / (m_alpha * I_MeV)
    ln_term = np.log(ln_arg)
    
    # Stopping power using standard formulation:
    # -dE/dx = K * z² * (Z/A) * (rho/β²) * ln(2 m_e v² / I)
    # where β² = v²/c² = 2E/m_alpha (in natural units)
    # So: 1/β² = m_alpha / (2E)
    
    # In terms of v²: we have m_e v² = 2 m_e E / m_alpha
    # Prefactor: 4π [e²/(4πε₀)]² = K / N_A
    # Full formula: K * z² * (N/N_A) * Z * (m_alpha/(2E)) * ln_term
    
    dEdx = K * (z**2) * (Z / A) * rho * (m_alpha / (2.0 * E)) * ln_term
    
    return dEdx


def range_simpson(E1, E2, I, Z, A, rho, n_points=1000):
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


def range_simpson_with_error(E1, E2, I, Z, A, rho, n_points=1000):
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


def find_mean_excitation_energy_for_thickness(E1, E2, thickness_cm, Z, A, rho, I_min=40, I_max=400):
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


def propagate_uncertainty(E1, E2, thickness_cm, E1_unc, E2_unc, thickness_unc, rho, rho_unc, Z, A, I_nominal):
    """
    Propagate uncertainties to mean excitation energy using finite differences.
    
    Returns:
    --------
    dict with uncertainty contributions from each source
    """
    uncertainties = {}
    
    # Nominal mean excitation energy
    I_nominal_calc = find_mean_excitation_energy_for_thickness(E1, E2, thickness_cm, Z, A, rho)
    
    # E1 uncertainty contribution
    I_E1_plus = find_mean_excitation_energy_for_thickness(E1 + E1_unc, E2, thickness_cm, Z, A, rho)
    I_E1_minus = find_mean_excitation_energy_for_thickness(E1 - E1_unc, E2, thickness_cm, Z, A, rho)
    uncertainties['E1'] = abs(I_E1_plus - I_E1_minus) / 2.0
    
    # E2 uncertainty contribution
    I_E2_plus = find_mean_excitation_energy_for_thickness(E1, E2 + E2_unc, thickness_cm, Z, A, rho)
    I_E2_minus = find_mean_excitation_energy_for_thickness(E1, E2 - E2_unc, thickness_cm, Z, A, rho)
    uncertainties['E2'] = abs(I_E2_plus - I_E2_minus) / 2.0
    
    # Thickness uncertainty contribution
    I_thick_plus = find_mean_excitation_energy_for_thickness(E1, E2, thickness_cm + thickness_unc, Z, A, rho)
    I_thick_minus = find_mean_excitation_energy_for_thickness(E1, E2, thickness_cm - thickness_unc, Z, A, rho)
    uncertainties['thickness'] = abs(I_thick_plus - I_thick_minus) / 2.0
    
    # Density uncertainty contribution
    I_rho_plus = find_mean_excitation_energy_for_thickness(E1, E2, thickness_cm, Z, A, rho + rho_unc)
    I_rho_minus = find_mean_excitation_energy_for_thickness(E1, E2, thickness_cm, Z, A, rho - rho_unc)
    uncertainties['density'] = abs(I_rho_plus - I_rho_minus) / 2.0

    # Simpson integration uncertainty contribution (numerical model)
    R_nominal, sigma_R = range_simpson_with_error(E1, E2, I_nominal, Z, A, rho)
    I_simpson_plus = find_mean_excitation_energy_for_thickness(E1, E2, thickness_cm + sigma_R, Z, A, rho)
    I_simpson_minus = find_mean_excitation_energy_for_thickness(E1, E2, thickness_cm - sigma_R, Z, A, rho)
    uncertainties['simpson'] = abs(I_simpson_plus - I_simpson_minus) / 2.0
    
    # Total uncertainty (add in quadrature)
    total_unc = np.sqrt(sum(v**2 for v in uncertainties.values()))
    
    # Apply minimum uncertainty floor (0.01 eV) to prevent numerical issues
    if total_unc < 0.01:
        total_unc = 0.01
    
    uncertainties['total'] = total_unc
    
    return uncertainties


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    peaks_path = os.path.join(base_dir, 'peaks_by_thickness_ni.txt')
    
    if not os.path.isfile(peaks_path):
        print(f'File not found: {peaks_path}')
        return
    
    # Load data
    data = np.loadtxt(peaks_path, comments='#')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    thickness_um = data[:, 0]  # column 0: thickness in micrometers
    energy_mev = data[:, 6]    # column 6: energy in MeV
    energy_total_unc = data[:, 10]  # column 10: total energy uncertainty
    
    # Configuration
    print("\n=== Bethe-Bloch Analysis for Mean Excitation Energy ===\n")
    print(f"Material: Nickel (Z={Z_NI}, A={A_NI} g/mol)")
    print()
    
    rho = DENSITY_NI
    rho_unc = DENSITY_UNC
    thickness_sys_unc_um = THICKNESS_SYS_UNC
    
    print(f"Using density: {rho} ± {rho_unc} g/cm³")
    print(f"Thickness systematic uncertainty: {thickness_sys_unc_um} μm")
    print()
    
    # Reference energy (thickness = 0)
    E1 = energy_mev[0]
    E1_unc = energy_total_unc[0]
    
    print(f"Reference energy E1 (at thickness=0): {E1:.4f} ± {E1_unc:.4f} MeV\n")
    
    # Process each thickness
    results = []
    
    for i in range(1, len(thickness_um)):
        th_um = thickness_um[i]
        th_cm = th_um / 1e4  # convert micrometers to cm
        
        E2 = energy_mev[i]
        E2_unc = energy_total_unc[i]
        
        # Total thickness uncertainty (systematic + any from measurement)
        th_unc_cm = thickness_sys_unc_um / 1e4
        
        print(f"Processing thickness {th_um:.1f} μm:")
        print(f"  E2 = {E2:.4f} ± {E2_unc:.4f} MeV")
        
        # Find nominal mean excitation energy
        I_nominal = find_mean_excitation_energy_for_thickness(E1, E2, th_cm, Z_NI, A_NI, rho)
        
        print(f"  Nominal I = {I_nominal:.2f} eV")
        
        # Propagate uncertainties
        unc_dict = propagate_uncertainty(E1, E2, th_cm, E1_unc, E2_unc, th_unc_cm, rho, rho_unc, Z_NI, A_NI, I_nominal)
        
        print(f"  Uncertainty contributions:")
        print(f"    from E1:        {unc_dict['E1']:.2f} eV")
        print(f"    from E2:        {unc_dict['E2']:.2f} eV")
        print(f"    from thickness: {unc_dict['thickness']:.2f} eV")
        print(f"    from density:   {unc_dict['density']:.2f} eV")
        print(f"    from simpson:   {unc_dict['simpson']:.2f} eV")
        print(f"    Total:          {unc_dict['total']:.2f} eV")
        print()
        
        results.append({
            'thickness_um': th_um,
            'E2': E2,
            'I': I_nominal,
            'unc_E1': unc_dict['E1'],
            'unc_E2': unc_dict['E2'],
            'unc_thickness': unc_dict['thickness'],
            'unc_density': unc_dict['density'],
            'unc_simpson': unc_dict['simpson'],
            'unc_total': unc_dict['total']
        })
    
    # Extract arrays for analysis
    I_values = np.array([r['I'] for r in results])
    I_uncs = np.array([r['unc_total'] for r in results])
    thicknesses = np.array([r['thickness_um'] for r in results])
    
    # Add minimum uncertainty floor to prevent divide-by-zero (0.1 eV)
    I_uncs = np.maximum(I_uncs, 0.1)
    
    # Statistical analysis
    I_mean = np.mean(I_values)
    I_std = np.std(I_values, ddof=1)  # sample std
    I_sem = I_std / np.sqrt(len(I_values))  # standard error of mean
    
    # Weighted mean (by inverse variance)
    weights = 1.0 / (I_uncs ** 2)
    I_weighted_mean = np.sum(I_values * weights) / np.sum(weights)
    I_weighted_unc = np.sqrt(1.0 / np.sum(weights))
    
    # Average uncertainty from individual measurements
    I_avg_unc = np.mean(I_uncs)
    
    # Combined uncertainty: quadrature sum of systematic and statistical spread
    # Systematic component will be computed from the weighted source contributions below.
    
    # Extract individual uncertainty components (simple average)
    unc_E1_array = np.array([r['unc_E1'] for r in results])
    unc_E2_array = np.array([r['unc_E2'] for r in results])
    unc_thickness_array = np.array([r['unc_thickness'] for r in results])
    unc_density_array = np.array([r['unc_density'] for r in results])
    unc_simpson_array = np.array([r['unc_simpson'] for r in results])
    
    # Average systematic uncertainties
    avg_unc_E1 = np.mean(unc_E1_array)
    avg_unc_E2 = np.mean(unc_E2_array)
    avg_unc_thickness = np.mean(unc_thickness_array)
    avg_unc_density = np.mean(unc_density_array)
    avg_unc_simpson = np.mean(unc_simpson_array)
    
    # Combine systematic uncertainties in quadrature
    I_sys_unc = np.sqrt(
        avg_unc_E1**2
        + avg_unc_E2**2
        + avg_unc_thickness**2
        + avg_unc_density**2
        + avg_unc_simpson**2
    )
    
    # Total uncertainty (systematic + statistical spread)
    I_total_unc = np.sqrt(I_sys_unc**2 + I_std**2)
    
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\nMean Excitation Energy: I = {I_mean:.2f} eV")
    print(f"")
    print(f"Systematic uncertainties:")
    print(f"  σ(E1):        {avg_unc_E1:.2f} eV")
    print(f"  σ(E2):        {avg_unc_E2:.2f} eV")
    print(f"  σ(thickness): {avg_unc_thickness:.2f} eV")
    print(f"  σ(density):   {avg_unc_density:.2f} eV")
    print(f"  σ(simpson):   {avg_unc_simpson:.2f} eV")
    print(f"  Systematic (quadrature): {I_sys_unc:.2f} eV")
    print(f"")
    print(f"Statistical uncertainty:")
    print(f"  σ(spread):    {I_std:.2f} eV  ({len(I_values)} measurements)")
    print(f"")
    print(f"Total uncertainty (quadrature): {I_total_unc:.2f} eV")
    print(f"")
    print(f"RESULT: I = {I_mean:.2f} ± {I_total_unc:.2f} eV")
    print(f"   or:  I = {I_mean:.2f} ± {avg_unc_E2:.2f}(E₂) ± {avg_unc_E1:.2f}(E₁) ± {avg_unc_density:.2f}(ρ) ± {avg_unc_thickness:.2f}(x) ± {I_std:.2f}(stat) eV")
    print("=" * 70)
    print()
    
    # Plot distribution of I values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: I vs thickness
    ax1.errorbar(thicknesses, I_values, yerr=I_uncs, fmt='o', capsize=5, label='Individual fits')
    ax1.axhline(I_mean, color='r', linestyle='--', linewidth=2, label=f'$I$ = {I_mean:.1f} eV')
    ax1.axhline(I_mean + I_total_unc, color='r', linestyle=':', alpha=0.5)
    ax1.axhline(I_mean - I_total_unc, color='r', linestyle=':', alpha=0.5)
    ax1.fill_between([thicknesses.min(), thicknesses.max()], 
                     I_mean - I_total_unc, I_mean + I_total_unc,
                     color='r', alpha=0.2, label=f'±{I_total_unc:.1f} eV')
    ax1.set_xlabel('Thickness (μm)')
    ax1.set_ylabel('Mean Excitation Energy $I$ (eV)')
    ax1.set_title('Mean Excitation Energy vs Thickness')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Histogram of I values
    ax2.hist(I_values, bins=max(5, len(I_values)//2), edgecolor='black', alpha=0.7)
    ax2.axvline(I_mean, color='r', linestyle='--', linewidth=2, label=f'$I$ = {I_mean:.1f} eV')
    ax2.set_xlabel('Mean Excitation Energy $I$ (eV)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Mean Excitation Energy Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

    # Plot table of mean excitation energy values and uncertainty contributions
    col_labels = [
        'Thickness (μm)',
        'I (eV)',
        'σ(E₂) (eV)',
        'σ(E₁) (eV)',
        'σ(ρ) (eV)',
        'σ(x) (eV)',
        'σ(num) (eV)',
        'σ(total) (eV)',
    ]
    cell_text = []
    for r in results:
        cell_text.append([
            f"{r['thickness_um']:.1f}",
            f"{r['I']:.2f}",
            f"{r['unc_E2']:.2f}",
            f"{r['unc_E1']:.2f}",
            f"{r['unc_density']:.2f}",
            f"{r['unc_thickness']:.2f}",
            f"{r['unc_simpson']:.2f}",
            f"{r['unc_total']:.2f}",
        ])

    cell_text.append([
        'FINAL',
        f"{I_mean:.2f}",
        f"{avg_unc_E2:.2f}",
        f"{avg_unc_E1:.2f}",
        f"{avg_unc_density:.2f}",
        f"{avg_unc_thickness:.2f}",
        f"{avg_unc_simpson:.2f}",
        f"{I_total_unc:.2f}",
    ])

    fig_tbl, ax_tbl = plt.subplots(figsize=(12, 0.4 * (len(cell_text) + 2)))
    ax_tbl.axis('off')
    table = ax_tbl.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.2)
    ax_tbl.set_title('Mean Excitation Energy and Uncertainty Table', pad=12)
    plt.tight_layout()
    plt.show()
    
    # Save results
    out_path = os.path.join(base_dir, 'ionization_results_ni.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('# Mean Excitation Energy Analysis Results\n')
        f.write(f'# Material: Nickel (Z={Z_NI}, A={A_NI} g/mol)\n')
        f.write(f'# Density: {rho} ± {rho_unc} g/cm³\n')
        f.write(f'# Reference energy E1: {E1:.4f} ± {E1_unc:.4f} MeV\n')
        f.write('#\n')
        f.write(f'# FINAL RESULT:\n')
        f.write(f'# I = {I_mean:.2f} ± {I_total_unc:.2f} eV\n')
        f.write(f'#\n')
        f.write(f'# Uncertainty breakdown:\n')
        f.write(f'#   σ(E1)        = {avg_unc_E1:.2f} eV\n')
        f.write(f'#   σ(E2)        = {avg_unc_E2:.2f} eV\n')
        f.write(f'#   σ(thickness) = {avg_unc_thickness:.2f} eV\n')
        f.write(f'#   σ(density)   = {avg_unc_density:.2f} eV\n')
        f.write(f'#   σ(simpson)   = {avg_unc_simpson:.2f} eV\n')
        f.write(f'#   σ(statistical) = {I_std:.2f} eV\n')
        f.write(f'#   σ(total)     = {I_total_unc:.2f} eV\n')
        f.write('#\n')
        f.write('# thickness_um  E2_MeV  I_eV  unc_E1  unc_E2  unc_thickness  unc_density  unc_simpson  unc_total\n')
        for r in results:
            f.write(f"{r['thickness_um']:.3f} {r['E2']:.4f} {r['I']:.2f} {r['unc_E1']:.2f} {r['unc_E2']:.2f} {r['unc_thickness']:.2f} {r['unc_density']:.2f} {r['unc_simpson']:.2f} {r['unc_total']:.2f}\n")
    
    print(f"Results saved to: {out_path}")


if __name__ == '__main__':
    main()
