from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


AVOGADRO = 6.02214076e23
ALPHA_MASS_MEV_C2 = 3727.379378
ALPHA_CHARGE_NUMBER = 2
BETHE_CONSTANT = 0.307075  # MeV cm^2 / g
ELECTRON_MASS_MEV_C2 = 0.51099895
BOHR_RADIUS_CM = 5.29177210903e-9
HBAR_C_MEV_FM = 197.3269804
COULOMB_COUPLING_MEV_FM = 1.43996448  # e^2/(4*pi*epsilon0) in MeV*fm
FM2_TO_CM2 = 1e-26

# Monte Carlo configuration.
RANDOM_SEED = 42
NUMBER_OF_PARTICLES = 50
# Set per-material initial alpha energies here (MeV).
INITIAL_ENERGY_BY_SYMBOL_MEV = {
    "Al": 5.806,
    "Ni": 5.45,
    "N": 4.69,
    "Ar": 4.69,
    "He": 4.69,
}
MIN_ENERGY_MEV = 0.05
SOLID_THICKNESS_UM = 27
GAS_THICKNESS_CM = 20
DEPTH_BINS = 100
STEPS_PER_TRACK = 200
MAX_DEFLECTION_RAD = np.deg2rad(30.0)
LARGE_ANGLE_THRESHOLD_RAD = np.deg2rad(3.0)
MAX_SINGLE_SCATTER_ANGLE_RAD = np.deg2rad(30.0)
CHARGE_EXCHANGE_ENABLED = True
CHARGE_EXCHANGE_MAX_ENERGY_MEV = 1.0
CHARGE_EXCHANGE_REFERENCE_MEAN_FREE_PATH_CM = 5e-5
REFERENCE_NUMBER_DENSITY_CM3 = 1e22
TRACK_PLOT_PARTICLES = 20
BRAGG_CURVE_POINTS = 4000
BRAGG_SMOOTHING_WINDOW = 20
MAIN_BRAGG_X_LIMIT_UM = 30.0
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
SUMMARY_PLOT_PATH = OUTPUT_DIR / "alpha_materials_summary.png"
PARTICLE_PLOT_PATH = OUTPUT_DIR / "alpha_particle_tracks.png"
BRAGG_PLOT_PATH = OUTPUT_DIR / "alpha_bragg_profiles.png"
BRAGG_SOLID_PLOT_PATH = OUTPUT_DIR / "alpha_bragg_solids.png"
BRAGG_GAS_PLOT_PATH = OUTPUT_DIR / "alpha_bragg_gases.png"
LATERAL_DIST_PLOT_PATH = OUTPUT_DIR / "alpha_lateral_distributions.png"

MATERIAL_COLORS = ["#1A1A1A", "#8B0000", "#4D4D4D", "#CC2200", "#2B2B2B"]
MODE_COLORS = {"without_effective": "#365c77", "with_effective": "#dd2c2c"}


@dataclass(frozen=True)
class Material:
    name: str
    symbol: str
    z: int
    a_g_mol: float
    density_g_cm3: float
    mean_excitation_energy_ev: float
    is_gas: bool


@dataclass
class SimulationSummary:
    mean_final_energy_mev: float
    std_final_energy_mev: float
    mean_lateral_spread_um: float
    rms_exit_angle_deg: float
    transmitted_fraction: float
    thickness_cm: float
    mean_stopping_profile_mev_per_cm: np.ndarray
    sample_track_depths_cm: np.ndarray
    sample_track_x_cm: np.ndarray
    exit_x_um_all: np.ndarray


MATERIALS: List[Material] = [
    Material("Aluminum", "Al", 13, 26.9815385, 2.70, 166.0, False),
    Material("Nickel", "Ni", 28, 58.6934, 8.908, 311.0, False),
    Material("Nitrogen (gas)", "N", 7, 14.0067, 0.001165, 82.0, True),
    Material("Argon (gas)", "Ar", 18, 39.948, 0.001784, 188.0, True),
    Material("Helium (gas)", "He", 2, 4.002602, 0.000166, 41.8, True),
]


def material_thickness_cm(material: Material) -> float:
    return GAS_THICKNESS_CM if material.is_gas else SOLID_THICKNESS_UM * 1e-4


def material_display_length_scale(material: Material) -> float:
    return 1.0 if material.is_gas else 1e4


def material_display_length_unit(material: Material) -> str:
    return "cm" if material.is_gas else "um"


def initial_energy_mev_for_material(material: Material) -> float:
    if material.symbol not in INITIAL_ENERGY_BY_SYMBOL_MEV:
        raise KeyError(f"Missing initial energy for material symbol: {material.symbol}")
    return float(INITIAL_ENERGY_BY_SYMBOL_MEV[material.symbol])


def smooth_profile(profile: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return profile.copy()

    valid_window = min(window, profile.size)
    if valid_window % 2 == 0:
        valid_window = max(1, valid_window - 1)
    if valid_window <= 1:
        return profile.copy()

    kernel = np.ones(valid_window, dtype=float) / valid_window
    padded = np.pad(profile, (valid_window // 2, valid_window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def resample_profile(depth_values: np.ndarray, profile: np.ndarray, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    target_points = max(2, num_points)
    if depth_values.size == target_points:
        return depth_values, profile

    smooth_depth = np.linspace(depth_values[0], depth_values[-1], target_points)
    smooth_profile_values = np.interp(smooth_depth, depth_values, profile)
    return smooth_depth, smooth_profile_values


def number_density_cm3(material: Material) -> float:
    return material.density_g_cm3 * AVOGADRO / material.a_g_mol


def alpha_beta_gamma(energy_mev: float) -> Tuple[float, float]:
    gamma = 1.0 + energy_mev / ALPHA_MASS_MEV_C2
    beta_sq = max(1.0 - 1.0 / (gamma * gamma), 1e-10)
    return np.sqrt(beta_sq), gamma


def equilibrium_effective_charge_number(energy_mev: float) -> float:
    beta, _ = alpha_beta_gamma(energy_mev)
    reduced_charge = ALPHA_CHARGE_NUMBER * (1.0 - np.exp(-125.0 * beta / (ALPHA_CHARGE_NUMBER ** (2.0 / 3.0))))
    return float(np.clip(reduced_charge, 0.0, ALPHA_CHARGE_NUMBER))


def sample_charge_state_from_mean(rng: np.random.Generator, mean_charge: float) -> int:
    mean_charge_clamped = float(np.clip(mean_charge, 1.0, ALPHA_CHARGE_NUMBER))
    probability_double = mean_charge_clamped - 1.0
    return 2 if rng.random() < probability_double else 1


def maybe_update_charge_state(
    rng: np.random.Generator,
    current_charge_state: int,
    energy_mev: float,
    material: Material,
    step_cm: float,
) -> int:
    if not CHARGE_EXCHANGE_ENABLED or energy_mev >= CHARGE_EXCHANGE_MAX_ENERGY_MEV or current_charge_state <= 1:
        return current_charge_state
    target_mean_charge = equilibrium_effective_charge_number(energy_mev)
    density_scale = REFERENCE_NUMBER_DENSITY_CM3 / max(number_density_cm3(material), 1e-30)
    mean_free_path_cm = CHARGE_EXCHANGE_REFERENCE_MEAN_FREE_PATH_CM * density_scale
    energy_factor = np.clip((CHARGE_EXCHANGE_MAX_ENERGY_MEV - energy_mev) / CHARGE_EXCHANGE_MAX_ENERGY_MEV, 0.0, 1.0)
    transition_probability = 1.0 - np.exp(-energy_factor * step_cm / max(mean_free_path_cm, 1e-20))
    if rng.random() >= transition_probability:
        return current_charge_state

    sampled_state = sample_charge_state_from_mean(rng, target_mean_charge)
    return max(1, min(current_charge_state, sampled_state))


def bethe_stopping_power_mev_per_cm(energy_mev: float, material: Material, projectile_charge_number: float) -> float:
    if energy_mev <= 0.0:
        return 0.0

    beta, gamma = alpha_beta_gamma(energy_mev)
    beta_sq = max(beta * beta, 1e-10)
    i_mev = material.mean_excitation_energy_ev * 1e-6
    t_max = (
        2.0
        * ELECTRON_MASS_MEV_C2
        * beta_sq
        * gamma
        * gamma
        / (1.0 + (2.0 * gamma * ELECTRON_MASS_MEV_C2 / ALPHA_MASS_MEV_C2) + (ELECTRON_MASS_MEV_C2 / ALPHA_MASS_MEV_C2) ** 2)
    )

    log_arg = max((2.0 * ELECTRON_MASS_MEV_C2 * beta_sq * gamma * gamma * t_max) / (i_mev * i_mev), 1.000001)
    bracket = max(0.5 * np.log(log_arg) - beta_sq, 1e-5)

    mass_stopping_power = (
        BETHE_CONSTANT
        * (material.z / material.a_g_mol)
        * (projectile_charge_number ** 2)
        * (1.0 / beta_sq)
        * bracket
    )
    return mass_stopping_power * material.density_g_cm3


def radiation_length_cm(material: Material) -> float:
    x0_g_cm2 = 716.4 * material.a_g_mol / (material.z * (material.z + 1.0) * np.log(287.0 / np.sqrt(material.z)))
    return x0_g_cm2 / material.density_g_cm3


def screening_angle_rad(energy_mev: float, material: Material) -> float:
    beta, gamma = alpha_beta_gamma(energy_mev)
    momentum_mev_c = beta * gamma * ALPHA_MASS_MEV_C2
    if momentum_mev_c <= 0.0:
        return 0.0

    a_tf_cm = 0.8853 * BOHR_RADIUS_CM / (material.z ** (1.0 / 3.0))
    a_tf_fm = a_tf_cm * 1e13
    return HBAR_C_MEV_FM / (2.0 * momentum_mev_c * a_tf_fm)


def highland_theta0_rad(energy_mev: float, material: Material, step_cm: float, projectile_charge_number: float) -> float:
    beta, gamma = alpha_beta_gamma(energy_mev)
    momentum_mev_c = beta * gamma * ALPHA_MASS_MEV_C2
    if momentum_mev_c <= 0.0 or step_cm <= 0.0:
        return 0.0

    x0_cm = radiation_length_cm(material)
    ratio = max(step_cm / x0_cm, 1e-12)
    theta0 = (13.6 / (beta * momentum_mev_c)) * projectile_charge_number * np.sqrt(ratio) * (1.0 + 0.038 * np.log(ratio))
    return float(max(theta0, 0.0))


def rutherford_prefactor_fm2(energy_mev: float, material: Material, projectile_charge_number: float) -> float:
    if energy_mev <= 0.0:
        return 0.0
    prefactor = (projectile_charge_number * material.z * COULOMB_COUPLING_MEV_FM / (4.0 * energy_mev)) ** 2
    return float(max(prefactor, 0.0))


def screened_rutherford_sigma_gt_theta_cm2(
    energy_mev: float,
    material: Material,
    projectile_charge_number: float,
    theta_min_rad: float,
    theta_max_rad: float,
    screened: bool,
) -> float:
    if energy_mev <= 0.0:
        return 0.0

    theta_min = max(theta_min_rad, 1e-8)
    theta_max = max(theta_max_rad, theta_min)
    theta_s = screening_angle_rad(energy_mev, material) if screened else 0.0
    prefactor_fm2 = rutherford_prefactor_fm2(energy_mev, material, projectile_charge_number)

    inv_min = 1.0 / (theta_min * theta_min + theta_s * theta_s)
    inv_max = 1.0 / (theta_max * theta_max + theta_s * theta_s)
    sigma_fm2 = 16.0 * np.pi * prefactor_fm2 * max(inv_min - inv_max, 0.0)
    return float(max(sigma_fm2 * FM2_TO_CM2, 0.0))


def sample_screened_rutherford_angle_rad(
    rng: np.random.Generator,
    energy_mev: float,
    material: Material,
    theta_min_rad: float,
    theta_max_rad: float,
    screened: bool,
) -> float:
    theta_min = max(theta_min_rad, 1e-8)
    theta_max = max(theta_max_rad, theta_min)
    theta_s = screening_angle_rad(energy_mev, material) if screened else 0.0

    a = 1.0 / (theta_min * theta_min + theta_s * theta_s)
    b = 1.0 / (theta_max * theta_max + theta_s * theta_s)
    u = rng.random()
    inv_val = a - u * (a - b)
    theta_sq = max(1.0 / max(inv_val, 1e-20) - theta_s * theta_s, 0.0)
    return float(np.sqrt(theta_sq))


def sample_step_deflection_rad(
    rng: np.random.Generator,
    energy_mev: float,
    material: Material,
    step_cm: float,
    screened: bool,
    projectile_charge_number: float,
) -> float:
    if energy_mev <= 0.0 or step_cm <= 0.0 or projectile_charge_number <= 0.0:
        return 0.0

    theta0 = highland_theta0_rad(energy_mev, material, step_cm, projectile_charge_number)
    small_angle_kick = rng.normal(0.0, max(theta0, 1e-10))

    sigma_large_cm2 = screened_rutherford_sigma_gt_theta_cm2(
        energy_mev=energy_mev,
        material=material,
        projectile_charge_number=projectile_charge_number,
        theta_min_rad=LARGE_ANGLE_THRESHOLD_RAD,
        theta_max_rad=MAX_SINGLE_SCATTER_ANGLE_RAD,
        screened=screened,
    )
    n_density = number_density_cm3(material)
    mean_large_events = max(n_density * sigma_large_cm2 * step_cm, 0.0)
    n_large_events = int(rng.poisson(mean_large_events)) if mean_large_events > 0.0 else 0

    large_angle_kick = 0.0
    for _ in range(n_large_events):
        theta_abs = sample_screened_rutherford_angle_rad(
            rng=rng,
            energy_mev=energy_mev,
            material=material,
            theta_min_rad=LARGE_ANGLE_THRESHOLD_RAD,
            theta_max_rad=MAX_SINGLE_SCATTER_ANGLE_RAD,
            screened=screened,
        )
        large_angle_kick += theta_abs if rng.random() < 0.5 else -theta_abs

    return float(np.clip(small_angle_kick + large_angle_kick, -MAX_DEFLECTION_RAD, MAX_DEFLECTION_RAD))


def simulate_one_condition(
    material: Material,
    screened: bool,
    use_effective_charge: bool,
    rng: np.random.Generator,
) -> SimulationSummary:
    thickness_cm = material_thickness_cm(material)
    step_cm = thickness_cm / STEPS_PER_TRACK
    depth_edges_cm = np.linspace(0.0, thickness_cm, DEPTH_BINS + 1)
    bin_width_cm = depth_edges_cm[1] - depth_edges_cm[0]

    stopping_profiles = np.zeros((NUMBER_OF_PARTICLES, DEPTH_BINS), dtype=float)
    final_energy = np.zeros(NUMBER_OF_PARTICLES, dtype=float)
    exit_x_um = np.zeros(NUMBER_OF_PARTICLES, dtype=float)
    exit_angle_deg = np.zeros(NUMBER_OF_PARTICLES, dtype=float)
    transmitted = np.zeros(NUMBER_OF_PARTICLES, dtype=bool)
    track_depths_cm = np.full((TRACK_PLOT_PARTICLES, STEPS_PER_TRACK + 1), np.nan, dtype=float)
    track_x_cm = np.full((TRACK_PLOT_PARTICLES, STEPS_PER_TRACK + 1), np.nan, dtype=float)

    for particle_index in range(NUMBER_OF_PARTICLES):
        energy = initial_energy_mev_for_material(material)
        charge_state = ALPHA_CHARGE_NUMBER
        x_cm = 0.0
        z_cm = 0.0
        angle_rad = 0.0
        if particle_index < TRACK_PLOT_PARTICLES:
            track_depths_cm[particle_index, 0] = z_cm
            track_x_cm[particle_index, 0] = x_cm
        track_step_index = 1

        for _ in range(STEPS_PER_TRACK):
            if energy <= MIN_ENERGY_MEV:
                break
            if z_cm >= thickness_cm:
                transmitted[particle_index] = True
                break

            if use_effective_charge:
                charge_state = maybe_update_charge_state(rng, charge_state, energy, material, step_cm)
            else:
                charge_state = ALPHA_CHARGE_NUMBER

            d_edx = bethe_stopping_power_mev_per_cm(energy, material, float(charge_state))
            ds_cm = step_cm / max(np.cos(angle_rad), 1e-6)
            delta_e = min(energy, d_edx * ds_cm)

            energy_after = max(energy - delta_e, 0.0)
            z_next_cm = z_cm + step_cm
            x_next_cm = x_cm + step_cm * np.tan(angle_rad)

            segment_z_min = max(0.0, min(z_cm, z_next_cm))
            segment_z_max = min(thickness_cm, max(z_cm, z_next_cm))
            if segment_z_max > segment_z_min:
                first_bin = max(0, np.searchsorted(depth_edges_cm, segment_z_min, side="right") - 1)
                last_bin = min(DEPTH_BINS - 1, np.searchsorted(depth_edges_cm, segment_z_max, side="left"))
                traversed = segment_z_max - segment_z_min

                for bin_index in range(first_bin, last_bin + 1):
                    overlap_start = max(segment_z_min, depth_edges_cm[bin_index])
                    overlap_end = min(segment_z_max, depth_edges_cm[bin_index + 1])
                    overlap = overlap_end - overlap_start
                    if overlap <= 0.0:
                        continue
                    stopping_profiles[particle_index, bin_index] += (delta_e * overlap / traversed) / bin_width_cm

            z_cm = z_next_cm
            x_cm = x_next_cm
            energy = energy_after
            angle_rad += sample_step_deflection_rad(rng, energy, material, step_cm, screened, float(charge_state))
            if particle_index < TRACK_PLOT_PARTICLES and track_step_index <= STEPS_PER_TRACK:
                track_depths_cm[particle_index, track_step_index] = z_cm
                track_x_cm[particle_index, track_step_index] = x_cm
                track_step_index += 1

        final_energy[particle_index] = energy
        exit_x_um[particle_index] = x_cm * 1e4
        exit_angle_deg[particle_index] = np.rad2deg(angle_rad)
        if z_cm >= thickness_cm:
            transmitted[particle_index] = True

    mean_profile_mev_per_cm = stopping_profiles.mean(axis=0)

    return SimulationSummary(
        mean_final_energy_mev=float(np.mean(final_energy)),
        std_final_energy_mev=float(np.std(final_energy)),
        mean_lateral_spread_um=float(np.std(exit_x_um)),
        rms_exit_angle_deg=float(np.sqrt(np.mean(exit_angle_deg ** 2))),
        transmitted_fraction=float(np.mean(transmitted)),
        thickness_cm=thickness_cm,
        mean_stopping_profile_mev_per_cm=mean_profile_mev_per_cm,
        sample_track_depths_cm=track_depths_cm,
        sample_track_x_cm=track_x_cm,
        exit_x_um_all=exit_x_um.copy(),
    )


def run_all_simulations() -> Dict[str, Dict[str, SimulationSummary]]:
    rng_no_effective = np.random.default_rng(RANDOM_SEED)
    rng_with_effective = np.random.default_rng(RANDOM_SEED + 1)
    results: Dict[str, Dict[str, SimulationSummary]] = {}

    for material in MATERIALS:
        results[material.symbol] = {
            "without_effective": simulate_one_condition(
                material=material,
                screened=False,
                use_effective_charge=False,
                rng=rng_no_effective,
            ),
            "with_effective": simulate_one_condition(
                material=material,
                screened=False,
                use_effective_charge=True,
                rng=rng_with_effective,
            ),
        }
    return results


def print_table(results: Dict[str, Dict[str, SimulationSummary]]) -> None:
    header = (
        "Material | E_final no-eff (MeV) | E_final eff (MeV) | "
        "RMS angle no-eff (deg) | RMS angle eff (deg) | Transmitted no-eff | Transmitted eff"
    )
    print(header)
    print("-" * len(header))
    for material in MATERIALS:
        no_eff = results[material.symbol]["without_effective"]
        eff = results[material.symbol]["with_effective"]
        print(
            f"{material.symbol:>7} | {no_eff.mean_final_energy_mev:>20.4f} | {eff.mean_final_energy_mev:>17.4f} | "
            f"{no_eff.rms_exit_angle_deg:>22.3f} | {eff.rms_exit_angle_deg:>19.3f} | "
            f"{no_eff.transmitted_fraction:>17.3f} | {eff.transmitted_fraction:>15.3f}"
        )


def print_bragg_peak_positions(results: Dict[str, Dict[str, SimulationSummary]]) -> None:
    print("\nBragg peak thickness locations (no-effective vs with-effective):")
    for material in MATERIALS:
        scale = material_display_length_scale(material)
        unit = material_display_length_unit(material)

        summary_no_eff = results[material.symbol]["without_effective"]
        summary_eff = results[material.symbol]["with_effective"]

        depth_no_eff = np.linspace(
            0.5 * summary_no_eff.thickness_cm / DEPTH_BINS,
            summary_no_eff.thickness_cm - 0.5 * summary_no_eff.thickness_cm / DEPTH_BINS,
            DEPTH_BINS,
        ) * scale
        depth_eff = np.linspace(
            0.5 * summary_eff.thickness_cm / DEPTH_BINS,
            summary_eff.thickness_cm - 0.5 * summary_eff.thickness_cm / DEPTH_BINS,
            DEPTH_BINS,
        ) * scale

        profile_no_eff = smooth_profile(summary_no_eff.mean_stopping_profile_mev_per_cm / scale, BRAGG_SMOOTHING_WINDOW)
        profile_eff = smooth_profile(summary_eff.mean_stopping_profile_mev_per_cm / scale, BRAGG_SMOOTHING_WINDOW)

        peak_idx_no_eff = int(np.argmax(profile_no_eff))
        peak_idx_eff = int(np.argmax(profile_eff))
        print(
            f"  {material.symbol}: no-eff peak at {depth_no_eff[peak_idx_no_eff]:.3f} {unit}, "
            f"eff peak at {depth_eff[peak_idx_eff]:.3f} {unit}"
        )


def _manual_kde(vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    n = len(vals)
    if n < 2:
        return np.zeros_like(grid, dtype=float)
    bw = max(1.06 * float(np.std(vals)) * n ** (-0.2), 1e-30)
    diff = (grid[:, None] - vals[None, :]) / bw
    return np.mean(np.exp(-0.5 * diff * diff), axis=1) / (bw * np.sqrt(2.0 * np.pi))


def load_experimental_bragg_points_for_material(material: Material) -> Tuple[np.ndarray, np.ndarray]:
    """Build experimental Bragg points from thickness-energy tables for one material.

    Preferred input is peaks_all_trials*.txt (thickness_um, energy_MeV per trial).
    Falls back to peaks_by_thickness*.txt when needed.

    Returns:
    - x_mid_um: midpoint thickness in um
    - dedx_mev_per_um: stopping power in MeV/um
    """
    material_dir = Path(__file__).resolve().parent.parent / material.symbol

    if material.is_gas:
        # Station 4 files contain the single-source datasets (e.g. th),
        # avoiding station-3 mixed-source rows (pu/am/cm).
        candidate_files = sorted(material_dir.glob("peaks_all_trials*4*.txt"))
        if not candidate_files:
            candidate_files = sorted(material_dir.glob("peaks_by_thickness*4*.txt"))
    else:
        candidate_files = sorted(material_dir.glob("peaks_all_trials*.txt"))
        if not candidate_files:
            candidate_files = sorted(material_dir.glob("peaks_by_thickness*.txt"))
    if not candidate_files:
        raise FileNotFoundError(
            f"No peaks_all_trials or peaks_by_thickness files found for {material.symbol} in {material_dir}"
        )

    x_all = []
    e_all = []

    for file_path in candidate_files:
        # Determine thickness/energy columns from header tokens so we support
        # both gas and solid table variants.
        header_tokens = []
        with file_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line.startswith("#"):
                    header_tokens = line.lstrip("#").strip().split()
                    break
                if line:
                    break

        if "thickness_um" in header_tokens:
            thickness_col = header_tokens.index("thickness_um")
        elif "thickness_mm" in header_tokens:
            thickness_col = header_tokens.index("thickness_mm")
        else:
            # Fallbacks for known legacy layouts.
            thickness_col = 3 if ("peaks_all_trials" in file_path.name and material.is_gas) else 0

        if "energy_MeV" in header_tokens:
            energy_col = header_tokens.index("energy_MeV")
        elif "avg_energy_MeV" in header_tokens:
            energy_col = header_tokens.index("avg_energy_MeV")
        else:
            # Fallbacks for known legacy layouts.
            energy_col = 10 if ("peaks_all_trials" in file_path.name and material.is_gas) else 6

        usecols = (thickness_col, energy_col)

        try:
            data = np.genfromtxt(file_path, comments="#", usecols=usecols, dtype=float, invalid_raise=False)
        except Exception:
            continue

        data = np.atleast_2d(data)
        if data.size == 0 or data.shape[1] < 2:
            continue

        thickness_um = data[:, 0]
        if "thickness_mm" in header_tokens and "thickness_um" not in header_tokens:
            thickness_um = thickness_um * 1000.0
        energy_mev = data[:, 1]
        finite = np.isfinite(thickness_um) & np.isfinite(energy_mev)
        if np.any(finite):
            x_all.append(thickness_um[finite])
            e_all.append(energy_mev[finite])

    if not x_all:
        raise ValueError(f"No valid thickness/energy rows found for {material.symbol}")

    # Concatenate all measurements.
    x_all_concat = []
    e_all_concat = []
    for x_data, e_data in zip(x_all, e_all):
        finite = np.isfinite(x_data) & np.isfinite(e_data) & (x_data > 0.0)
        if np.any(finite):
            x_all_concat.append(x_data[finite])
            e_all_concat.append(e_data[finite])
    
    if not x_all_concat:
        raise ValueError(f"No valid thickness/energy rows found for {material.symbol}")

    x = np.concatenate(x_all_concat)
    e = np.concatenate(e_all_concat)
    order = np.argsort(x)
    x = x[order]
    e = e[order]

    # Merge duplicate x entries (for repeated sources/trials) by averaging energy.
    x_rounded = np.round(x, 6)
    unique_x, inverse = np.unique(x_rounded, return_inverse=True)
    x_group = np.zeros(unique_x.size, dtype=float)
    e_group = np.zeros(unique_x.size, dtype=float)
    for idx in range(unique_x.size):
        mask = inverse == idx
        x_group[idx] = float(np.mean(x[mask]))
        e_group[idx] = float(np.mean(e[mask]))

    # Make E(x) non-increasing to suppress noisy tails.
    e_group = np.minimum.accumulate(e_group)

    # Compute stopping power from finite differences.
    dx = np.diff(x_group)
    de = np.diff(e_group)
    valid = dx > 0.0
    if np.count_nonzero(valid) == 0:
        raise ValueError(f"Not enough distinct thickness points for {material.symbol}")

    x_mid_um = 0.5 * (x_group[1:] + x_group[:-1])[valid]
    dedx_mev_per_um = (-(de / dx))[valid]
    
    return x_mid_um, dedx_mev_per_um


def plot_results(results: Dict[str, Dict[str, SimulationSummary]]) -> Tuple[Path, Path, Path, Path, Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = [material.symbol for material in MATERIALS]
    x = np.arange(len(labels))
    bar_width = 0.38

    e_vals_no_eff = [results[s]["without_effective"].mean_final_energy_mev for s in labels]
    e_vals_eff = [results[s]["with_effective"].mean_final_energy_mev for s in labels]
    spread_vals_no_eff = [results[s]["without_effective"].mean_lateral_spread_um for s in labels]
    spread_vals_eff = [results[s]["with_effective"].mean_lateral_spread_um for s in labels]
    ang_vals_no_eff = [results[s]["without_effective"].rms_exit_angle_deg for s in labels]
    ang_vals_eff = [results[s]["with_effective"].rms_exit_angle_deg for s in labels]
    transmitted_vals_no_eff = [results[s]["without_effective"].transmitted_fraction for s in labels]
    transmitted_vals_eff = [results[s]["with_effective"].transmitted_fraction for s in labels]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax0, ax1, ax2, ax3 = axes.ravel()

    ax0.bar(x - bar_width / 2, e_vals_no_eff, width=bar_width, color=MODE_COLORS["without_effective"], alpha=0.60)
    ax0.bar(x + bar_width / 2, e_vals_eff, width=bar_width, color=MODE_COLORS["with_effective"], alpha=0.50)
    ax0.set_xticks(x, labels)
    ax0.set_ylabel("Mean final energy (MeV)")
    ax0.set_title("Energy retained after transport")
    ax0.grid(alpha=0.20, axis="y", color="#AAAAAA")
    ax0.legend(["No effective charge", "With effective charge"], loc="best")

    ax1.bar(x - bar_width / 2, spread_vals_no_eff, width=bar_width, color=MODE_COLORS["without_effective"], alpha=0.60)
    ax1.bar(x + bar_width / 2, spread_vals_eff, width=bar_width, color=MODE_COLORS["with_effective"], alpha=0.50)
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("RMS lateral spread (um)")
    ax1.set_title("Beam lateral broadening (RMS)")
    ax1.grid(alpha=0.20, axis="y", color="#AAAAAA")

    ax2.bar(x - bar_width / 2, ang_vals_no_eff, width=bar_width, color=MODE_COLORS["without_effective"], alpha=0.60)
    ax2.bar(x + bar_width / 2, ang_vals_eff, width=bar_width, color=MODE_COLORS["with_effective"], alpha=0.50)
    ax2.set_xticks(x, labels)
    ax2.set_ylabel("RMS exit angle (deg)")
    ax2.set_title("Angular straggling")
    ax2.grid(alpha=0.20, axis="y", color="#AAAAAA")

    ax3.bar(x - bar_width / 2, transmitted_vals_no_eff, width=bar_width, color=MODE_COLORS["without_effective"], alpha=0.60)
    ax3.bar(x + bar_width / 2, transmitted_vals_eff, width=bar_width, color=MODE_COLORS["with_effective"], alpha=0.50)
    ax3.set_xticks(x, labels)
    ax3.set_ylabel("Transmitted fraction")
    ax3.set_title("Transmission")
    ax3.grid(alpha=0.20, axis="y", color="#AAAAAA")

    fig.suptitle("Alpha transport Monte Carlo - with vs without effective charge", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(SUMMARY_PLOT_PATH, dpi=200, bbox_inches="tight")

    dist_fig, dist_axes = plt.subplots(2, 3, figsize=(15, 9))
    dist_axes_flat = dist_axes.ravel()
    dist_axes_flat[-1].set_visible(False)
    for ax_d, material in zip(dist_axes_flat, MATERIALS):
        unit = material_display_length_unit(material)
        scale = material_display_length_scale(material)

        vals_no_eff = results[material.symbol]["without_effective"].exit_x_um_all
        vals_eff = results[material.symbol]["with_effective"].exit_x_um_all
        vals_no_eff = vals_no_eff / scale if material.is_gas else vals_no_eff
        vals_eff = vals_eff / scale if material.is_gas else vals_eff

        vals_all = np.concatenate([vals_no_eff, vals_eff])
        mean_all = float(np.mean(vals_all))
        rms_all = float(np.std(vals_all))
        low = mean_all - 2.0 * rms_all
        high = mean_all + 2.0 * rms_all
        if rms_all <= 0.0:
            low = mean_all - 1e-6
            high = mean_all + 1e-6

        win_no_eff = vals_no_eff[(vals_no_eff >= low) & (vals_no_eff <= high)]
        win_eff = vals_eff[(vals_eff >= low) & (vals_eff <= high)]
        if win_no_eff.size < 2:
            win_no_eff = vals_no_eff
        if win_eff.size < 2:
            win_eff = vals_eff

        n_bins = max(10, int(np.sqrt(len(vals_all))) + 5)
        _, edges = np.histogram(np.concatenate([win_no_eff, win_eff]), bins=n_bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        hist_no_eff, _ = np.histogram(win_no_eff, bins=edges, density=True)
        hist_eff, _ = np.histogram(win_eff, bins=edges, density=True)

        ax_d.bar(centers, hist_no_eff, width=np.diff(edges), color=MODE_COLORS["without_effective"], alpha=0.60, edgecolor=MODE_COLORS["without_effective"], linewidth=0.8)
        ax_d.bar(centers, hist_eff, width=np.diff(edges), color=MODE_COLORS["with_effective"], alpha=0.50, edgecolor=MODE_COLORS["with_effective"], linewidth=0.8)

        grid = np.linspace(low, high, 400)
        ax_d.plot(grid, _manual_kde(win_no_eff, grid), color=MODE_COLORS["without_effective"], linewidth=1.8)
        ax_d.plot(grid, _manual_kde(win_eff, grid), color=MODE_COLORS["with_effective"], linewidth=1.8)

        rms_no_eff = float(np.std(vals_no_eff))
        rms_eff = float(np.std(vals_eff))
        ax_d.axvline(mean_all - rms_no_eff, color=MODE_COLORS["without_effective"], linestyle=":", linewidth=1.2)
        ax_d.axvline(mean_all + rms_no_eff, color=MODE_COLORS["without_effective"], linestyle=":", linewidth=1.2)
        ax_d.axvline(mean_all - rms_eff, color=MODE_COLORS["with_effective"], linestyle="--", linewidth=1.2)
        ax_d.axvline(mean_all + rms_eff, color=MODE_COLORS["with_effective"], linestyle="--", linewidth=1.2)

        ax_d.set_xlim(low, high)
        ax_d.set_title(f"{material.name} ({material.symbol})", fontsize=12, fontweight="bold")
        ax_d.set_xlabel(f"Lateral exit position ({unit})", fontsize=10)
        ax_d.set_ylabel("Probability density", fontsize=10)
        ax_d.legend(
            [
                f"No-eff RMS={rms_no_eff:.3g} {unit}",
                f"Eff RMS={rms_eff:.3g} {unit}",
            ],
            fontsize=8,
            loc="best",
        )
        ax_d.grid(alpha=0.18, color="#AAAAAA")

    dist_fig.suptitle("Exit lateral position distributions: with vs without effective charge", fontsize=14, fontweight="bold")
    dist_fig.tight_layout()
    dist_fig.savefig(LATERAL_DIST_PLOT_PATH, dpi=200, bbox_inches="tight")

    track_fig, track_axes = plt.subplots(1, len(MATERIALS), figsize=(16, 4.8), sharey=False)
    cmap = plt.cm.get_cmap("tab20")
    for axis, material in zip(track_axes, MATERIALS):
        summary_no_eff = results[material.symbol]["without_effective"]
        summary_eff = results[material.symbol]["with_effective"]
        scale = material_display_length_scale(material)
        unit = material_display_length_unit(material)

        for track_index, (track_depths_cm, track_x_cm) in enumerate(
            zip(summary_no_eff.sample_track_depths_cm, summary_no_eff.sample_track_x_cm)
        ):
            valid = np.isfinite(track_depths_cm) & np.isfinite(track_x_cm)
            if np.count_nonzero(valid) < 2:
                continue
            axis.plot(track_depths_cm[valid] * scale, track_x_cm[valid] * scale, linewidth=1.0, alpha=0.60, color=cmap((2 * track_index) % 20))

        for track_index, (track_depths_cm, track_x_cm) in enumerate(
            zip(summary_eff.sample_track_depths_cm, summary_eff.sample_track_x_cm)
        ):
            valid = np.isfinite(track_depths_cm) & np.isfinite(track_x_cm)
            if np.count_nonzero(valid) < 2:
                continue
            axis.plot(track_depths_cm[valid] * scale, track_x_cm[valid] * scale, linewidth=1.0, alpha=0.50, color=cmap((2 * track_index + 1) % 20))

        axis.set_title(material.symbol, fontweight="bold")
        axis.set_xlabel(f"Depth ({unit})")
        axis.set_ylabel(f"Lateral displacement ({unit})")
        axis.grid(alpha=0.18, color="#AAAAAA")

    track_fig.suptitle("Sample alpha-particle trajectories: with and without effective charge", fontsize=14, fontweight="bold")
    track_fig.tight_layout()
    track_fig.savefig(PARTICLE_PLOT_PATH, dpi=200, bbox_inches="tight")

    bragg_fig, solid_axis = plt.subplots(figsize=(11, 6.5))
    gas_axis = inset_axes(solid_axis, width="45%", height="45%", loc="upper right", borderpad=1.2)
    solid_profile_max = 0.0
    gas_profile_max = 0.0
    gas_x_max = 0.0
    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    for material, color, ls in zip(MATERIALS, MATERIAL_COLORS, line_styles):
        summary_no_eff = results[material.symbol]["without_effective"]
        summary_eff = results[material.symbol]["with_effective"]
        scale = material_display_length_scale(material)
        unit = material_display_length_unit(material)

        depth_values = np.linspace(
            0.5 * summary_no_eff.thickness_cm / DEPTH_BINS,
            summary_no_eff.thickness_cm - 0.5 * summary_no_eff.thickness_cm / DEPTH_BINS,
            DEPTH_BINS,
        ) * scale
        profile_no_eff = smooth_profile(summary_no_eff.mean_stopping_profile_mev_per_cm / scale, BRAGG_SMOOTHING_WINDOW)
        profile_eff = smooth_profile(summary_eff.mean_stopping_profile_mev_per_cm / scale, BRAGG_SMOOTHING_WINDOW)
        depth_no_eff, profile_no_eff = resample_profile(depth_values, profile_no_eff, BRAGG_CURVE_POINTS)
        depth_eff, profile_eff = resample_profile(depth_values, profile_eff, BRAGG_CURVE_POINTS)

        peak_idx_no_eff = int(np.argmax(profile_no_eff))
        peak_idx_eff = int(np.argmax(profile_eff))
        peak_depth_no_eff = float(depth_no_eff[peak_idx_no_eff])
        peak_depth_eff = float(depth_eff[peak_idx_eff])

        axis = gas_axis if material.is_gas else solid_axis
        axis.plot(depth_no_eff, profile_no_eff, linewidth=1.8, color=color, linestyle="--", alpha=0.60, label=f"{material.symbol} no-eff peak: {peak_depth_no_eff:.3g} {unit}")
        axis.plot(depth_eff, profile_eff, linewidth=2.0, color=color, linestyle=ls, alpha=0.50, label=f"{material.symbol} eff peak: {peak_depth_eff:.3g} {unit}")
        axis.axvline(peak_depth_no_eff, color=color, linestyle="--", linewidth=0.9, alpha=0.45)
        axis.axvline(peak_depth_eff, color=color, linestyle=":", linewidth=0.9, alpha=0.45)

        # Overlay experimental points for this material if peaks_by_thickness files are present.
        try:
            x_mid_um, dedx_um = load_experimental_bragg_points_for_material(material)
            if material.is_gas:
                x_plot = x_mid_um * 1e-4
                y_plot = dedx_um * 1e4
            else:
                x_plot = x_mid_um
                y_plot = dedx_um

            # Draw a faint connector plus high-contrast markers so experimental points are unmistakable.
            axis.plot(x_plot, y_plot, color="#000000", linewidth=0.9, alpha=0.35, zorder=6)
            gas_marker_map = {"N": "s", "Ar": "^", "He": "X"}
            marker_style = gas_marker_map.get(material.symbol, "o") if material.is_gas else "o"
            marker_size = 82 if material.is_gas else 44
            axis.scatter(
                x_plot,
                y_plot,
                s=marker_size,
                marker=marker_style,
                facecolors="none" if material.is_gas else color,
                edgecolors="#000000",
                linewidths=1.6 if material.is_gas else 1.1,
                alpha=1.0,
                zorder=20 if material.is_gas else 9,
                clip_on=False if material.is_gas else True,
                label=f"{material.symbol} experimental ({x_plot.size} pts)",
            )

            if material.is_gas and x_plot.size > 0:
                gas_x_max = max(gas_x_max, float(np.max(x_plot)))
                gas_profile_max = max(gas_profile_max, float(np.max(y_plot)))
            elif x_plot.size > 0:
                in_solid_view = (x_plot >= 0.0) & (x_plot <= MAIN_BRAGG_X_LIMIT_UM)
                if np.any(in_solid_view):
                    solid_profile_max = max(solid_profile_max, float(np.max(y_plot[in_solid_view])))
        except Exception as exc:
            print(f"Warning: could not overlay experimental points for {material.symbol}: {exc}")

        if not material.is_gas:
            solid_profile_max = max(solid_profile_max, float(np.max(profile_no_eff)), float(np.max(profile_eff)))
        else:
            gas_x_max = max(gas_x_max, float(np.max(depth_no_eff)), float(np.max(depth_eff)))
            gas_profile_max = max(gas_profile_max, float(np.max(profile_no_eff)), float(np.max(profile_eff)))

    solid_axis.set_title("Bragg-style stopping profile with gas inset", fontweight="bold")
    solid_axis.set_xlabel("Depth (um)")
    solid_axis.set_ylabel("Mean dE/dz (MeV/um)")
    solid_axis.set_xlim(0.0, MAIN_BRAGG_X_LIMIT_UM)
    solid_axis.set_ylim(bottom=0.0, top=solid_profile_max * 1.10)
    solid_axis.grid(alpha=0.18, color="#AAAAAA")
    solid_axis.legend(fontsize=8, loc="upper left", framealpha=0.9)

    gas_axis.set_title("Gases", fontsize=10, fontweight="bold")
    gas_axis.set_xlabel("Depth (cm)", fontsize=9)
    gas_axis.set_ylabel("Mean dE/dz (MeV/cm)", fontsize=9)
    if gas_x_max > 0.0:
        gas_axis.set_xlim(0.0, gas_x_max * 1.02)
    if gas_profile_max > 0.0:
        gas_axis.set_ylim(-0.03 * gas_profile_max, gas_profile_max * 1.10)
    gas_axis.tick_params(axis="both", labelsize=8)
    gas_axis.grid(alpha=0.18, color="#AAAAAA")
    gas_axis.legend(fontsize=7, loc="best", framealpha=0.9)

    bragg_fig.suptitle("Depth-resolved stopping power: with vs without effective charge", fontsize=14, fontweight="bold")
    bragg_fig.tight_layout()
    bragg_fig.savefig(BRAGG_PLOT_PATH, dpi=200, bbox_inches="tight")

    # Additional standalone Bragg views (besides inset figure): solids-only and gases-only.
    solid_bragg_fig, solid_bragg_axis = plt.subplots(figsize=(10.5, 6.2))
    gas_bragg_fig, gas_bragg_axis = plt.subplots(figsize=(10.5, 6.2))
    solid_only_profile_max = 0.0
    gas_only_profile_max = 0.0
    gas_only_x_max = 0.0

    for material, color, ls in zip(MATERIALS, MATERIAL_COLORS, line_styles):
        summary_no_eff = results[material.symbol]["without_effective"]
        summary_eff = results[material.symbol]["with_effective"]
        scale = material_display_length_scale(material)
        unit = material_display_length_unit(material)

        depth_values = np.linspace(
            0.5 * summary_no_eff.thickness_cm / DEPTH_BINS,
            summary_no_eff.thickness_cm - 0.5 * summary_no_eff.thickness_cm / DEPTH_BINS,
            DEPTH_BINS,
        ) * scale
        profile_no_eff = smooth_profile(summary_no_eff.mean_stopping_profile_mev_per_cm / scale, BRAGG_SMOOTHING_WINDOW)
        profile_eff = smooth_profile(summary_eff.mean_stopping_profile_mev_per_cm / scale, BRAGG_SMOOTHING_WINDOW)
        depth_no_eff, profile_no_eff = resample_profile(depth_values, profile_no_eff, BRAGG_CURVE_POINTS)
        depth_eff, profile_eff = resample_profile(depth_values, profile_eff, BRAGG_CURVE_POINTS)

        peak_idx_no_eff = int(np.argmax(profile_no_eff))
        peak_idx_eff = int(np.argmax(profile_eff))
        peak_depth_no_eff = float(depth_no_eff[peak_idx_no_eff])
        peak_depth_eff = float(depth_eff[peak_idx_eff])

        axis = gas_bragg_axis if material.is_gas else solid_bragg_axis
        axis.plot(
            depth_no_eff,
            profile_no_eff,
            linewidth=1.8,
            color=color,
            linestyle="--",
            alpha=0.60,
            label=f"{material.symbol} no-eff peak: {peak_depth_no_eff:.3g} {unit}",
        )
        axis.plot(
            depth_eff,
            profile_eff,
            linewidth=2.0,
            color=color,
            linestyle=ls,
            alpha=0.50,
            label=f"{material.symbol} eff peak: {peak_depth_eff:.3g} {unit}",
        )
        axis.axvline(peak_depth_no_eff, color=color, linestyle="--", linewidth=0.9, alpha=0.45)
        axis.axvline(peak_depth_eff, color=color, linestyle=":", linewidth=0.9, alpha=0.45)

        try:
            x_mid_um, dedx_um = load_experimental_bragg_points_for_material(material)
            if material.is_gas:
                x_plot = x_mid_um * 1e-4
                y_plot = dedx_um * 1e4
                marker_style = {"N": "s", "Ar": "^", "He": "X"}.get(material.symbol, "o")
                marker_size = 82
            else:
                x_plot = x_mid_um
                y_plot = dedx_um
                marker_style = "o"
                marker_size = 52

            axis.plot(x_plot, y_plot, color="#000000", linewidth=0.9, alpha=0.35, zorder=6)
            axis.scatter(
                x_plot,
                y_plot,
                s=marker_size,
                marker=marker_style,
                facecolors="none" if material.is_gas else color,
                edgecolors="#000000",
                linewidths=1.6 if material.is_gas else 1.2,
                alpha=1.0,
                zorder=20 if material.is_gas else 12,
                clip_on=False if material.is_gas else True,
                label=f"{material.symbol} experimental ({x_plot.size} pts)",
            )
        except Exception as exc:
            print(f"Warning: could not overlay experimental points for {material.symbol}: {exc}")

        if material.is_gas:
            gas_only_x_max = max(gas_only_x_max, float(np.max(depth_no_eff)), float(np.max(depth_eff)))
            gas_only_profile_max = max(gas_only_profile_max, float(np.max(profile_no_eff)), float(np.max(profile_eff)))
        else:
            solid_only_profile_max = max(solid_only_profile_max, float(np.max(profile_no_eff)), float(np.max(profile_eff)))

    solid_bragg_axis.set_title("Bragg profile: solids only", fontweight="bold")
    solid_bragg_axis.set_xlabel("Depth (um)")
    solid_bragg_axis.set_ylabel("Mean dE/dz (MeV/um)")
    solid_bragg_axis.set_xlim(0.0, MAIN_BRAGG_X_LIMIT_UM)
    if solid_only_profile_max > 0.0:
        solid_bragg_axis.set_ylim(bottom=0.0, top=solid_only_profile_max * 1.10)
    solid_bragg_axis.grid(alpha=0.18, color="#AAAAAA")
    solid_bragg_axis.legend(fontsize=8, loc="upper left", framealpha=0.9)
    solid_bragg_fig.tight_layout()
    solid_bragg_fig.savefig(BRAGG_SOLID_PLOT_PATH, dpi=200, bbox_inches="tight")

    gas_bragg_axis.set_title("Bragg profile: gases only", fontweight="bold")
    gas_bragg_axis.set_xlabel("Depth (cm)")
    gas_bragg_axis.set_ylabel("Mean dE/dz (MeV/cm)")
    if gas_only_x_max > 0.0:
        gas_bragg_axis.set_xlim(0.0, gas_only_x_max * 1.02)
    if gas_only_profile_max > 0.0:
        gas_bragg_axis.set_ylim(-0.03 * gas_only_profile_max, gas_only_profile_max * 1.10)
    gas_bragg_axis.grid(alpha=0.18, color="#AAAAAA")
    gas_bragg_axis.legend(fontsize=8, loc="best", framealpha=0.9)
    gas_bragg_fig.tight_layout()
    gas_bragg_fig.savefig(BRAGG_GAS_PLOT_PATH, dpi=200, bbox_inches="tight")

    plt.show()
    return (
        SUMMARY_PLOT_PATH,
        PARTICLE_PLOT_PATH,
        BRAGG_PLOT_PATH,
        BRAGG_SOLID_PLOT_PATH,
        BRAGG_GAS_PLOT_PATH,
        LATERAL_DIST_PLOT_PATH,
    )


def main() -> None:
    results = run_all_simulations()
    print_table(results)
    print_bragg_peak_positions(results)
    (
        summary_plot_path,
        particle_plot_path,
        bragg_plot_path,
        bragg_solid_plot_path,
        bragg_gas_plot_path,
        lateral_dist_plot_path,
    ) = plot_results(results)
    print(f"Solid thickness used: {SOLID_THICKNESS_UM:.1f} um")
    print(f"Gas thickness used: {GAS_THICKNESS_CM:.2f} cm")
    print(f"Depth intervals used in Monte Carlo: {DEPTH_BINS}")
    print(f"Bragg curve plot points: {BRAGG_CURVE_POINTS}")
    print(f"Bragg smoothing window: {BRAGG_SMOOTHING_WINDOW}")
    print(f"Main Bragg x-limit: {MAIN_BRAGG_X_LIMIT_UM:.1f} um")
    print(f"Summary plot saved to: {summary_plot_path}")
    print(f"Particle plot saved to: {particle_plot_path}")
    print(f"Bragg plot saved to: {bragg_plot_path}")
    print(f"Bragg solids plot saved to: {bragg_solid_plot_path}")
    print(f"Bragg gases plot saved to: {bragg_gas_plot_path}")
    print(f"Lateral distribution plot saved to: {lateral_dist_plot_path}")


if __name__ == "__main__":
    main()
