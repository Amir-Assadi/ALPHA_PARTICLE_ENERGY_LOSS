# ALPHA_PARTICLE_ENERGY_LOSS

# Alpha Particle Energy Loss: Bethe-Bloch Stopping Power

This repository contains the analysis scripts, datasets, figures, and Monte Carlo simulation code developed for a third-year undergraduate Physics project at the University of Manchester. 

The project investigates the energy loss of alpha particles in various materials using the Bethe-Bloch stopping power framework. It combines experimental data analysis, theoretical modelling, and Monte Carlo simulations to study charged particle interactions with matter.

---

## Project Components

* **Experimental Data Analysis:** Energy calibration of silicon detector spectra.
* **Automated Peak Extraction:** Utilizing multiple fitting models to isolate relevant data.
* **Stopping Power Calculation:** Derived directly from measured alpha-particle energy loss.
* **I-Value Extraction:** Determining mean excitation energies using the Bethe-Bloch formalism.
* **Monte Carlo Simulation:** Independent stochastic particle transport modelling to validate the analytical framework.
* **Theory vs. Experiment:** Direct comparison between simulated energy loss, experimental measurements, and theoretical predictions.

## Materials Analyzed

Experimental measurements and theoretical simulations were performed across a range of gaseous and metallic targets. Each target has an independent analysis pipeline.

| Material | State | Symbol | Directory |
| :--- | :--- | :--- | :--- |
| **Aluminium** | Metallic | Al | `/Al/` |
| **Nickel** | Metallic | Ni | `/Ni/` |
| **Helium** | Gaseous | He | `/He/` |
| **Nitrogen** | Gaseous | N₂ | `/N/` |
| **Argon** | Gaseous | Ar | `/Ar/` |

## Repository Structure

Each material directory is self-contained with its own data visualization scripts, processed datasets, and generated figures.

```text
.
├── Al/                 
│   ├── code/
│   ├── data/
│   └── figures/
├── Ar/                 
│   ├── code/
│   ├── data/
│   └── figures/
├── He/                 
│   ├── code/
│   ├── data/
│   └── figures/
├── N/                  
│   ├── code/
│   ├── data/
│   └── figures/
├── Ni/                 
│   ├── code/
│   ├── data/
│   └── figures/
└── Simulation/         
    └── Monte Carlo implementation for alpha-particle transport
