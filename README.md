# ALPHA_PARTICLE_ENERGY_LOSS

A third-year Physics project investigating the energy loss of alpha particles in different materials using the Bethe-Bloch stopping power framework.

The project combines experimental data analysis, theoretical modelling, and Monte Carlo simulation to study charged particle interactions with matter. Experimental measurements were performed for gaseous helium (He), nitrogen (N₂), and argon (Ar), alongside metallic aluminium (Al) and nickel (Ni). For each material, the repository contains the corresponding analysis code, processed datasets, and generated figures.

Project Components
Experimental Data Analysis
Energy calibration of silicon detector spectra.
Automated peak extraction using multiple fitting models.
Stopping power calculation from measured alpha-particle energy loss.
Extraction of mean excitation energies (I-values) using the Bethe-Bloch formalism.

Material-Specific Analysis

Aluminium (Al)
Nickel (Ni)
Helium (He)
Nitrogen (N₂)
Argon (Ar)

Each material has its own directory containing analysis scripts, datasets, and figures.

Monte Carlo Simulation
Independent Monte Carlo simulation of alpha-particle transport through the same materials.
Comparison between simulated energy loss and theoretical Bethe-Bloch predictions.
Validation of the analytical framework against stochastic particle transport.
Repository Structure
Al/
├── code/
├── data/
└── figures/

Ni/
├── code/
├── data/
└── figures/

He/
├── code/
├── data/
└── figures/

N/
├── code/
├── data/
└── figures/

Ar/
├── code/
├── data/
└── figures/

Simulation/
└── Monte Carlo implementation for alpha-particle transport
Topics Covered
Bethe-Bloch stopping power
Alpha-particle energy loss
Mean excitation energy (I-value)
Silicon detector calibration
Peak fitting and uncertainty analysis
Numerical integration
Monte Carlo particle transport
Comparison between experiment, simulation, and theory

This repository contains all analysis scripts, datasets, figures, and simulation code developed for the third-year undergraduate Physics project at the University of Manchester.
