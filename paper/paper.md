---
title: 'Blobmodel: A Python package for generating superpositions of pulses in one and two dimensions'
tags:
  - Python
  - pulses
  - imaging data 
  - tokamaks
  - turbulence
authors:
  - name: Juan M. Losada
    orcid: 0000-0003-2054-1384
    equal-contrib: true
    affiliation: 1
  - name: Gregor Decristoforo
    orcid: 0000-0002-7616-0946
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
affiliations:
 - name: UiT, The Arctic University of Norway
   index: 1
date: 11 February 2025
bibliography: paper.bib

---

# Summary

`blobmodel` is a Python library for generating synthetic data that mimics the behavior
of moving pulses in a turbulent environment. It creates controlled datasets where the
true motion of each pulse is known, allowing researchers to gain further understanding on
the statistical outputs of such systems as well as to test and improve analysis 
and tracking algorithms. While originally developed for studying
turbulence in fusion experiments, `blobmodel` can be applied to any field where
turbulence leads to the generation of structures propagating in one or two dimensions.
The software is open source, easy to use, and designed to support reproducible research.

# Statement of need

Understanding and analyzing the motion of structures in turbulent systems is crucial
in many areas of research, including plasma physics [@dippolito_convective_2011],
fluid dynamics [@fiedler_coherent_1988], and atmospheric science [@nosov_coherent_2009]. 

More widely, the study of the statistical characteristics resulting from the superposition
of uncorrelated propagating pulses is of importance in fields such as astrophysical plasmas [@veltri_mhd_1999], 
detection rates of interplanetary dust [@kociscak_modeling_2023], 
$1/f$ noise in self-organized critical systems [@bak_self_1988], 
and shot noise in electronics [@lowen_power_1990].

In experimental studies, imaging diagnostics are often used to capture the 
evolution of these structures [@zweben_invited_2017], but extracting reliable velocity information from such 
data remains challenging [@offeddu_analysis_2023]. Many existing analysis methods rely on assumptions about 
the underlying dynamics and must be tested against known reference data to ensure
accuracy.

Several stochastic models have been developed describing a point process resulting from the superposition of 
uncorrelated structures with random arrival times [@garcia_stochastic_2016]; or propagating in one [@losada_stochastic_2023]
or two [@militello_two-dimensional_2018] spatial dimensions. In the simplest cases, it is possible
to derive analytical expressions for different statistical quantities such as probability density functions, 
autocorrelation functions, power spectral densities and spatial dependence of the mean or other higher-order
moments [@garcia_stochastic_2016; @militello_relation_2016; @losada_stochastic_2023]. More general scenarios 
require numerical tools [@losada_stochastic_2024], and synthetic realizations of the model.

`blobmodel` addresses this need by providing a framework for generating synthetic 
datasets resulting from a superposition of uncorrelated pulses [@militello_two-dimensional_2018; @losada_tde_2025]:
$$
    \Phi(x,y,t) = \sum_{k=1}^{K} a_k \varphi\left( \frac{x-v_k(t-t_k)}{\ell_{x, k}}, \frac{(y-y_k)-w_k(t-t_k)}{\ell_{y, k}}\right) ,
$$
where:

- $a_k$ represents the initial pulse amplitude.
- $v_k$ and $w_k$ are the horizontal and vertical velocity components, respectively.
- $t_k$ is the pulse arrival time at the position $x=0$, $y=y_k$.
- $y_k$ is the pulse vertical position at time $t=t_k$.
- $\ell_{x, k}$ and $\ell_{y, k}$ are the horizontal and vertical pulse sizes, respectively.
- $\varphi$ is an unspecified pulse shape.

All these parameters, except for the pulse shape $\varphi$ are assumed to be random variables. 
Additionally, each pulse may be tilted on an angle given by an additional random variable $\theta_k$ with respect to its centre.

The framework allows an explicit definition of all relevant process parameters, including:

- All pulse parameters if defined as degenerate random variables.
- All distribution functions of the pulse parameters otherwise.
- Optionally, a drainage term $\tau_\shortparallel$ that models an exponential decay in the pulse amplitude through an
additional factor $e^{-\frac{t-t_k}{\tau_\shortparallel}}$ in the pulse evolution.
- Spatial and temporal resolution.
- Degree of pulse overlap by setting different ratios of number of pulses to signal length and domain size.
- Total duration of the process.

This allows researchers to systematically test and benchmark
tracking algorithms and velocity estimation techniques in a controlled setting. 
Originally developed for studying turbulence-driven transport in fusion plasma
experiments, blobmodel is applicable to any system where turbulence leads to the
formation of moving structures in one or two-dimensional space. By offering an open-source
and easily accessible tool, blobmodel supports the development and validation of 
analysis methods used in experimental and computational research. 

To the authors' knowledge, no other packages exist which provide a comprehensive, 
open-source framework for generating synthetic datasets of uncorrelated, propagating
pulses in one or two spatial dimensions, with fully customizable statistical distributions
for pulse parameters and explicit control over spatial and temporal resolution, pulse overlap,
and drainage effects, as implemented in `blobmodel`.

For more details visit [blobmodel's documentation](https://blobmodel.readthedocs.io/en/latest/).

The package has been used to generate synthetic data to study and compare the robustness of
velocity estimation techniques on coarse-grained imaging data [@losada_three-point_2024].

Additionally, theoretically predicted radial profiles from stochastic modelling
[@garcia_stochastic_2016; @militello_relation_2016] agree with those obtained with `blobmodel`. 

# Implementation details

The evolution of the pulses is discretized by the `Blob` class in a three dimensional grid
(two space and one time dimensions) according to the above formula. The discretization grid is provided
by the `Geometry` and the superposition of all pulses is performed by the `Model` class, which also contains functions
for the model initialization. The generation of pulses with pulse parameters following user-specified distribution
functions is performed by the `BlobFactory`.

Since the simulation domain has finite spatial extent, pulses may originate or extend beyond its boundaries.
If a pulse has a non-bound shape, such as a Gaussian, its tails can still contribute to the
superposition inside the domain. However, in long simulations, most pulses will exist outside the 
domain for the majority of the time, making it computationally inefficient to account for all of them.
To improve efficiency, a `speed_up` flag has been added to `Model.make_realization`. When enabled, the model
ignores pulses whose contribution within the domain falls below a user-defined `error` threshold. 
This allows for significant computational savings while maintaining accuracy in the simulation.

Periodicity in the vertical direction is an optional feature. It is implemented by replicating each pulse 
at vertical positions $y_b\pm L_y$, where $y_b$ is the pulse's original position and $L_y$ is the vertical
size of the simulation domain. This ensures that blobs crossing the upper or lower boundaries are correctly 
wrapped around, maintaining continuity in the periodic direction.

This package is fully compatible with `xarray` [@hoyer_xarray_2017], with all outputs provided as `xarray` datasets
for easy handling and analysis.

# Acknowledgements

This work was supported by the UiT Aurora Centre Program, UiT The Arctic University of Norway (2020).

# References
