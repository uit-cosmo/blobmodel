# CLAUDE.md

## What this repo is

`blobmodel` is a small scientific Python package (uit-cosmo, ~1500 lines) that
generates 2D (or 1D) superpositions of propagating "blobs" — a stochastic model
used in fusion plasma edge-turbulence studies. Output is an xarray Dataset
(density field `n(y, x, t)`), optionally written to netCDF. Published in JOSS
(`paper/`), docs on Read the Docs (`docs/`, Sphinx + napoleon).

## Architecture

All source lives in `blobmodel/` (flat, one class-cluster per file):

- `model.py` — `Model`: owns the grid, sums discretized blobs into the density
  field (`make_realization`), handles `speed_up`/truncation and blob labels.
  Also `to_imaging_dataset` (module-level): converts the output dataset to
  the GPI/APD `frames(y, x, time)` + `R`/`Z` format, also reachable via
  `make_realization(layout="imaging")`.
- `blobs.py` — `Blob`: one blob's parameters + `discretize_blob` (analytic
  shape evaluated on the mesh, with periodic-y ghost copies and tilt `theta`).
- `stochasticality.py` — `BlobFactory` (ABC) / `DefaultBlobFactory`: samples
  blob parameters independently; configured via the chainable
  `set_sampler(parameter, sampler, free_parameter=None)` with parameter keys
  "amplitude"/"wp"/"ws"/"vx"/"vy"/"spp"/"sps"/"posx" and `sampler` either a
  `DistributionEnum` or a `ParameterSampler` callable
  (`(rng, num_blobs) -> np.ndarray`); ctor takes only `t_drain`,
  `blob_alignment`, `seed`, `t_lifetime`. Also `BlobListFactory` (pre-built blob
  list; used by `Model.from_blobs`) and `CallableBlobFactory`
  (`blob_getter(rng) -> Blob`, the seedable path for hand-rolled sampling).
  Subclassing `BlobFactory` remains the general extension point (see
  `examples/custom_blobfactory.py`).
- `distributions.py` — `DistributionEnum` → sampling functions (`DISTRIBUTIONS`
  dict of plain functions).
- `blob_shape.py` — `AbstractBlobShape` / `BlobShapeImpl` / `BlobShapeEnum`:
  pulse shapes in propagation (p) and perpendicular (s) directions.
- `geometry.py` — `Geometry`: grid definition (1D coordinate arrays `x`, `y`,
  `t`; the Model broadcasts them, no full meshgrids are stored).
- `plotting.py` — `show_model` animation helper (matplotlib is imported lazily
  inside the functions, so importing blobmodel stays matplotlib-free).

Public API is whatever `blobmodel/__init__.py` re-exports. Downstream users
(uit-cosmo repos `fusion_scripts`, `imaging_methods`) call `discretize_blob`
directly and subclass `BlobFactory`, so treat those as public too. Local
checkouts live at `../fusion_scripts` and `../imaging-methods` (hyphenated
directory; the package inside is `imaging_methods`) — grep them before
changing public API.

`main` is protected: changes land via GitHub PR (`gh` CLI; remote
`uit-cosmo/blobmodel`), one branch per change. Behavior changes must be
covered by tests (`tests/test_speed_up.py` shows the property-based style).

## Commands

```bash
pip install -e ".[dev]"     # dev tools are the `dev` extra; docs deps are `docs`
pytest                      # full suite, ~tests/ ; some tests are statistical
                            #   and unseeded — a rare flake is known
black .                     # formatting; CI runs `black --check .`
mypy --ignore-missing-imports blobmodel   # CI runs this too
```

No pytest/mypy/black config sections exist in pyproject.toml — defaults apply.
CI is `.github/workflows/workflow.yml`: one job on a Python 3.10/3.11 matrix;
black, mypy and the Codecov upload run on the 3.10 leg only.

## Conventions and gotchas

- **Formatting/typing**: black-formatted, numpydoc docstrings, type hints use
  `nptyping.NDArray` (nptyping is unmaintained and blocks newer Python; don't
  add new nptyping usage if a plain `np.ndarray` annotation works).
- **Array axis order** is `(y, x, t)` i.e. `(Ny, Nx, Nt)`, although the math
  notation in the docs writes `(x, y, t)`.
- **theta vs blob_alignment**: explicit `theta` (not None) wins and
  `blob_alignment` is ignored; `theta=None` falls back to alignment. Both
  `Blob` and `DefaultBlobFactory` default `blob_alignment=False` (the factory
  defaulted `True` before v1.2.2's theta fix — a behavior change worth a
  changelog mention).
- **Randomness**: seeding exists — `Model(seed=...)` and
  `DefaultBlobFactory(seed=...)` thread a `numpy.random.Generator` through
  (`BlobFactory.set_rng` / `self.rng`); a seed passed to `Model` overrides the
  factory's. Custom factories are only seedable if they draw from `self.rng`;
  `CallableBlobFactory` passes its rng to the `blob_getter` for the same
  reason — downstream `blob_getter` functions still use global `np.random`
  until they migrate. Don't introduce global-state `np.random.*` randomness.
- **Angles** (`theta`) are measured from the x-axis, not from the velocity
  vector.
- **`lam` (double_exp asymmetry)**: since 2.0.0 `lam` weighs the *leading*
  (`theta >= 0`) side of the spatial pulse — i.e. the temporal *rise*
  fraction at a fixed probe for `v_x > 0`, matching the FPP-literature
  convention. Pre-2.0 it weighted the trailing side (temporal fall), which
  is why old downstream code passes `1 - lam`.
- `t_drain` is a drain *time scale* (exponential decay), not a start time; it
  may be a scalar or an array of length Nx (length checked in
  `make_realization`). It lives on `Blob`/the factory
  (`DefaultBlobFactory(t_drain=...)`), not on `Model`, and defaults to
  `np.inf` = no draining (the documented replacement for the old `1e10`
  folk convention).
- **`t_drain` vs `t_lifetime`** (added in 2.1.0): easy to confuse, and they
  coexist and multiply. `t_drain` is a *one-sided* exponential decay from
  `t_init` (it grows without bound backwards in time); `t_lifetime` is a
  Gaussian envelope *symmetric about* `t_init`, `exp(-((t - t_init)/tau_d)^2)`,
  scalar only, `None` by default. `None` takes the drain-only path in
  `Blob._temporal_factor` — bit-for-bit unchanged — which `tests/test_lifetime.py`
  asserts against reference values computed on the pre-2.1.0 `main`. A finite
  lifetime sums both exponents before `np.exp` (separate factors give
  `inf * 0 = NaN` far before `t_init`).
- **`DefaultBlobFactory` seeds blobs at `pos_x0 = 0`** unless the `posx`
  sampler is reconfigured (added in 2.1.0, default `DistributionEnum.zeros`,
  which draws nothing and so leaves the RNG stream untouched). The factory
  cannot see `Lx` (issue #140), so seeding blobs across the domain means
  passing a callable: `set_sampler("posx", lambda rng, n: rng.uniform(...))`.
  With a finite `t_lifetime` this is a prerequisite, not a nicety — otherwise
  every pulse peaks at the inflow edge.
- Version is the static `version` in pyproject.toml (bumped manually per
  release, see recent "Up version number" commits).
- Keep docstrings in sync with code when changing behavior — docstring drift is
  a recurring problem here, and docs/RTD autodoc pulls from them.
