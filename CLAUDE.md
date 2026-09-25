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
directly and subclass `BlobFactory`, so treat those as public too.

## Ongoing effort: code-quality cleanup (feedback.txt)

`feedback.txt` at the repo root is the tracking document: a prioritized code
review (P0 correctness → P3 hygiene) with stable item numbers. Working
convention:

- One branch/PR per item or small group of related items, merged to `main` via
  GitHub PR (`gh` CLI available; remote is `uit-cosmo/blobmodel`).
- When an item is fixed and merged, remove its entry from `feedback.txt` and
  note it in the "Progress" header (keep remaining numbers unchanged — they are
  stable IDs).
- Suggested order of attack is at the bottom of feedback.txt. Item 1 (speed_up
  math) was fixed in PR #144; item 2 (theta/blob_alignment contract) is in
  progress on the current `theta_update` branch.
- Behavior changes must be covered by tests (see `tests/test_speed_up.py` for
  the property-based style used for item 1).

## Ongoing effort: API-improvement work package (feedback.md)

`feedback.md` at the repo root is a second tracking document (created
2026-07-22, distinct from feedback.txt): API-usability suggestions derived from
surveying how downstream repos actually use blobmodel. The downstream repos
live locally at `../fusion_scripts` and `../imaging-methods` (directory has a
hyphen; the package inside is `imaging_methods`) — grep them before changing
public API.

Key survey finding driving the package: downstream almost never uses
`DefaultBlobFactory`; the dominant workflow is hand-built `Blob` lists wrapped
in a trivial factory, plus boilerplate converting the output dataset to the
GPI/APD `frames(y, x, time)` + `R`/`Z` format.

- Same working convention as feedback.txt: one branch/PR per item, tests for
  behavior changes.
- Open GitHub issues mapping onto items, to be closed by the implementing
  PRs: #132 dataset in cmod_functions format (→ item 6), #101 rework
  DefaultBlobFactory config (→ item 10, low priority).
- Suggested order is at the bottom of feedback.md:
  1) Blob defaults, `lam` convention docs; 2) output-layout helper,
  speed_up default; 3) docs-script fixes.
- Item 1 (Geometry flexibility, #140) merged in PR #152 (2026-07-22): grid
  params moved from `Model` to `Geometry` (breaking — 2.0.0), `x0`/`y0`
  offsets, `Geometry.from_arrays`, read-only `model.geometry`.
- Items 2, 3, 8 merged in PR #153 (2026-07-23, closed #93): added
  `BlobListFactory`/`CallableBlobFactory`/`Model.from_blobs`; breaking —
  `t_drain` removed from `Model` and from the `BlobFactory.sample_blobs`
  signature (now a `DefaultBlobFactory` constructor arg, default `np.inf` =
  no draining, so a bare `Model()` no longer drains; previously
  `t_drain=10`). Downstream repos not yet migrated.
- Items 4 and 5 implemented 2026-07-23 on branch `blob_defaults_lam`: all
  `Blob.__init__` parameters now have defaults (order unchanged;
  `Blob()` = unit Gaussian blob, `v_x=1`, no draining) and
  `labels="individual"` labels blobs by factory-output position
  (`blob_id` is pure metadata now). Breaking: `double_exp` `lam` flipped
  to the FPP convention (see gotcha below) — downstream `1 - lam`
  workarounds must be removed on migration.
- Item 6 implemented 2026-07-23: PR #155 (closed #132) added
  `make_realization(layout="imaging")` / `to_imaging_dataset` (non-breaking);
  branch `one_dim_squeeze` then made Ly=0 output squeezed — `n(x, t)`, no
  `y` dimension (breaking: downstream `isel(y=0)` on 1D output must go).
- Items 7 and 9 implemented 2026-07-23 on branch `misc_api_cleanup`
  (PR #157): `make_realization` defaults to `speed_up=True`,
  `truncation_error=1e-10` (breaking: `error` renamed to
  `truncation_error` — downstream `speed_up=True, error=1e-10`
  boilerplate can just be dropped); `get_blobs()` before
  `make_realization()` raises RuntimeError; burn-in documented
  ("Stationarity and burn-in" in `blob_factory.rst`: negative blob
  `t_init` via `CallableBlobFactory`; a real `burn_in=` option was
  deliberately deferred).
- Items 10 and 11 implemented 2026-07-23 on branch `docs_factory_cleanup`
  (closes #101). Breaking: `DefaultBlobFactory` lost its fourteen
  `*_dist`/`*_parameter` ctor args in favor of `set_sampler` (see
  Architecture above); defaults unchanged (exp amplitude, rest
  degenerate). Item 11: `docs/create_logo.py` repaired; docs plot
  scripts now covered by headless smoke tests
  (`tests/test_docs_scripts.py`). All feedback.md items done.
- With all breaking items landed, 2.0.0 was released (bump in `6bdf76a`,
  2026-07-23); 2.1.0 followed with PR #163 (pulse lifetime, `posx`
  sampler).

## Commands

```bash
pip install -e .            # deps currently include dev tools (feedback item 14)
pytest                      # full suite, ~tests/ ; some tests are statistical
                            #   and unseeded (item 17) — a rare flake is known
black .                     # formatting; CI runs `black --check .`
mypy --ignore-missing-imports blobmodel   # CI runs this too
```

No pytest/mypy/black config sections exist in pyproject.toml — defaults apply.
CI is `.github/workflows/workflow.yml` (currently duplicated jobs on Python
3.10; item 16). Tests currently drop `*.nc` artifacts in the repo root and
`tests/` (item 17) — don't commit them.

## Conventions and gotchas

- **Formatting/typing**: black-formatted, numpydoc docstrings, type hints use
  `nptyping.NDArray` (nptyping is unmaintained and blocks newer Python; don't
  add new nptyping usage if a plain `np.ndarray` annotation works).
- **Array axis order** is `(y, x, t)` i.e. `(Ny, Nx, Nt)` — several docstrings
  wrongly say `(x, y, t)` (item 9). When in doubt trust the code, not the
  docstring.
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
  release, see recent "Up version number" commits); setuptools-scm in
  build-system is vestigial (item 14).
- Keep docstrings in sync with code when changing behavior — docstring drift is
  a recurring problem here (item 9), and docs/RTD autodoc pulls from them.
