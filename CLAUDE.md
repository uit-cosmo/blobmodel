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
- `blobs.py` — `Blob`: one blob's parameters + `discretize_blob` (analytic
  shape evaluated on the mesh, with periodic-y ghost copies and tilt `theta`).
- `stochasticality.py` — `BlobFactory` (ABC) / `DefaultBlobFactory`: samples
  blob parameters from distributions. The advertised extension point is
  subclassing `BlobFactory` (see `examples/custom_blobfactory.py`).
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
- **Randomness** is currently the legacy global `np.random.*` API — there is no
  per-model seed yet (item 5 proposes threading a `numpy.random.Generator`
  through). Don't introduce more global-state randomness.
- **Angles** (`theta`) are measured from the x-axis, not from the velocity
  vector.
- `t_drain` is a drain *time scale* (exponential decay), not a start time; it
  may be a scalar or an array of length Nx.
- Version is the static `version` in pyproject.toml (bumped manually per
  release, see recent "Up version number" commits); setuptools-scm in
  build-system is vestigial (item 14).
- Keep docstrings in sync with code when changing behavior — docstring drift is
  a recurring problem here (item 9), and docs/RTD autodoc pulls from them.
