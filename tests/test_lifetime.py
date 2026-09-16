"""Tests for the optional Gaussian pulse-lifetime envelope (`t_lifetime`).

`Blob.t_lifetime` multiplies the blob by ``exp(-((t - t_init) / t_lifetime)**2)``,
a factor symmetric about `t_init` — as opposed to `t_drain`, which is a
one-sided exponential decay *from* `t_init`. The two are independent and
multiply. The default (None) must leave existing behaviour bit-for-bit
unchanged, which is what the regression tests below pin down.
"""

import numpy as np
import pytest

from blobmodel import (
    Blob,
    BlobShapeEnum,
    BlobShapeImpl,
    DefaultBlobFactory,
    DistributionEnum,
    Geometry,
    Model,
)

ERROR = 1e-10


def _geometry_2d(**kwargs):
    params = dict(Nx=16, Ny=8, Lx=8, Ly=4, dt=0.2, T=10)
    params.update(kwargs)
    return Geometry(**params)


def _geometry_1d(**kwargs):
    params = dict(Nx=16, Ny=1, Lx=8, Ly=0, dt=0.2, T=10)
    params.update(kwargs)
    return Geometry(**params)


def _realize(blob, geometry, one_dimensional=False, **realization_kwargs):
    model = Model.from_blobs(
        [blob],
        geometry=geometry,
        one_dimensional=one_dimensional,
        verbose=False,
    )
    return model.make_realization(**realization_kwargs).n.values


def _blob(**kwargs):
    params = dict(
        amplitude=1.5,
        width_p=0.8,
        width_s=0.8,
        v_x=1.0,
        v_y=0.0,
        pos_x0=1.0,
        pos_y0=2.0,
        t_init=4.0,
    )
    params.update(kwargs)
    return Blob(**params)


# ---------------------------------------------------------------------------
# 1. Regression: the default must not change anything
# ---------------------------------------------------------------------------

# Produced on main (commit 6bdf76a) before t_lifetime and the posx sampler
# existed. They pin the DefaultBlobFactory RNG stream: `posx` is drawn from a
# `zeros` sampler, which consumes no random numbers, so every previously
# seeded realization must come out unchanged.
REFERENCE_BLOB_ATTRIBUTES = [
    [
        0.20653275401650553,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        1.5151621340965677,
        9.955002834343926,
    ],
    [
        0.5685486573832514,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        3.9853471437602312,
        5.045482589579533,
    ],
    [
        0.7075292557919215,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.026326522827873622,
        2.548695876541246,
    ],
    [0.8951098635951629, 1.0, 1.0, 1.0, 0.0, 0.0, 2.339674764218604, 5.534973520744924],
    [1.025203348294905, 1.0, 1.0, 1.0, 0.0, 0.0, 4.106142091913831, 4.450763058826466],
    [3.3836373514362537, 1.0, 1.0, 1.0, 0.0, 0.0, 1.3921280605038666, 7.92661919213753],
]
REFERENCE_FIELD_2D = (210.60648322832878, 0.23808641802939298)
REFERENCE_FIELD_1D = (1093.2711657100344, 0.980969451352523)


@pytest.mark.parametrize("one_dimensional", [False, True])
def test_lifetime_none_is_identical_to_omitting_it(one_dimensional):
    """`t_lifetime=None` multiplies by the float 1.0, which is exact in
    IEEE754: the field must be bit-for-bit what it was without the argument."""
    geometry = _geometry_1d() if one_dimensional else _geometry_2d()
    without = _realize(_blob(), geometry, one_dimensional, speed_up=False)
    with_none = _realize(
        _blob(t_lifetime=None), geometry, one_dimensional, speed_up=False
    )
    assert np.array_equal(with_none, without)


@pytest.mark.parametrize("one_dimensional", [False, True])
def test_lifetime_inf_is_identical_to_none(one_dimensional):
    """An infinite lifetime gives an envelope of exactly 1, so it must agree
    bit-for-bit with no envelope at all (documented equivalence)."""
    geometry = _geometry_1d() if one_dimensional else _geometry_2d()
    none = _realize(_blob(), geometry, one_dimensional, speed_up=False)
    infinite = _realize(
        _blob(t_lifetime=np.inf), geometry, one_dimensional, speed_up=False
    )
    assert np.array_equal(infinite, none)


def test_default_factory_rng_stream_unchanged():
    """Adding the `posx` sampler must not perturb the RNG stream: a seeded
    factory produces exactly the blobs it produced before."""
    factory = DefaultBlobFactory(t_drain=3.0, seed=7)
    blobs = factory.sample_blobs(
        Ly=5.0, T=10.0, num_blobs=6, blob_shape=BlobShapeImpl()
    )
    attributes = [
        [b.amplitude, b.width_p, b.width_s, b.v_x, b.v_y, b.pos_x0, b.pos_y0, b.t_init]
        for b in blobs
    ]
    assert attributes == REFERENCE_BLOB_ATTRIBUTES


@pytest.mark.parametrize(
    "one_dimensional, reference",
    [(False, REFERENCE_FIELD_2D), (True, REFERENCE_FIELD_1D)],
)
def test_seeded_realization_matches_reference(one_dimensional, reference):
    """End-to-end regression against fields computed on main."""
    if one_dimensional:
        geometry, num_blobs, seed = _geometry_1d(Nx=16, Lx=4, dt=0.1, T=6), 20, 3
    else:
        geometry, num_blobs, seed = (
            Geometry(Nx=8, Ny=8, Lx=4, Ly=4, dt=0.2, T=4),
            12,
            11,
        )
    n = (
        Model(
            geometry=geometry,
            num_blobs=num_blobs,
            blob_factory=DefaultBlobFactory(t_drain=2.0, seed=seed),
            one_dimensional=one_dimensional,
            verbose=False,
        )
        .make_realization()
        .n.values
    )
    assert (float(n.sum()), float(n.std())) == reference


def test_default_factory_still_seeds_blobs_at_zero():
    """The `posx` sampler defaults to `zeros`, the previous hardcoded value."""
    factory = DefaultBlobFactory(seed=1)
    blobs = factory.sample_blobs(
        Ly=5.0, T=10.0, num_blobs=8, blob_shape=BlobShapeImpl()
    )
    assert all(blob.pos_x0 == 0.0 for blob in blobs)
    assert factory.t_lifetime is None
    assert all(blob.t_lifetime is None for blob in blobs)


# ---------------------------------------------------------------------------
# 2. Single blob, analytic properties of the envelope
# ---------------------------------------------------------------------------


def test_envelope_is_one_at_t_init_and_one_over_e_at_tau_d():
    blob = _blob(t_lifetime=2.5)
    assert blob._envelope(blob.t_init) == pytest.approx(1.0)
    assert blob._envelope(blob.t_init + 2.5) == pytest.approx(1 / np.e)
    assert blob._envelope(blob.t_init - 2.5) == pytest.approx(1 / np.e)


def test_envelope_is_symmetric_about_t_init():
    blob = _blob(t_lifetime=1.3)
    shifts = np.linspace(0, 5, 51)
    np.testing.assert_allclose(
        blob._envelope(blob.t_init + shifts),
        blob._envelope(blob.t_init - shifts),
        rtol=1e-14,
        atol=0,
    )


def test_envelope_without_lifetime_is_exactly_one():
    assert _blob()._envelope(np.linspace(0, 10, 11)) == 1.0


def test_field_maximum_is_at_t_init():
    """With a lifetime and no draining, `t_init` is the time of maximum
    amplitude — the property `t_drain` alone does not give."""
    geometry = _geometry_1d(Nx=200, Lx=20, dt=0.05, T=10)
    blob = _blob(pos_x0=10.0, t_init=5.0, t_lifetime=1.0, t_drain=np.inf)
    n = _realize(blob, geometry, one_dimensional=True, speed_up=False)
    peak_over_x = n.max(axis=0)
    assert geometry.t[np.argmax(peak_over_x)] == pytest.approx(blob.t_init, abs=0.05)


# ---------------------------------------------------------------------------
# 3. Composition with t_drain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("t_drain", [2.0, "array"])
def test_envelope_multiplies_the_drained_field(t_drain):
    """The lifetime factor is independent of the drain: the field with both
    equals the drain-only field times the envelope, elementwise."""
    geometry = _geometry_2d()
    if t_drain == "array":
        t_drain = np.linspace(3.0, 1.0, geometry.Nx)
    tau_d = 1.7

    drained = _realize(_blob(t_drain=t_drain), geometry, speed_up=False)
    both = _realize(_blob(t_drain=t_drain, t_lifetime=tau_d), geometry, speed_up=False)
    envelope = np.exp(-(((geometry.t - 4.0) / tau_d) ** 2))

    assert drained.max() > 0.1  # guard: the blob really contributes
    np.testing.assert_allclose(both, drained * envelope, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# 4. Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("t_lifetime", [0, -1, -0.5, np.nan])
def test_blob_rejects_non_positive_lifetime(t_lifetime):
    with pytest.raises(ValueError, match="t_lifetime must be positive"):
        Blob(t_lifetime=t_lifetime)


@pytest.mark.parametrize("t_lifetime", [0, -1, -0.5, np.nan])
def test_factory_rejects_non_positive_lifetime(t_lifetime):
    with pytest.raises(ValueError, match="t_lifetime must be positive"):
        DefaultBlobFactory(t_lifetime=t_lifetime)


# ---------------------------------------------------------------------------
# 5. speed_up truncation window
# ---------------------------------------------------------------------------

TRUNCATION_CONFIGS = [
    dict(t_lifetime=1.0),  # lifetime only
    dict(t_lifetime=0.5, t_drain=1.0),  # lifetime + scalar drain
    dict(t_lifetime=2.0, t_drain=0.3),  # short drain: window grows backwards
    dict(t_lifetime=1.0, v_x=0.0),  # no x-velocity: lifetime window alone
    dict(t_lifetime=1.0, v_x=0.0, v_y=1.0),  # purely vertical propagation
    dict(t_lifetime=1.0, t_init=-3.0),  # peak before the time grid
    dict(t_lifetime=1.0, t_init=14.0),  # peak after the time grid
    dict(t_lifetime=3.0, v_x=-1.0, pos_x0=7.0),  # leftward
    dict(t_lifetime=1.0, width_p=0.1, width_s=4.0, theta=np.pi / 2),  # tilted
]


@pytest.mark.parametrize("blob_kwargs", TRUNCATION_CONFIGS)
def test_speed_up_matches_full_realization(blob_kwargs):
    """speed_up must only discard contributions below `truncation_error`."""
    geometry = _geometry_2d(dt=0.05, T=12)
    full = _realize(_blob(**blob_kwargs), geometry, speed_up=False)
    fast = _realize(_blob(**blob_kwargs), geometry, truncation_error=ERROR)
    np.testing.assert_allclose(fast, full, atol=10 * ERROR)


def test_speed_up_matches_full_realization_with_array_drain():
    geometry = _geometry_2d(dt=0.05, T=12)
    kwargs = dict(t_lifetime=1.0, t_drain=np.linspace(2.0, 0.5, geometry.Nx))
    full = _realize(_blob(**kwargs), geometry, speed_up=False)
    fast = _realize(_blob(**kwargs), geometry, truncation_error=ERROR)
    assert full.max() > 0.1  # guard: the blob really contributes
    np.testing.assert_allclose(fast, full, atol=10 * ERROR)


def test_lifetime_window_truncates_a_zero_velocity_blob():
    """Before the lifetime existed, `v_x == 0` returned the full time axis.
    A finite lifetime bounds the support, so the window must shrink."""
    geometry = _geometry_1d(dt=0.05, T=60)
    model = Model.from_blobs(
        [_blob(v_x=0.0, t_lifetime=1.0)], geometry=geometry, one_dimensional=True
    )
    start, stop = model._compute_start_stop(
        _blob(v_x=0.0, t_lifetime=1.0), speed_up=True, truncation_error=ERROR
    )
    assert 0 <= start < stop <= geometry.t.size
    assert stop - start < geometry.t.size / 2


def test_lifetime_window_is_intersected_with_the_crossing_window():
    """A blob that crosses the domain long before its peak time contributes
    nothing: the two windows do not overlap and the result must collapse."""
    geometry = _geometry_1d(dt=0.05, T=60)
    blob = _blob(pos_x0=0.0, v_x=1.0, t_init=50.0, t_lifetime=0.5)
    model = Model.from_blobs([blob], geometry=geometry, one_dimensional=True)
    start, stop = model._compute_start_stop(blob, True, ERROR)
    assert start <= stop  # never inverted
    full = _realize(blob, geometry, one_dimensional=True, speed_up=False)
    fast = _realize(blob, geometry, one_dimensional=True, truncation_error=ERROR)
    np.testing.assert_allclose(fast, full, atol=10 * ERROR)


def test_lifetime_window_widens_as_error_decreases():
    geometry = _geometry_1d(dt=0.05, T=60)
    blob = _blob(v_x=0.0, t_lifetime=1.0)
    model = Model.from_blobs([blob], geometry=geometry, one_dimensional=True)
    widths = [
        np.diff(model._compute_start_stop(blob, True, error))[0]
        for error in [1e-2, 1e-5, 1e-10, 1e-14]
    ]
    assert widths == sorted(widths)


# ---------------------------------------------------------------------------
# 6. Statistical: the process moments of the manuscript model
# ---------------------------------------------------------------------------


def _shape_integrals(blob_shape):
    """(int phi, int phi^2) for a pulse shape, by direct quadrature."""
    theta = np.linspace(-30, 30, 600001)
    phi = blob_shape.get_blob_shape_p(theta)
    d_theta = theta[1] - theta[0]
    return float(np.sum(phi) * d_theta), float(np.sum(phi**2) * d_theta)


@pytest.mark.parametrize("velocity_spread", [0.0, 0.8])
def test_process_moments_with_uniform_seeding(velocity_spread):
    """With a finite lifetime and blobs seeded uniformly in x, the process is
    homogeneous and (Campbell's theorem)

        <Phi>   = gamma <a>,     gamma = sqrt(pi) tau_d l / (tau_w L)
        Var Phi = gamma <a^2> I2 / sqrt(2),   I2 = int phi^2 dtheta

    with L the *seeding* length. Both are independent of the velocity — that
    independence is the point of the envelope, and it is what the old model
    (all blobs seeded at x = 0) does not have. The `velocity_spread`
    parametrization checks exactly that: the same two predictions must hold
    for a degenerate and for a broad velocity distribution.
    """
    tau_d, ell, velocity = 1.0, 1.0, 1.0
    pad, Lx, T, num_blobs = 10.0, 10.0, 200.0, 3400
    seeding_length = Lx + 2 * pad
    tau_w = T / num_blobs
    gamma = np.sqrt(np.pi) * tau_d * ell / (tau_w * seeding_length)

    blob_shape = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)
    _, i_2 = _shape_integrals(blob_shape)

    factory = DefaultBlobFactory(t_lifetime=tau_d, seed=2024)
    factory.set_sampler("wp", DistributionEnum.deg, ell)
    if velocity_spread == 0.0:
        factory.set_sampler("vx", DistributionEnum.deg, velocity)
    else:
        factory.set_sampler(
            "vx",
            lambda rng, n: rng.uniform(
                velocity - velocity_spread, velocity + velocity_spread, n
            ),
        )
    factory.set_sampler("posx", lambda rng, n: rng.uniform(-pad, Lx + pad, n))

    ds = Model(
        geometry=Geometry(Nx=32, Ny=1, Lx=Lx, Ly=0, dt=0.1, T=T),
        blob_shape=blob_shape,
        num_blobs=num_blobs,
        blob_factory=factory,
        one_dimensional=True,
        verbose=False,
    ).make_realization()

    # Discard the time edges, where the seeding window [0, T) makes the
    # process inhomogeneous over a few lifetimes.
    interior = ds.n.sel(t=slice(8 * tau_d, T - 8 * tau_d)).values

    # Exponential amplitudes with mean 1: <a> = 1, <a^2> = 2.
    assert interior.mean() == pytest.approx(gamma * 1.0, rel=0.05)
    assert interior.var() == pytest.approx(gamma * 2.0 * i_2 / np.sqrt(2), rel=0.12)


# ---------------------------------------------------------------------------
# 7. Factory plumbing
# ---------------------------------------------------------------------------


def test_factory_forwards_lifetime_to_every_blob():
    factory = DefaultBlobFactory(t_lifetime=2.5, seed=5)
    blobs = factory.sample_blobs(
        Ly=5.0, T=10.0, num_blobs=10, blob_shape=BlobShapeImpl()
    )
    assert all(blob.t_lifetime == 2.5 for blob in blobs)


def test_posx_sampler_accepts_a_distribution():
    factory = DefaultBlobFactory(seed=5).set_sampler("posx", DistributionEnum.deg, 3.0)
    blobs = factory.sample_blobs(
        Ly=5.0, T=10.0, num_blobs=5, blob_shape=BlobShapeImpl()
    )
    assert all(blob.pos_x0 == 3.0 for blob in blobs)


def test_posx_callable_is_reproducible_through_model_seed():
    def sampler(rng, num_blobs):
        return rng.uniform(-5, 15, num_blobs)

    def positions(seed):
        factory = DefaultBlobFactory().set_sampler("posx", sampler)
        model = Model(
            geometry=_geometry_1d(),
            num_blobs=12,
            blob_factory=factory,
            one_dimensional=True,
            verbose=False,
            seed=seed,
        )
        model.make_realization()
        return [blob.pos_x0 for blob in model.get_blobs()]

    assert positions(17) == positions(17)
    assert positions(17) != positions(18)


def test_posx_sampler_with_wrong_shape_raises():
    factory = DefaultBlobFactory(seed=5).set_sampler(
        "posx", lambda rng, num_blobs: np.zeros(num_blobs + 1)
    )
    with pytest.raises(ValueError, match="sampler for 'posx' returned shape"):
        factory.sample_blobs(Ly=5.0, T=10.0, num_blobs=5, blob_shape=BlobShapeImpl())
