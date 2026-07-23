"""This module defines a 2D model of propagating blobs."""

import numpy as np
import xarray as xr
from tqdm import tqdm
from typing import List, Union
from .blobs import Blob
from .stochasticality import BlobFactory, BlobListFactory, DefaultBlobFactory
from .geometry import Geometry
import warnings
from .blob_shape import AbstractBlobShape, BlobShapeImpl


class Model:
    """
    Class storing all parameters relevant for the realization of a random process of a superposition of
    uncorrelated pulses propagating in two dimensions. The realization is performed by calling `make_realization` which:
        - uses a `BlobFactory` to make a list of blobs following the specified blob parameter distribution functions,
        - each 'Blob' is discretized by calling its `discretize_blob` function and
        - the discretization is performed on a grid given by the `Geometry`.
    """

    def __init__(
        self,
        geometry: Union[Geometry, None] = None,
        blob_shape: Union[AbstractBlobShape, None] = None,
        num_blobs: int = 1000,
        blob_factory: Union[BlobFactory, None] = None,
        labels: str = "off",
        label_border: float = 0.75,
        one_dimensional: bool = False,
        verbose: bool = True,
        seed: Union[int, np.random.Generator, None] = None,
    ) -> None:
        """
        Initialize the 2D Model of propagating blobs.

        Parameters
        ----------
        geometry : Geometry, optional
            Grid on which the blobs are discretized. By default None, in which
            case a default `Geometry` is created (`Geometry()` for a 2D model,
            `Geometry(Ny=1, Ly=0)` if ``one_dimensional``). Use
            `Geometry.from_arrays` to build a geometry from explicit
            coordinate arrays.
        blob_shape : AbstractBlobShape, optional
            Shape of the blobs. Can be an instance of AbstractBlobShape.
            By default None, in which case a gaussian `BlobShapeImpl` is created.
        num_blobs : int, optional
            Number of blobs.
        blob_factory : BlobFactory, optional
            BlobFactory instance for setting blob parameter distributions.
            By default None, in which case a `DefaultBlobFactory` is created
            (whose default is non-draining blobs, `t_drain=np.inf` — blob
            draining is owned by the factory, not the model).
        labels : str, optional
            Blob label setting. Possible values: "off", "same", "individual".
            "off": no blob labels returned
            "same": regions where blobs are present are set to label 1
            "individual": each blob gets its own label, 1..num_blobs in the
            order the factory returns the blobs
            Used for creating training data for supervised machine learning algorithms
        label_border : float, optional
            Defines region of blob as region where density >= label_border * amplitude of Blob
            Only used if labels = "same" or "individual"
        one_dimensional : bool, optional
            If True, the perpendicular shape of the blobs will be discarded
            and the perpendicular blob shape will be set to 1. Requires a
            geometry with Ny=1 and Ly=0 (the default geometry is adjusted
            accordingly).
        verbose : bool, optional
            If True, print a loading bar.
        seed : int, np.random.Generator or None, optional
            Seed (or an already constructed `numpy.random.Generator`) for the
            random number generator used to sample blob parameters. Two models
            constructed with the same seed and parameters produce identical
            realizations. The generator is handed to the blob factory via
            `BlobFactory.set_rng`, replacing any seed the factory was
            constructed with; custom factories only honor it if they draw from
            `self.rng`. By default None, i.e. the factory's own generator is
            kept (non-reproducible unless the factory was seeded).

        Notes
        -----
        - `num_blobs` and `blob_shape` are only forwarded to the blob
          factory's `sample_blobs`; a custom factory may ignore either of
          them (`BlobListFactory` ignores both). For pre-built blob lists
          prefer `Model.from_blobs`, which hides these parameters.

        Raises
        ------
        TypeError
            If geometry is not a Geometry instance, blob_shape is not an
            AbstractBlobShape instance or blob_factory is not a BlobFactory
            instance.
        ValueError
            If the model is one-dimensional and the geometry does not have
            Ny=1 and Ly=0.

        Warns
        -----
        UserWarning
            If the model is one-dimensional and the blob factory is not one-dimensional.
        """
        if blob_shape is None:
            blob_shape = BlobShapeImpl()
        if blob_factory is None:
            blob_factory = DefaultBlobFactory()
        if geometry is None:
            geometry = Geometry(Ny=1, Ly=0) if one_dimensional else Geometry()
        if not isinstance(geometry, Geometry):
            raise TypeError(
                f"geometry must be a Geometry, got {type(geometry).__name__}."
            )
        if not isinstance(blob_shape, AbstractBlobShape):
            raise TypeError(
                f"blob_shape must be an AbstractBlobShape, got {type(blob_shape).__name__}."
            )
        if not isinstance(blob_factory, BlobFactory):
            raise TypeError(
                f"blob_factory must be a BlobFactory, got {type(blob_factory).__name__}."
            )
        if seed is not None:
            blob_factory.set_rng(np.random.default_rng(seed))
        self._one_dimensional = one_dimensional
        if self._one_dimensional:
            if geometry.Ny != 1 or geometry.Ly != 0:
                raise ValueError(
                    "A one dimensional model requires a geometry with Ny=1 and "
                    f"Ly=0, got Ny={geometry.Ny} and Ly={geometry.Ly}."
                )
            if not blob_factory.is_one_dimensional():
                warnings.warn(
                    "Using a one dimensional model with a blob factory that is not one-dimensional. Are you sure you know what you are doing?"
                )

        self._geometry: Geometry = geometry
        self.blob_shape = blob_shape
        self.num_blobs: int = num_blobs

        self._blobs: List[Blob] = []
        self._blob_factory = blob_factory
        self._labels = labels
        self._label_border = label_border
        self._reset_fields()
        self._verbose = verbose

    def __str__(self) -> str:
        """
        Return a string representation of the Model.

        Returns
        -------
        str
            String representation of the Model.
        """
        return f"2d Blob Model with num_blobs:{self.num_blobs}"

    @classmethod
    def from_blobs(
        cls,
        blobs: List[Blob],
        geometry: Union[Geometry, None] = None,
        labels: str = "off",
        label_border: float = 0.75,
        one_dimensional: bool = False,
        verbose: bool = True,
    ) -> "Model":
        """
        Create a Model that realizes a pre-built list of blobs.

        Shortcut for the hand-built-blobs workflow: wraps `blobs` in a
        `BlobListFactory`, so the sampling parameters of `Model.__init__`
        (`num_blobs`, `blob_shape`) need not be supplied — each `Blob`
        already carries its own parameters.

        Parameters
        ----------
        blobs : List[Blob]
            Blobs to sum in `make_realization`.
        geometry : Geometry, optional
            Grid on which the blobs are discretized, as in `Model.__init__`.
        labels : str, optional
            Blob label setting, as in `Model.__init__`.
        label_border : float, optional
            Defines region of blob, as in `Model.__init__`.
        one_dimensional : bool, optional
            If True, the perpendicular shape of the blobs is discarded, as in
            `Model.__init__`.
        verbose : bool, optional
            If True, print a loading bar.

        Returns
        -------
        Model
            Model whose realizations sum exactly the given blobs.
        """
        return cls(
            geometry=geometry,
            num_blobs=len(blobs),
            blob_factory=BlobListFactory(blobs),
            labels=labels,
            label_border=label_border,
            one_dimensional=one_dimensional,
            verbose=verbose,
        )

    @property
    def geometry(self) -> Geometry:
        """Geometry: The grid the model discretizes the blobs on (read-only)."""
        return self._geometry

    def get_blobs(self) -> List[Blob]:
        """
        Return the list of blobs summed up in the last realization.

        Returns
        -------
        List[Blob]
            List of Blob objects.

        Raises
        ------
        RuntimeError
            If no blobs have been sampled yet. Blobs are sampled by
            ``make_realization``; call it first.

        """
        if not self._blobs:
            raise RuntimeError(
                "No blobs have been sampled yet: blobs are sampled by "
                "make_realization(), call it first."
            )
        return self._blobs

    def make_realization(
        self,
        file_name: Union[str, None] = None,
        speed_up: bool = True,
        truncation_error: float = 1e-10,
        layout: str = "default",
    ) -> xr.Dataset:
        """
        Integrate the Model over time and write out data as an xarray dataset.

        Parameters
        ----------
        file_name : str, optional
            File name for the .nc file containing data as an xarray dataset.
        speed_up : bool, optional
            Speed up the code by summing up each blob only over the time
            window where its amplitude on the grid exceeds
            ``truncation_error``; the rest of the blob is discarded.
            Enabled by default; set to False for blob shapes with
            slowly-decaying tails (see Notes).
        truncation_error : float, optional
            Amplitude below which a blob is truncated when ``speed_up`` is
            enabled.
        layout : str, optional
            Layout of the returned (and saved) dataset. Possible values:
            "default": density `n(y, x, t)` with 1D coordinates `x`, `y`, `t`.
            "imaging": the GPI/APD imaging format `frames(y, x, time)` with
            2D coordinates `R(y, x)`, `Z(y, x)`, as returned by
            `to_imaging_dataset`. Requires a two-dimensional geometry.

        Returns
        -------
        xr.Dataset
            xarray dataset with the data resulting from a realization of the process described by the model
            and evaluated in a three-dimensional grid with dimensions:
            - x: Horizontal coordinate
            - y: Vertical coordinate
            - t: Time coordinate
            The resulting blob density, evaluated in the grid, is given by the `DataArray`, `n`, with
            dimension order (y, x, t), i.e. shape (Ny, Nx, Nt). For a grid
            with Ly = 0 (one-dimensional model), the `y` dimension is dropped
            and `n` has dimensions (x, t).
            With ``layout="imaging"`` the density is instead the `DataArray`
            `frames` with dimensions (y, x, time), see `to_imaging_dataset`.


        Raises
        ------
        ValueError
            If ``layout`` is not one of the values listed above, if
            ``layout="imaging"`` is requested for a one-dimensional model, or
            if a sampled blob has an array-valued t_drain whose length does
            not match the geometry's Nx.

        Warns
        -----
        UserWarning
            If periodic_y is set and a sampled blob width is large compared to
            the domain size Ly, in which case the mirror blobs used to
            implement the periodicity may become apparent.

        Notes
        -----
        - The truncation window used by speed_up assumes an exponentially
          decaying pulse shape (blob_shape="exp"). For shapes with more
          slowly decaying tails (e.g. lorentz) pass ``speed_up=False``.
        """
        # Validate the layout before doing any expensive work.
        if layout not in {"default", "imaging"}:
            raise ValueError(
                f'layout must be "default" or "imaging", got layout = "{layout}".'
            )
        if layout == "imaging" and self._geometry.Ly == 0:
            raise ValueError(
                'layout="imaging" requires a two-dimensional geometry (Ly > 0).'
            )

        # Reset density field
        self._reset_fields()

        self._blobs = self._blob_factory.sample_blobs(
            Ly=self._geometry.Ly,
            T=self._geometry.T,
            num_blobs=self.num_blobs,
            blob_shape=self.blob_shape,
        )

        # Array-valued t_drain (drain time varying along x) must match the
        # grid; only the model knows Nx, so this cannot be checked by the
        # factory or the blob itself. Blob normalizes t_drain to a float
        # scalar or a float array at construction.
        for blob in self._blobs:
            if (
                isinstance(blob.t_drain, np.ndarray)
                and blob.t_drain.size != self._geometry.Nx
            ):
                raise ValueError(
                    f"t_drain must be a scalar or of length Nx = {self._geometry.Nx}, "
                    f"got length {blob.t_drain.size}."
                )

        if self._geometry.periodic_y and not self._one_dimensional and self._blobs:
            max_width = max(max(blob.width_p, blob.width_s) for blob in self._blobs)
            if max_width > self._geometry.Ly / 3:
                warnings.warn(
                    f"Blob width up to {max_width:.3g} is big compared to "
                    f"Ly = {self._geometry.Ly:.3g}, mirrored blobs might become apparent."
                )

        iterable = (
            tqdm(self._blobs, desc="Summing up Blobs") if self._verbose else self._blobs
        )
        for blob_index, blob in enumerate(iterable):
            self._sum_up_blobs(blob, blob_index, speed_up, truncation_error)

        dataset = self._create_xr_dataset()
        if layout == "imaging":
            dataset = to_imaging_dataset(dataset)

        if file_name is not None:
            dataset.to_netcdf(file_name)

        return dataset

    def _create_xr_dataset(self) -> xr.Dataset:
        """
        Create an xarray dataset from the density field.

        Returns
        -------
        xr.Dataset
            xarray dataset with the density field data.
        """
        if self._geometry.Ly == 0:
            # 1D output: drop the size-1 y dimension entirely, so consumers
            # get n(x, t) without having to .squeeze().
            dataset = xr.Dataset(
                data_vars=dict(
                    n=(["x", "t"], self._density[0]),
                ),
                coords=dict(
                    x=(["x"], self._geometry.x),
                    t=(["t"], self._geometry.t),
                ),
                attrs=dict(description="1D propagating blobs."),
            )
            if self._labels in {"same", "individual"}:
                dataset = dataset.assign(
                    blob_labels=(["x", "t"], self._labels_field[0])
                )
        else:
            dataset = xr.Dataset(
                data_vars=dict(
                    n=(["y", "x", "t"], self._density),
                ),
                coords=dict(
                    x=(["x"], self._geometry.x),
                    y=(["y"], self._geometry.y),
                    t=(["t"], self._geometry.t),
                ),
                attrs=dict(description="2D propagating blobs."),
            )
            if self._labels in {"same", "individual"}:
                dataset = dataset.assign(
                    blob_labels=(["y", "x", "t"], self._labels_field)
                )

        return dataset

    def _sum_up_blobs(
        self,
        blob: Blob,
        blob_index: int,
        speed_up: bool,
        truncation_error: float,
    ):
        """
        Sum up the contribution of a single blob to the density field.

        Parameters
        ----------
        blob : Blob
            Blob object.
        blob_index : int
            Position of the blob in the factory output; used to assign the
            blob label when ``labels="individual"``.
        speed_up : bool
            Flag for speeding up the code by discretizing each single blob at a smaller time window.
        truncation_error : float
            Amplitude below which the blob is truncated.
        """
        _start, _stop = self._compute_start_stop(blob, speed_up, truncation_error)
        # 1D coordinate arrays shaped to broadcast against each other as
        # (Ny, Nx, Nt) — avoids materializing three full meshgrids.
        _single_blob = blob.discretize_blob(
            x=self._geometry.x[np.newaxis, :, np.newaxis],
            y=self._geometry.y[:, np.newaxis, np.newaxis],
            t=self._geometry.t[np.newaxis, np.newaxis, _start:_stop],
            periodic_y=self._geometry.periodic_y,
            Ly=self._geometry.Ly,
            one_dimensional=self._one_dimensional,
            y0=self._geometry.y0,
        )

        self._density[:, :, _start:_stop] += _single_blob

        if self._labels == "same":
            __max_amplitudes = np.max(_single_blob, axis=(0, 1))
            __max_amplitudes[__max_amplitudes == 0] = np.inf
            self._labels_field[:, :, _start:_stop][
                _single_blob >= __max_amplitudes * self._label_border
            ] = 1
        elif self._labels == "individual":
            __max_amplitudes = np.max(_single_blob, axis=(0, 1))
            __max_amplitudes[__max_amplitudes == 0] = np.inf
            self._labels_field[:, :, _start:_stop][
                _single_blob >= __max_amplitudes * self._label_border
            ] = (blob_index + 1)

    def _compute_start_stop(self, blob: Blob, speed_up: bool, truncation_error: float):
        """
        Compute the start and stop indices for summing up the contribution of a single blob.

        Parameters
        ----------
        blob : Blob
            Blob object.
        speed_up : bool
            Flag for speeding up the code by discretizing each single blob at a smaller time window.
        truncation_error : float
            Amplitude below which the blob is truncated.

        Returns
        -------
        Tuple[int, int]
            Start and stop indices.
        """
        if not speed_up or blob.v_x == 0:
            return 0, self._geometry.t.size

        dt, t0 = self._geometry.dt, self._geometry.t[0]
        idx_x0 = (blob.t_init - t0) / dt + (self._geometry.x[0] - blob.pos_x0) / (
            blob.v_x * dt
        )
        idx_Lx = idx_x0 + self._geometry.Lx / (blob.v_x * dt)
        # Decay length of the blob along the x-axis: the blob-frame widths
        # projected onto x through the tilt angle (width_p for an untilted
        # blob, width_s at theta = pi/2).
        width_x = (
            np.abs(np.cos(blob.theta)) * blob.width_p
            + np.abs(np.sin(blob.theta)) * blob.width_s
        )
        margin = (
            -width_x * np.log(truncation_error * np.sqrt(np.pi)) / np.abs(blob.v_x * dt)
        )
        start = int(np.clip(min(idx_x0, idx_Lx) - margin, 0, self._geometry.t.size))
        stop = int(np.clip(max(idx_x0, idx_Lx) + margin, 0, self._geometry.t.size))

        return start, stop

    def _reset_fields(self):
        """Reset the density and labels fields."""
        self._density = np.zeros(
            shape=(self._geometry.Ny, self._geometry.Nx, self._geometry.t.size)
        )
        if self._labels in {"same", "individual"}:
            self._labels_field = np.zeros(
                shape=(self._geometry.Ny, self._geometry.Nx, self._geometry.t.size)
            )


def to_imaging_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """
    Convert a blobmodel output dataset to the GPI/APD imaging format.

    Downstream analysis tooling for experimental gas-puff-imaging (GPI) and
    avalanche-photodiode (APD) data standardizes on a dataset with the density
    stored as ``frames(y, x, time)`` and the grid stored as two-dimensional
    coordinate arrays ``R(y, x)``, ``Z(y, x)``. This helper converts the
    default blobmodel output ``n(y, x, t)`` (1D coordinates ``x``, ``y``,
    ``t``) into that format. It is also available directly from the model via
    ``make_realization(..., layout="imaging")``.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset as returned by `Model.make_realization` with the default
        layout. Must be two-dimensional, i.e. contain a `y` coordinate.
        A `blob_labels` variable, if present, is carried over.

    Returns
    -------
    xr.Dataset
        Dataset with the density as `frames` with dimension order
        (y, x, time), coordinates `R` and `Z` of shape (Ny, Nx) holding the
        meshgrid of the `x` and `y` coordinates, and the coordinate `time`.

    Raises
    ------
    ValueError
        If the dataset has no `y` coordinate (one-dimensional model output).
    """
    if "y" not in dataset.coords:
        raise ValueError(
            "The imaging layout requires two-dimensional model output "
            "with a y coordinate."
        )
    grid_r, grid_z = np.meshgrid(dataset.x.values, dataset.y.values)
    data_vars = {"frames": (["y", "x", "time"], dataset.n.values)}
    if "blob_labels" in dataset:
        data_vars["blob_labels"] = (["y", "x", "time"], dataset.blob_labels.values)
    return xr.Dataset(
        data_vars,
        coords={
            "R": (["y", "x"], grid_r),
            "Z": (["y", "x"], grid_z),
            "time": (["time"], dataset.t.values),
        },
        attrs=dataset.attrs,
    )
