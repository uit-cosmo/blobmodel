"""This module defines a 2D model of propagating blobs."""

import numpy as np
import xarray as xr
from tqdm import tqdm
from typing import List, Union
from .blobs import Blob
from .stochasticality import BlobFactory, DefaultBlobFactory
from .geometry import Geometry
from nptyping import NDArray
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
        Nx: int = 100,
        Ny: int = 100,
        Lx: float = 10,
        Ly: float = 10,
        dt: float = 0.1,
        T: float = 10,
        periodic_y: bool = False,
        blob_shape: AbstractBlobShape = BlobShapeImpl(),
        num_blobs: int = 1000,
        t_drain: Union[float, NDArray, int] = 10,
        blob_factory: BlobFactory = DefaultBlobFactory(),
        t_init: float = 0,
        labels: str = "off",
        label_border: float = 0.75,
        one_dimensional: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the 2D Model of propagating blobs.

        Parameters
        ----------
        Nx : int, optional
            Number of grid points in x.
        Ny : int, optional
            Number of grid points in y.
        Lx : float, optional
            Length of the domain in x.
        Ly : float, optional
            Length of the domain in y.
        dt : float, optional
            Time step.
        T : float, optional
            Time length.
        periodic_y : bool, optional
            Allow periodicity in the y-direction.
            Important: only good approximation for Ly >> blob width
        blob_shape : AbstractBlobShape, optional
            Shape of the blobs. Can be an instance of AbstractBlobShape. Defaults to gaussian.
        num_blobs : int, optional
            Number of blobs.
        t_drain : float or array-like, optional
            Drain time for the blobs. Can be a single float value or an array-like
            of length Nx.
        blob_factory : BlobFactory, optional
            BlobFactory instance for setting blob parameter distributions.
        t_init : float, optional
            Initial time for simulation, default 0.
        labels : str, optional
            Blob label setting. Possible values: "off", "same", "individual".
            "off": no blob labels returned
            "same": regions where blobs are present are set to label 1
            "individual": different blobs return individual labels
            Used for creating training data for supervised machine learning algorithms
        label_border : float, optional
            Defines region of blob as region where density >= label_border * amplitude of Blob
            Only used if labels = "same" or "individual"
        one_dimensional : bool, optional
            If True, the perpendicular shape of the blobs will be discarded.
            Parameters for the y-component (Ny and Ly) will be overwritten to Ny=1, Ly=0.
            The perpendicular blob shape will be set to 1.
        verbose : bool, optional
            If True, print a loading bar.

        Raises
        ------
        AssertionError
            If t_drain is not a single value or an array-like of length Nx.

        Warns
        -----
        UserWarning
            If the model is one-dimensional and Ny and Ly are not appropriate.

        UserWarning
            If the model is one-dimensional and the blob factory is not one-dimensional.
        """
        assert isinstance(blob_shape, AbstractBlobShape)
        assert isinstance(blob_factory, BlobFactory)
        self._one_dimensional = one_dimensional
        if self._one_dimensional:
            if Ny != 1 or Ly != 0:
                warnings.warn(
                    "Overwritting Ny=1 and Ly=0 to allow one dimensional model"
                )
                Ny = 1
                Ly = 0
            if not blob_factory.is_one_dimensional():
                warnings.warn(
                    "Using a one dimensional model with a blob factory that is not one-dimensional. Are you sure you know what you are doing?"
                )

        self._geometry: Geometry = Geometry(
            Nx=Nx,
            Ny=Ny,
            Lx=Lx,
            Ly=Ly,
            dt=dt,
            T=T,
            t_init=t_init,
            periodic_y=periodic_y,
        )
        self.blob_shape = blob_shape
        self.num_blobs: int = num_blobs
        self.t_drain: Union[float, NDArray] = t_drain

        assert (
            isinstance(t_drain, (int, float)) or len(t_drain) == Nx
        ), "t_drain must be of either length 1 or Nx"

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
        return (
            f"2d Blob Model with"
            + f" num_blobs:{self.num_blobs} and t_drain:{self.t_drain}"
        )

    def get_blobs(self) -> List[Blob]:
        """
        Return the list of blobs.

        Returns
        -------
        List[Blob]
            List of Blob objects.

        Notes
        -----
        - Note that if Model.sample_blobs has not been called, the list will be empty

        """
        return self._blobs

    def make_realization(
        self,
        file_name: Union[str, None] = None,
        speed_up: bool = False,
        error: float = 1e-10,
    ) -> xr.Dataset:
        """
        Integrate the Model over time and write out data as an xarray dataset.

        Parameters
        ----------
        file_name : str, optional
            File name for the .nc file containing data as an xarray dataset.
        speed_up : bool, optional
            Flag for speeding up the code by discretizing each single blob at smaller time window
            when blob values fall under given error value the blob gets discarded
        error : float, optional
            Numerical error at x = Lx when the blob gets truncated.

        Returns
        -------
        xr.Dataset
            xarray dataset with the data resulting from a realization of the process described by the model
            and evaluated in a three-dimensional grid with dimensions:
            - x: Horizontal coordinate
            - y: Vertical coordinate
            - t: Time coordinate
            The resulting blob density, evaluated in the grid, is given by the `DataArray`, `n`. In case that
            the model is one-dimensional, the vertical coordinate `y` will be of length 1.


        Notes
        -----
        - speed_up is only a good approximation for blob_shape="exp"
        """

        # Reset density field
        self._reset_fields()

        self._blobs = self._blob_factory.sample_blobs(
            Ly=self._geometry.Ly,
            T=self._geometry.T,
            num_blobs=self.num_blobs,
            blob_shape=self.blob_shape,
            t_drain=self.t_drain,
        )

        iterable = (
            tqdm(self._blobs, desc="Summing up Blobs") if self._verbose else self._blobs
        )
        for blob in iterable:
            self._sum_up_blobs(blob, speed_up, error)

        dataset = self._create_xr_dataset()

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
            dataset = xr.Dataset(
                data_vars=dict(
                    n=(["y", "x", "t"], self._density),
                ),
                coords=dict(
                    x=(["x"], self._geometry.x),
                    t=(["t"], self._geometry.t),
                ),
                attrs=dict(description="2D propagating blobs."),
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
            dataset = dataset.assign(blob_labels=(["y", "x", "t"], self._labels_field))

        return dataset

    def _sum_up_blobs(
        self,
        blob: Blob,
        speed_up: bool,
        error: float,
    ):
        """
        Sum up the contribution of a single blob to the density field.

        Parameters
        ----------
        blob : Blob
            Blob object.
        speed_up : bool
            Flag for speeding up the code by discretizing each single blob at a smaller time window.
        error : float
            Numerical error when the blob gets truncated.
        """
        _start, _stop = self._compute_start_stop(blob, speed_up, error)
        _single_blob = blob.discretize_blob(
            x=self._geometry.x_matrix[:, :, _start:_stop],
            y=self._geometry.y_matrix[:, :, _start:_stop],
            t=self._geometry.t_matrix[:, :, _start:_stop],
            periodic_y=self._geometry.periodic_y,
            Ly=self._geometry.Ly,
            one_dimensional=self._one_dimensional,
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
            ] = (blob.blob_id + 1)

    def _compute_start_stop(self, blob: Blob, speed_up: bool, error: float):
        """
        Compute the start and stop indices for summing up the contribution of a single blob.

        Parameters
        ----------
        blob : Blob
            Blob object.
        speed_up : bool
            Flag for speeding up the code by discretizing each single blob at a smaller time window.
        error : float
            Numerical error when the blob gets truncated.

        Returns
        -------
        Tuple[int, int]
            Start and stop indices.
        """
        if not speed_up or blob.v_x == 0:
            return 0, self._geometry.t.size
        start = np.maximum(
            0,
            int(
                (
                    np.abs(blob.t_init * blob.v_x)
                    + blob.width_prop * np.log(error * np.sqrt(np.pi))
                    + blob.pos_x
                    - self._geometry.t[0]
                )
                / np.abs(self._geometry.dt * blob.v_x)
            ),
        )
        # ignores t_drain when calculating stop time
        stop = np.minimum(
            self._geometry.t.size,
            int(
                (
                    np.abs(blob.t_init * blob.v_x)
                    - blob.width_prop * np.log(error * np.sqrt(np.pi))
                    + self._geometry.Lx
                    - blob.pos_x
                )
                / np.abs(blob.v_x * self._geometry.dt)
            ),
        )
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
