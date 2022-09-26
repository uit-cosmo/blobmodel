import numpy as np
import xarray as xr
from tqdm import tqdm
from typing import List, Union
from .blobs import Blob
from .stochasticality import BlobFactory, DefaultBlobFactory
from .geometry import Geometry
from nptyping import NDArray


class Model:
    """2D Model of propagating blobs."""

    def __init__(
        self,
        Nx: int = 100,
        Ny: int = 100,
        Lx: float = 10,
        Ly: float = 10,
        dt: float = 0.1,
        T: float = 10,
        periodic_y: bool = False,
        blob_shape: str = "gauss",
        num_blobs: int = 1000,
        t_drain: Union[float, NDArray] = 10,
        blob_factory: BlobFactory = DefaultBlobFactory(),
        labels: str = "off",
        label_border: float = 0.75,
    ) -> None:
        """
        Attributes
        ----------
        Nx: int, grid points in x
        Ny: int, grid points in y
        Lx: float, length of grid in x
        Ly: float, length of grid in y
        dt: float, time step
        T: float, time length
        periodic_y: bool, optional
            allow periodicity in y-direction

            Important: only good approximation for Ly >> blob width
        num_blobs:
            number of blobs
        blob_shape: str, optional
            see Blob dataclass for available shapes
        t_drain: float or array of length Nx, optional
            drain time for blobs
        blob_factory: BlobFactory, optional
            sets distributions of blob parameters
        labels: str, optional
            "off": no blob labels returned
            "same": regions where blobs are present are set to label 1
            "individual": different blobs return individual labels
            used for creating training data for supervised machine learning algorithms
        label_border: float, optional
            defines region of blob as region where density >= label_border * amplitude of Blob
            only used if labels = "same" or "individual"
        """
        self._geometry: Geometry = Geometry(
            Nx=Nx,
            Ny=Ny,
            Lx=Lx,
            Ly=Ly,
            dt=dt,
            T=T,
            periodic_y=periodic_y,
        )
        self.blob_shape: str = blob_shape
        self.num_blobs: int = num_blobs
        self.t_drain: Union[float, NDArray] = t_drain

        assert (
            isinstance(t_drain, (int, float)) or len(t_drain) == Nx
        ), "t_drain must be of either length 1 or Nx"

        self._blobs: list[Blob] = []
        self._blob_factory = blob_factory
        self._labels = labels
        self._label_border = label_border
        self._reset_fields()

    def __str__(self) -> str:
        """string representation of Model."""
        return (
            f"2d Blob Model with blob shape:{self.blob_shape},"
            + f" num_blobs:{self.num_blobs} and t_drain:{self.t_drain}"
        )

    def get_blobs(self) -> List[Blob]:
        """Returns blobs list.

        Note that if Model.sample_blobs has not been called, the list
        will be empty
        """
        return self._blobs

    def make_realization(
        self,
        file_name: str = None,
        speed_up: bool = False,
        error: float = 1e-10,
    ) -> xr.Dataset:
        """Integrate Model over time and write out data as xarray dataset.

        Parameters
        ----------
        file_name: str, optional
            file name for .nc file containing data as xarray dataset
        speed_up: bool, optional
            speeding up code by discretizing each single blob at smaller time window
            when blob values fall under given error value the blob gets discarded
            !!!  this is only a good approximation for blob_shape='exp' !!!

        error: float, optional
            numerical error at x = Lx when blob gets truncated
            only used if speed_up = True

        Returns
        ----------
            xarray dataset with result data
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

        for blob in tqdm(self._blobs, desc="Summing up Blobs"):
            self._sum_up_blobs(blob, speed_up, error)

        dataset = self._create_xr_dataset()

        if file_name is not None:
            dataset.to_netcdf(file_name)

        return dataset

    def _create_xr_dataset(self) -> xr.Dataset:
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
        _start, _stop = self._compute_start_stop(blob, speed_up, error)
        _single_blob = blob.discretize_blob(
            x=self._geometry.x_matrix[:, :, _start:_stop],
            y=self._geometry.y_matrix[:, :, _start:_stop],
            t=self._geometry.t_matrix[:, :, _start:_stop],
            periodic_y=self._geometry.periodic_y,
            Ly=self._geometry.Ly,
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
        if speed_up:
            _start = int(blob.t_init / self._geometry.dt)
            if blob.v_x == 0:
                _stop = self._geometry.t.size
            else:
                # ignores t_drain when calculating stop time
                _stop = np.minimum(
                    self._geometry.t.size,
                    _start
                    + int(
                        (
                            -np.log(error * np.sqrt(np.pi))
                            + self._geometry.Lx
                            - blob.pos_x
                        )
                        / (blob.v_x * self._geometry.dt)
                    ),
                )
        else:
            _start = 0
            _stop = self._geometry.t.size

        return _start, _stop

    def _reset_fields(self):
        self._density = np.zeros(
            shape=(self._geometry.Ny, self._geometry.Nx, self._geometry.t.size)
        )
        if self._labels in {"same", "individual"}:
            self._labels_field = np.zeros(
                shape=(self._geometry.Ny, self._geometry.Nx, self._geometry.t.size)
            )
