from .blobs import Blob
from .stochasticality import BlobFactory, DefaultBlobFactory
from .geometry import Geometry
import numpy as np
import xarray as xr
from tqdm import tqdm


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
        t_drain: float = 10,
        blob_factory: BlobFactory = DefaultBlobFactory(),
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
        periodic_y: bool, optional
            allow periodicity in y-direction

            Important: only good approximation for Ly >> blob width

        blob_shape: str, optional
            see Blob dataclass for available shapes
        t_drain: float, optional
            drain time for blobs
        blob_factory: BlobFactory, optional
            sets distributions of blob parameters
        """
        self.__geometry: Geometry = Geometry(
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
        self.t_drain: float = t_drain
        self.__blobs: list[Blob] = []
        self.__blob_factory = blob_factory

    def __str__(self) -> str:
        """string representation of Model."""
        return f"2d Blob Model with blob shape:{self.blob_shape}, num_blobs:{self.num_blobs} and t_drain:{self.t_drain}"

    def make_realization(
        self, file_name: str = None, speed_up: bool = False, error: float = 1e-10
    ) -> xr.Dataset:
        """

        Integrate Model over time and write out data as xarray dataset.

        Parameters
        ----------
        file_name: str, optional
            file name for .nc file containing data as xarray dataset
        speed_up: bool, optional
            speeding up code by discretizing each single blob at smaller time window given by
            t in (Blob.t_init, truncation_Lx*Lx/Blob.v_x + Blob.t_init)

            !!!  this is only a good approximation for blob_shape='exp' !!!

        truncation_Lx: float, optional
            number of times blob propagate through length Lx before blob is neglected
            only used if speed_up = True

        Returns
        ----------
            xarray dataset with result data
        """

        self.__blobs = self.__blob_factory.sample_blobs(
            Ly=self.__geometry.Ly,
            T=self.__geometry.T,
            num_blobs=self.num_blobs,
            blob_shape=self.blob_shape,
            t_drain=self.t_drain,
        )

        output = np.zeros(
            shape=(self.__geometry.Ny, self.__geometry.Nx, self.__geometry.t.size)
        )

        for b in tqdm(self.__blobs, desc="Summing up Blobs"):
            # speedup implemeted for exponential pulses
            # can also be used for gaussian pulses since they converge faster than exponential pulses
            if speed_up:
                start = int(b.t_init / self.__geometry.dt)
                if b.v_x == 0:
                    stop = self.__geometry.t.size
                else:
                    # ignores t_drain when calculating stop time
                    stop = start + int(
                        (-np.log(error * np.sqrt(np.pi)) + self.__geometry.Lx - b.pos_x)
                        / (b.v_x * self.__geometry.dt)
                    )
                output[:, :, start:stop] += b.discretize_blob(
                    x=self.__geometry.x_matrix[:, :, start:stop],
                    y=self.__geometry.y_matrix[:, :, start:stop],
                    t=self.__geometry.t_matrix[:, :, start:stop],
                    periodic_y=self.__geometry.periodic_y,
                    Ly=self.__geometry.Ly,
                )
            else:
                output += b.discretize_blob(
                    x=self.__geometry.x_matrix,
                    y=self.__geometry.y_matrix,
                    t=self.__geometry.t_matrix,
                    periodic_y=self.__geometry.periodic_y,
                    Ly=self.__geometry.Ly,
                )
        if self.__geometry.Ly == 0:
            ds = xr.Dataset(
                data_vars=dict(
                    n=(["y", "x", "t"], output),
                ),
                coords=dict(
                    x=(["x"], self.__geometry.x),
                    t=(["t"], self.__geometry.t),
                ),
                attrs=dict(description="2D propagating blobs."),
            )
        else:
            ds = xr.Dataset(
                data_vars=dict(
                    n=(["y", "x", "t"], output),
                ),
                coords=dict(
                    x=(["x"], self.__geometry.x),
                    y=(["y"], self.__geometry.y),
                    t=(["t"], self.__geometry.t),
                ),
                attrs=dict(description="2D propagating blobs."),
            )

        if file_name is not None:
            ds.to_netcdf(file_name)

        return ds

    def get_blobs(self) -> list[Blob]:
        """Returns blobs list.

        Note that if Model.sample_blobs has not been called, the list
        will be empty
        """
        return self.__blobs
