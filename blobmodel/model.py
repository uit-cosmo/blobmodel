from .blobs import Blob
from .stochasticality import BlobFactory, DefaultBlobFactory
import numpy as np
import xarray as xr
from tqdm import tqdm
from nptyping import NDArray, Float
from typing import Any


class Model:
    """
    2D Model of propagating blobs 
    """

    def __init__(
        self,
        Nx: int,
        Ny: int,
        Lx: float,
        Ly: float,
        dt: float,
        T: float,
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
        blob_shape: str, optional
            see Blob dataclass for available shapes
        t_drain: float, optional
            drain time for blobs 
        """
        self.Nx: int = Nx
        self.Ny: int = Ny
        self.Lx: float = Lx
        self.Ly: float = Ly
        self.dt: float = dt
        self.T: float = T
        self.__blobs: list[Blob] = []
        self.periodic_y: bool = periodic_y
        self.blob_shape: str = blob_shape
        self.num_blobs: int = num_blobs
        self.t_drain: float = t_drain
        self.x: NDArray[Any, Float[64]] = np.arange(0, self.Lx, self.Lx / self.Nx)
        # For Ly == 0, model reduces to 1 spatial dimension
        if self.Ly == 0:
            self.y: NDArray[Any, Float[64]] = 0
        else:
            self.y = np.arange(0, self.Ly, self.Ly / self.Ny)
        self.t: NDArray[Any, Float[64]] = np.arange(0, self.T, self.dt)

        self.blob_factory = blob_factory
        self.__blobs = self.blob_factory.sample_blobs(
            Ly=self.Ly,
            T=self.T,
            num_blobs=self.num_blobs,
            blob_shape=self.blob_shape,
            t_drain=self.t_drain,
        )

    def __str__(self) -> str:
        """
        string representation of Model 
        """
        return (
            f"2d Blob Model with  Nx:{self.Nx},  Ny:{self.Ny}, Lx:{self.Lx}, Ly:{self.Ly}, "
            + f"dt:{self.dt}, T:{self.T}, y-periodicity:{self.periodic_y} and blob shape:{self.blob_shape}"
        )

    def integrate(
        self, file_name: str = None, speed_up: bool = False, truncation_Lx: float = 3
    ) -> xr.Dataset:
        """
        Integrate Model over time and write out data as xarray dataset

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

        __xx, __yy, __tt = np.meshgrid(self.x, self.y, self.t)
        output = np.zeros(shape=(self.Ny, self.Nx, self.t.size))

        for b in tqdm(self.__blobs, desc="Summing up Blobs"):
            if speed_up:
                start = int(b.t_init / self.dt)
                stop = int(truncation_Lx * self.Lx / (b.v_x * self.dt)) + start
                output[:, :, start:stop] += b.discretize_blob(
                    x=__xx[:, :, start:stop],
                    y=__yy[:, :, start:stop],
                    t=__tt[:, :, start:stop],
                    periodic_y=self.periodic_y,
                    Ly=self.Ly,
                )
            else:
                output += b.discretize_blob(
                    x=__xx, y=__yy, t=__tt, periodic_y=self.periodic_y, Ly=self.Ly
                )
        if self.Ly == 0:
            ds = xr.Dataset(
                data_vars=dict(n=(["y", "x", "t"], output),),
                coords=dict(x=(["x"], self.x), t=(["t"], self.t),),
                attrs=dict(description="2D propagating blobs."),
            )
        else:
            ds = xr.Dataset(
                data_vars=dict(n=(["y", "x", "t"], output),),
                coords=dict(x=(["x"], self.x), y=(["y"], self.y), t=(["t"], self.t),),
                attrs=dict(description="2D propagating blobs."),
            )

        if file_name is not None:
            ds.to_netcdf(file_name)

        return ds

    def get_blobs(self) -> list[Blob]:
        """
        Returns blobs list. Note that if Model.sample_blobs has not been called, the list will be empty
        """
        return self.__blobs
