from .blobs import Blob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray as xr

class Model:
    def __init__(self, Nx, Ny, Lx, Ly, dt, T, periodic_y=True):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.dt = dt
        self.T = T
        self.__blobs = []
        self.__dissipation = 'None'
        self.periodic_y = periodic_y
        self.x = np.linspace(0, self.Lx, num=self.Nx)
        self.y = np.linspace(0, self.Ly, num=self.Ny).reshape(-1, 1)
        self.t = np.arange(0, self.T, self.dt)

    def sample_blobs(self, 
                    num_blobs, 
                    A_dist='exp', 
                    W_dist='exp', 
                    vx_dist='deg', 
                    vy_dist='deg' 
                ):

        __amp = np.random.exponential(scale=1.0, size=num_blobs)
        __width = np.random.exponential(scale=1.0, size=num_blobs)
        __posx = np.zeros(num_blobs)
        __posy = np.random.uniform(low=0.0, high=self.Ly, size=num_blobs)
        __vx = np.random.gamma(shape=1.0, scale=1.0, size=num_blobs)
        __vy = np.random.normal(loc=0.0, scale=1.0, size=num_blobs)
        __t_init = np.random.uniform(low=0, high=self.T, size=num_blobs)

        # sort blobs by __t_init
        __t_init = np.sort(__t_init)

        for i in range(num_blobs):
            self.__blobs.append(Blob(id = i,
                        blob_shape='gauss',
                        amplitude=__amp[i],
                        width_x=__width[i],
                        width_y = __width[i],
                        v_x=__vx[i],
                        v_y=__vy[i],
                        pos_x=__posx[i],
                        pos_y = __posy[i],
                        t_init=__t_init[i]
                        ))


    def set_dissipation(self):
        raise NotImplementedError(self.__class__.__name__ + '.set_dissipation')

    def show_model(self, interval=100):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '5%', '5%')

        frames = []
        for t in self.t:
            curVals = np.zeros(shape=(self.Ny, self.Nx))
            for b in self.__blobs:
                curVals  += b.discretize_blob(x=self.x, y=self.y, t=t)
                if(self.periodic_y):
                    curVals  += b.discretize_blob(x=self.x, y=self.y-self.Ly, t=t)
                    curVals  += b.discretize_blob(x=self.x, y=self.y+self.Ly, t=t)
            frames.append(curVals)

        cv0 = frames[0]
        im = ax.imshow(cv0, origin='lower')
        cb = fig.colorbar(im, cax=cax)
        tx = ax.set_title('t = 0')

        def animate(i):
            arr = frames[i]
            vmax = np.max(arr)
            vmin = np.min(arr)
            im.set_data(arr)
            im.set_clim(vmin, vmax)
            tx.set_text(f't = {i*self.dt:.2f}')
        
        ani = FuncAnimation(fig, animate, frames=self.t.size,interval=interval)
        plt.show()

    def set_diagnostics(self):
        raise NotImplementedError(self.__class__.__name__ + '.set_diagnostics')

    def integrate(self,file_name='2d_blobs.nc'):
        __xx, __yy, __tt = np.meshgrid(self.y, self.x, self.t)
        output =  np.zeros(shape=(self.Nx, self.Ny, self.t.size))

        for b in self.__blobs:
            output += b.discretize_blob(x=__xx, y=__yy, t=__tt)

        ds = xr.Dataset(
            data_vars=dict(
            n = (['x', 'y', 't'], output),
            ),
            coords=dict(
                x = (['x'], np.linspace(0, self.Lx, num=self.Nx)), 
                y = (['y'], np.linspace(0, self.Ly, num=self.Ny)),
                t = (['t'], np.arange(0, self.T, self.dt)),
            ),
            attrs=dict(description="2D porpagating blobs."),
        )

        ds.to_netcdf(file_name)