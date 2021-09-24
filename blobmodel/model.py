from .blobs import Blob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray as xr
from tqdm import tqdm

class Model:
    '''
    2D Model of propagating blobs 
    '''
    def __init__(self, Nx, Ny, Lx, Ly, dt, T, periodic_y=False, blob_shape='gauss', t_drain=10):
        '''
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
            !!!!
            discuss implementation of periodicity
            !!!!
        blob_shape: str, optional
            see Blob dataclass for available shapes
        t_drain: float, optional
            drain time for blobs 
        '''
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.dt = dt
        self.T = T
        self.__blobs = []
        self.__dissipation = 'None'
        self.periodic_y = periodic_y
        self.blob_shape = blob_shape
        self.t_drain = t_drain
        self.x = np.linspace(0, self.Lx, num=self.Nx)
        self.y = np.linspace(0, self.Ly, num=self.Ny)
        self.t = np.arange(0, self.T, self.dt)

    def __str__(self):
        '''
        string representation of Model 
        '''
        return f'2d Blob Model with  Nx:{self.Nx},  Ny:{self.Ny}, Lx:{self.Lx}, Ly:{self.Ly}, '\
             +f'dt:{self.dt}, T:{self.T}, y-periodicity:{self.periodic_y} and blob shape:{self.blob_shape}'

    def sample_blobs(self, 
                    num_blobs, 
                    A_dist='exp', 
                    W_dist='exp', 
                    vx_dist='gamma', 
                    vy_dist='normal',
                    A_scale = 1.0,
                    W_scale = 1.0,
                    vx_scale = 1.0,
                    vy_scale = 1.0,
                    A_shape = 1.0,
                    W_shape = 1.0,
                    vx_shape = 1.0,
                    vy_shape = 1.0,
                    A_loc = 0.0,
                    W_loc = 0.0, 
                    vx_loc = 0.0, 
                    vy_loc = 0.0,
                    A_low = 0.0, 
                    W_low = 0.0, 
                    vx_low = 0.0, 
                    vy_low = 0.0, 
                    A_high = 1.0, 
                    W_high = 1.0, 
                    vx_high = 1.0, 
                    vy_high = 1.0
                ):
        '''
        Choose appropriate distribution functions for blob parameters

        Parameters
        ----------
        num_blobs: int, number of blobs
        A_dist: str, optional
            distribution of blob amplitudes
        W_dist: str, optional
            distribution of blob widths
        vx_dist: str, optinal
            distribution of blob velocities in x-dimension
        vy_dist: str, optinal
            distribution of blob velocities in y-dimension
        *_scale: float, optional
            scale parameter for exp, gamma, normal and rayleigh distributions
        *_shape: float, optional
            shape paremeter for gamma distribution
        *_loc:float, optional
            location parameter for normal distribution
        *_low: float, optional
            lower boundary for uniform distribution
        *_high: float, optional
            upper boundary for uniform distribution

        Note that * refers to either A, W, vx or vy
        '''

        def choose_distribution(dist_type, scale, shape, loc, low, high):
            '''
            The following distributions are implemented:
                exp: exponential distribution with scale parameter
                gamma: gamma distribution with shape and scale parameter
                normal: normal distribution with loc and scale parameter
                uniform: uniorm distribution with low and high parameter
                ray: rayleight distribution with scale parameter
                deg: array on ones 
                zeros: array of zeros
            '''
            if dist_type == 'exp':
                return np.random.exponential(scale=scale, size=num_blobs)
            elif dist_type == 'gamma':
                return np.random.gamma(shape=shape, scale=scale, size=num_blobs)
            elif dist_type == 'normal':
                return np.random.normal(loc=loc, scale=scale, size=num_blobs)
            elif dist_type == 'uniform':
                return np.random.uniform(low=low, high=high, size=num_blobs)
            elif dist_type == 'ray':
                return np.random.rayleigh(scale=scale, size=num_blobs)
            elif dist_type == 'deg':
                return np.ones(num_blobs)
            elif dist_type == 'zeros':
                return np.zeros(num_blobs)
            else:
                raise NotImplementedError(self.__class__.__name__ + '.distribution function not implemented')

        __amp = choose_distribution(A_dist, A_scale, A_shape, A_loc, A_low, A_high)
        __width = choose_distribution(W_dist, W_scale, W_shape, W_loc, W_low, W_high)
        __vx = choose_distribution(vx_dist, vx_scale, vx_shape, vx_loc, vx_low, vx_high)
        __vy = choose_distribution(vy_dist, vy_scale, vy_shape, vy_loc, vy_low, vy_high)

        # the following parameters are fixed
        __posx = np.zeros(num_blobs)
        __posy = np.random.uniform(low=0.0, high=self.Ly, size=num_blobs)
        __t_init = np.random.uniform(low=0, high=self.T, size=num_blobs)

        # sort blobs by __t_init
        __t_init = np.sort(__t_init)

        for i in range(num_blobs):
            self.__blobs.append(Blob(id = i,
                        blob_shape=self.blob_shape,
                        amplitude=__amp[i],
                        width_x=__width[i],
                        width_y = __width[i],
                        v_x=__vx[i],
                        v_y=__vy[i],
                        pos_x=__posx[i],
                        pos_y = __posy[i],
                        t_init=__t_init[i],
                        t_drain= self.t_drain
                        ))

    def show_model(self, interval=100, save = False, gif_name = '2d_blobs.gif', fps = 10):
        '''
        show animation of Model

        Parameters
        ----------
        interval: int, optional
            time interval between frames in ms
        save: bool, optional
            if True save animation as gif
        gif_name: str, optional
            set name for gif
        fps: int, optional
            set fps for gif
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)

        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '5%', '5%')

        frames = []
        
        __xx, __yy = np.meshgrid(self.x, self.y)

        for t in tqdm(self.t,desc="Creating frames for animation"):
            curVals = np.zeros(shape=(self.Ny, self.Nx))
            for b in self.__blobs:
                curVals  += b.discretize_blob(x=__xx, y=__yy, t=t)
                if(self.periodic_y):
                    curVals  += b.discretize_blob(x=__xx, y=__yy-self.Ly, t=t)
                    curVals  += b.discretize_blob(x=__xx, y=__yy+self.Ly, t=t)
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
        if save:
            ani.save(gif_name, writer='ffmpeg', fps=fps)
        plt.show()

    def integrate(self,file_name='2d_blobs.nc', speed_up = False, truncation_Lx = 3):
        '''
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
        '''

        __xx, __yy, __tt = np.meshgrid(self.x, self.y, self.t)
        output =  np.zeros(shape=(self.Ny, self.Nx, self.t.size))
        
        for b in tqdm(self.__blobs,desc="Summing up Blobs"):
            if speed_up:
                start = int(b.t_init/self.dt)
                stop = int(truncation_Lx*self.Lx/(b.v_x*self.dt)) + start
                output[:,:,start:stop] += b.discretize_blob(x=__xx[:,:,start:stop], y=__yy[:,:,start:stop], t=__tt[:,:,start:stop])
                if(self.periodic_y):
                    output[:,:,start:stop] += b.discretize_blob(x=__xx[:,:,start:stop], y=__yy[:,:,start:stop]-self.Ly, t=__tt[:,:,start:stop])
                    output[:,:,start:stop] += b.discretize_blob(x=__xx[:,:,start:stop], y=__yy[:,:,start:stop]+self.Ly, t=__tt[:,:,start:stop])
            else:
                output += b.discretize_blob(x=__xx, y=__yy, t=__tt)
                if(self.periodic_y):
                    output  += b.discretize_blob(x=__xx, y=__yy-self.Ly, t=__tt)
                    output  += b.discretize_blob(x=__xx, y=__yy+self.Ly, t=__tt)
        ds = xr.Dataset(
            data_vars=dict(
            n = (['y','x', 't'], output),
            ),
            coords=dict(
                x = (['x'], np.linspace(0, self.Lx, num=self.Nx)), 
                y = (['y'], np.linspace(0, self.Ly, num=self.Ny)),
                t = (['t'], np.arange(0, self.T, self.dt)),
            ),
            attrs=dict(description="2D propagating blobs."),
        )

        ds.to_netcdf(file_name)