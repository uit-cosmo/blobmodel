from .blobs import Blob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Model:
    def __init__(self, Nx, Ny, Lx, Ly, dt, T):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.dt = dt
        self.T = T
        self.__blobs = []
        self.__dissipation = 'None'
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
            curVals = np.zeros(shape=(self.Nx, self.Ny))
            for b in self.__blobs:
                curVals  += b.discretize_blob(x=self.x, y=self.y, t=t)
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

    def integrate(self):
        raise NotImplementedError(self.__class__.__name__ + '.integrate')
