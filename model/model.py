from .blobs import Blob
import numpy as np

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

    def sample_blobs(self, 
                    num_blobs, 
                    shape = 'gauss', 
                    A_dist='exp', 
                    W_dist='exp', 
                    vx_dist='deg', 
                    vy_dist='deg' 
                ):

        __amp = np.random.exponential(scale=1.0, size=num_blobs)
        __width = np.random.exponential(scale=1.0, size=num_blobs)
        __posx = np.zeros(num_blobs)
        __posy = np.random.uniform(low=0.0, high=self.Ly, size=num_blobs)
        __vx = np.ones(num_blobs)
        __vy = np.ones(num_blobs)
        __t_tinit = np.random.uniform(low=0, high=self.T, size=num_blobs)

        # sort blobs by __t_init
        __t_tinit = np.sort(__t_tinit)

        for i in range(num_blobs):
            self.__blobs.append(Blob(id = i,
                        amplitude=__amp[i],
                        width_x=__width[i],
                        width_y = __width[i],
                        v_x=__vx[i],
                        v_y=__vy[i],
                        pos_x=__posx[i],
                        pos_y = __posy[i],
                        t_init=__t_tinit[i]
                        ))


    def set_dissipation(self):
        raise NotImplementedError(self.__class__.__name__ + '.sample_blobs')

    def show_model(self):
        raise NotImplementedError(self.__class__.__name__ + '.show_model')

    def set_diagnostics(self):
        raise NotImplementedError(self.__class__.__name__ + '.set_diagnostics')

    def integrate(self):
        raise NotImplementedError(self.__class__.__name__ + '.integrate')
