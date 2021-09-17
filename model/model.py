from .blobs import Blob

class Model:
    def __init__(self, Nx, Ny, Lx, Ly, dt, T):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.dt = dt
        self.T = T

    def sample_blobs(self):
        raise NotImplementedError(self.__class__.__name__ + '.sample_blobs')

    def show_model(self):
        raise NotImplementedError(self.__class__.__name__ + '.show_model')

    def set_diagnostics(self):
        raise NotImplementedError(self.__class__.__name__ + '.set_diagnostics')

    def integrate(self):
        raise NotImplementedError(self.__class__.__name__ + '.integrate')
