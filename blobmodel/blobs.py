from dataclasses import dataclass
import numpy as np


@dataclass
class Blob:
    '''
    A single blob
    '''
    id: int
    blob_shape: str
    amplitude: float
    width_x: float
    width_y: float
    v_x: float
    v_y: float
    pos_x: float
    pos_y: float
    t_init: float
    t_drain: float

    def discretize_blob(self, x, y, t, periodic_y=False, Ly=0):
        '''
        Discretize blob on grid
        The following blob shapes are implemented:
                gauss: 2D gaussian function
                exp: one sided exponential in x and gaussian in y

                Returns
                -------
                discretized blob on 3d array with dimensions x,y and t : np.array
        '''
        return self.amplitude\
            * self.__drain(t)\
            * self.__x_shape(x, t)\
            * self.__y_shape(y, t, periodic_y, Ly)\
            * self.__blob_arrival(t)

    def __drain(self, t):
        return np.exp(-(t-self.t_init)/self.t_drain)

    def __blob_arrival(self, t):
        return np.heaviside(t-self.t_init, 1)

    def __x_shape(self, x, t):
        if self.blob_shape == 'gauss':
            return 1 / np.sqrt(np.pi) * np.exp(-((x - self.__get_x_blob_pos(t))**2/self.width_x**2))
        elif self.blob_shape == 'exp':
            return np.exp(x - self.__get_x_blob_pos(t))\
                * np.heaviside(-1.*(x - self.__get_x_blob_pos(t)), 1)
        else:
            raise NotImplementedError(
                self.__class__.__name__ + '.blob shape not implemented')

    def __y_shape(self, y, t, periodic_y, Ly):
        y_diffs = y - self.__get_y_blob_pos(t)
        if periodic_y:
            # The y_diff is centered in the simulation domain, if the difference is larger than half the domain,
            # the previous is used.
            y_diffs = y_diffs % Ly
            y_diffs[y_diffs > np.max(y)/2] -= Ly
        return 1 / np.sqrt(np.pi) * np.exp(-y_diffs**2/self.width_y**2)

    def __get_x_blob_pos(self, t):
        return self.pos_x + self.v_x*(t-self.t_init)

    def __get_y_blob_pos(self, t):
        return self.pos_y + self.v_y*(t-self.t_init)
