from dataclasses import dataclass
import numpy as np

@dataclass
class Blob():
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

	def discretize_blob(self,x, y, t):
		if(self.blob_shape == 'gauss'):
			return self.amplitude * np.exp(-((x-self.pos_x -self.v_x*(t-self.t_init))**2/(2*self.width_x**2)))\
                *np.exp(-((y-self.pos_y -self.v_y*(t-self.t_init))**2/(2*self.width_y**2)))\
                *np.heaviside(t-self.t_init, 1)
		else:
			raise NotImplementedError(self.__class__.__name__ + '.Non gaussian blobs not implemented')
