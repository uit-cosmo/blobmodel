from dataclasses import dataclass
import numpy as np

@dataclass
class Blob():
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

	def discretize_blob(self,x, y, t):
		''' 
		Discretize blob on grid
		The following blob shapes are implemented:
			gauss: 2D gaussian function
			exp: one sided exponential in x and gaussian in y
		'''
		if(self.blob_shape == 'gauss'):
			return self.amplitude * np.exp(-(t-self.t_init)/self.t_drain)\
				*np.exp(-((x-self.pos_x -self.v_x*(t-self.t_init))**2/(2*self.width_x**2)))\
                *np.exp(-((y-self.pos_y -self.v_y*(t-self.t_init))**2/(2*self.width_y**2)))\
                *np.heaviside(t-self.t_init, 1)
		elif(self.blob_shape == 'exp'):
			return self.amplitude * np.exp(-(t-self.t_init)/self.t_drain)\
				* np.exp( x - self.pos_x -self.v_x*(t-self.t_init))\
                *np.exp(-((y-self.pos_y -self.v_y*(t-self.t_init))**2/(2*self.width_y**2)))\
                *np.heaviside(t-self.t_init, 1)*np.heaviside(-1.*(x - self.pos_x -self.v_x*(t-self.t_init)), 1)
		else:
			raise NotImplementedError(self.__class__.__name__ + '.blob shape not implemented')
