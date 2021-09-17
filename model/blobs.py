import numpy as np
from .model import Model

class Blob(Model):
	def __init__(self, id, amplitude, blob_shape, width_x, width_y, v_x, v_y, pos_x, pos_y, t_init ):
		seld,id = id
		self.amplitude = amplitude
		self.blob_shape = blob_shape
		self.width_x = width_x
		self.width_y = width_y
		self.v_x = v_x
		self.v_y = v_y
		self.pos_x = pos_x
		self.pos_y = pos_y
		self.t_init = t_init	

