from dataclasses import dataclass

@dataclass
class Blob():
	id: int
	amplitude: float
	width_x: float
	width_y: float
	v_x: float 
	v_y: float
	pos_x: float
	pos_y: float
	t_init: float
