# 2d_propagating_blobs
Two dimensional model of advecting and dissipating blobs.

The code has been deveoped originally to model profiles in the scrape-off layer of fusion experiments but it can be used to model any 2d system consisting of advecting pulses. An example is shown below:

![Density evolution](example_gifs/2d_blobs.gif ) 


<!---## Installation
## Installation

Dev install:
```
git clone https://github.com/gregordecristoforo/xblobs.git
cd xblobs
pip install -e .
```



## Usage
```Python
from blobmodel import Model, Blob

tmp = Model(Nx=200, Ny=100, Lx=10, Ly=10, dt=0.1, T=20, blob_shape='gauss')

tmp.sample_blobs(num_blobs=100, vy_scale=0.33)

#tmp.integrate()

tmp.show_model(interval=100, save = True)
```
## Contact
If you have questions, suggestions or other comments you can contact me under gregor.decristoforo@uit.no--->

