# 2d_propagating_blobs
Two dimensional model of advecting and dissipating blobs.

The code has been deveoped originally to model profiles in the scrape-off layer of fusion experiments but it can be used to model any 2d system consisting of advecting pulses. An example is shown below:
![Density evolution](example_gifs/2d_blobs.gif ) 

## Installation

Dev install:
```
git clone https://github.com/gregordecristoforo/2d_propagating_blobs.git
cd 2d_propagating_blobs
pip install -e .
```


## Usage
Create the grid on which the blobs are discretized with using the `Model` class. The blobs are then seeded by the `sample_blobs` method. The blob parameters are sampled from distribution functions that are specified by the input parameters for `sample_blobs`. The `integrate()` method computes the output as an xarray dataset and writhes it out in a `netcdf` file. A simple example is shown below:

```Python
from blobmodel import Model

bm = Model(Nx=200, Ny=100, Lx=10, Ly=10, dt=0.1, T=20, blob_shape='gauss')

bm.sample_blobs(num_blobs=100)

bm.integrate()
```
Alternatively, the data can be shown as an animation using the `show_model` method:
```Python
bm.show_model(interval=100)
```

## Input parameters
### `Model()`
- `Nx`: int, grid points in x
- `Ny`: int, grid points in y
- `Lx`: float, length of grid in x
- `Ly`: float, length of grid in y
- `dt`: float, time step 
- `T`: float, time length 
- `periodic_y`: bool, optional,
            allow periodicity in y-direction 
- `blob_shape`: str, optional,
            switch between `gauss` and `exp` blob
- `t_drain`: float, optional,
            drain time for blobs 

### `sample_blobs()`
- `num_blobs`: int, number of blobs
- `A_dist`: str, optional,
            distribution of blob amplitudes
- `W_dist`: str, optional,
            distribution of blob widths
- `vx_dist`: str, optinal,
            distribution of blob velocities in x-dimension
- `vy_dist`: str, optinal,
            distribution of blob velocities in y-dimension
- `*_scale`: float, optional,
            scale parameter for exp, gamma, normal and rayleigh distributions
- `*_shape`: float, optional,
            shape paremeter for gamma distribution
- `*_loc`:float, optional,
            location parameter for normal distribution
- `*_low`: float, optional,
            lower boundary for uniform distribution
- `*_high`: float, optional,
            upper boundary for uniform distribution
            
Note that `*` refers to either `A`, `W`, `vx` or `vy`

The following distributions are implemented:

- `exp`: exponential distribution with scale parameter
- `gamma`: gamma distribution with shape and scale parameter
- `normal`: normal distribution with loc and scale parameter
- `uniform`: uniorm distribution with low and high parameter
- `ray`: rayleight distribution with scale parameter
- `deg`: array on ones 
- `zeros`: array of zeros
                
### `integrate()`
- `file_name`: str, optional, 
            file name for .nc file containing data as xarray dataset
- `speed_up`: bool, optional,
            speeding up code by discretizing each single blob at smaller time window given by
            `t` in (`Blob.t_init`, `truncation_Lx*Lx/Blob.v_x + Blob.t_init`)

            !!!  this is only a good approximation for blob_shape='exp' !!!
- `truncation_Lx`: float, optional,
            number of times blob propagate through length Lx before blob is neglected,
            only used if speed_up = True
            
### `show_model()`
- `interval`: int, optional,
            time interval between frames in ms
- `save`: bool, optional,
            if True save animation as gif
- `gif_name`: str, optional,
            set name for gif
- `fps`: int, optional,
            set fps for gif

## Contact
If you have questions, suggestions or other comments you can contact me under gregor.decristoforo@uit.no

