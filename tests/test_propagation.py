import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

#grid
X = np.linspace(start=0.0,stop=10.0, num = 100)
Y = np.linspace(start=0.0,stop=10.0, num = 100)
T = np.linspace(start=0.0,stop=10.0, num = 50)


# Blob parameters
amp = 1.0
delta_x = 0.50
delta_y = 1.0
x0 = 1.0
y0 = 1.0
v_x = 1
v_y = 1
t_init = 1

xx, yy, tt = np.meshgrid(X, Y, T)

n = amp * np.exp(-((xx-x0 -v_x*tt)**2/(2*delta_x**2)))*np.exp(-((yy-x0 -v_y*tt)**2/(2*delta_y**2)))

def f_exp(x,y,t):
    return amp * np.exp(-((x-x0 -v_x*t)**2/(2*delta_x**2)))\
        *np.exp(-((y-x0 -v_y*t)**2/(2*delta_y**2)))* np.heaviside(t-t_init, 1)

fig = plt.figure()
ax = fig.add_subplot(111)

div = make_axes_locatable(ax)
cax = div.append_axes('right', '5%', '5%')

x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100).reshape(-1, 1)
T = np.linspace(0, 10, 100)

frames = []
for t in T:
    curVals  = f_exp(x,y,t)
    frames.append(curVals)


cv0 = frames[0]
im = ax.imshow(cv0, origin='lower') # Here make an AxesImage rather than contour
cb = fig.colorbar(im, cax=cax)

def animate(i):
    arr = frames[i]
    vmax     = np.max(arr)
    vmin     = np.min(arr)
    im.set_data(arr)
    im.set_clim(vmin, vmax)
    # In this version you don't have to do anything to the colorbar,
    # it updates itself when the mappable it watches (im) changes

ani = FuncAnimation(fig, animate, frames=100,interval=10)

plt.show()
# def animate(i):
#     cb.remove()
#     plt.cla()
#     plt.contourf(X,Y,n[:,:,i],64)
#     cb = plt.colorbar()

# ani = FuncAnimation(plt.gcf(), animate, frames=50,interval=100)

# #plt.contourf(X,Y,n[:,:,5],64)
# #plt.axis('equal')
# plt.show()

