import numpy as np
from matplotlib import pyplot, rcParams

class shift:
    def __radd__(self, i): 
        return slice(i.start+1, i.stop+1)
    def __rsub__(self, i): 
        return slice(i.start-1, i.stop-1)

def flux(psi_l, psi_r, C):
    return .5 * (C + abs(C)) * psi_l + \
           .5 * (C - abs(C)) * psi_r
    
def upwind(psi, i, C):
    return psi[i] - flux(psi[i    ], psi[i+one], C) + \
                    flux(psi[i-one], psi[i    ], C) 
    
def psi_0(x):
    # https://en.wikipedia.org/wiki/Witch_of_Agnesi
    a = 5
    return 8*a**3 / (x**2 + 4*a**2)

one = shift()
halo=1

###
from mpi4py.MPI import COMM_WORLD as mpi
import math
###

def calc(psi, nt, C):
    i = slice(halo, psi.size-halo)

    ###
    rank = mpi.Get_rank()
    size = mpi.Get_size()
    
    right = (rank + 1) % size
    left = (rank - 1 + size) % size
    ###
    
    for _ in range(nt):
        ###
        mpi.send(psi[i][-halo:], dest=right)
        psi[:halo] = mpi.recv(source=left)
        mpi.send(psi[i][:halo], dest=left)
        psi[-halo:] = mpi.recv(source=right)
        ###
        
        psi[i] = upwind(psi, i, C)
    
    return psi

def plot(x, psi, psi_0, nt, v):
    ###
    rcParams["figure.figsize"] = [8/mpi.Get_size(), 5]
    ###
    pyplot.step(x, psi_0(x), label='initial', where='mid')
    pyplot.step(x, psi_0(x-v*nt), label='analytical', where='mid')
    pyplot.step(x, psi, label='numerical', where='mid')
    pyplot.grid()
    pyplot.gca().set_ylim([0,12])
    pyplot.legend()
    ###
    pyplot.savefig(f"out.{mpi.Get_rank()}.svg")
    ###

def main(nt, nx, dt, C, x_min, x_max):
    dx = (x_max - x_min) / nx

    ###
    rank = mpi.Get_rank()
    size = mpi.Get_size()
    
    nx_max = math.ceil(nx / size)
    x_min += dx * nx_max * rank
    x_max = min(x_max, x_min + dx * nx_max)
    nx = nx_max if (rank+1) * nx_max <= nx else nx - rank * nx_max
    assert nx > 0
    ###

    x = np.linspace(x_min-halo*dx, x_max+halo*dx, num=nx+2*halo, endpoint=False)
    psi = calc(psi_0(x), nt, C)
    plot(x[halo:-halo], psi[halo:-halo], psi_0, nt, v=C/dt*dx)
    
main(nt=50, nx=75, dt=1, C=.5, x_min=-100, x_max=200)