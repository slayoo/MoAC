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

def calc(psi, nt, C):
    i = slice(halo, psi.size-halo)
    
    for _ in range(nt):
        psi[:halo] = psi[i][-halo:]
        psi[-halo:] = psi[i][:halo]
        
        psi[i] = upwind(psi, i, C)
    
    return psi

def plot(x, psi, psi_0, nt, v):
    pyplot.step(x, psi_0(x), label='initial', where='mid')
    pyplot.step(x, psi_0(x-v*nt), label='analytical', where='mid')
    pyplot.step(x, psi, label='numerical', where='mid')
    pyplot.grid()
    pyplot.gca().set_ylim([0,12])
    pyplot.legend()
    pyplot.savefig("out.svg")

def main(nt, nx, dt, C, x_min, x_max):
    dx = (x_max - x_min) / nx

    x = np.linspace(x_min-halo*dx, x_max+halo*dx, num=nx+2*halo, endpoint=False)
    psi = calc(psi_0(x), nt, C)
    plot(x[halo:-halo], psi[halo:-halo], psi_0, nt, v=C/dt*dx)
    
main(nt=50, nx=75, dt=1, C=.5, x_min=-100, x_max=200)