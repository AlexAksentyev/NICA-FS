# comparison of the MAD-X and COSY infinity solenoid maps

from analysis import HOMEDIR, load_madx, load_cosy
import numpy as np
import matplotlib.pyplot as plt; plt.ion()

LEN = False

def load_madx_maps():
    MDIR = 'madx-scripts/elmap/'
    if LEN:
        MDIR += 'len_vary/'
    mmap = np.zeros((7,6,6)) 
    for i,e in enumerate(['01','02','03','04','05','06','10']): 
        mmap[i] = load_madx(HOMEDIR+MDIR+'SNK_'+e)
    return mmap

def load_cosy_maps():
    CDIR = 'data/ELEMENT_MAPS/SOL26/'
    if LEN:
        CDIR += 'LEN_VARY/'
    cmap = np.zeros((7,6,6)) 
    for i,e in enumerate(['01','02','03','04','05','06','1']): 
        cmap[i] = load_cosy(HOMEDIR+CDIR+'MAP_'+e)
    return cmap

def add_plot(i, j, ax, x, model, i1=None, j1=None):
    if i1 is None:
        i1 = i
    if j1 is None:
        j1 = j
    ax[i,j].plot(x, mmap[:,i1,j1], label='MADX')
    ax[i,j].plot(x, cmap[:,i1,j1], label='COSY')
    ax[i,j].plot(x, model, label='BOOK')
    ax[i,j].legend()
    ax[i,j].ticklabel_format(axis='both', style='sci',scilimits=(0,0), useMathText=True)
    if LEN:
        ax[i,j].set_xlabel('L [m]')
    else:
        ax[i,j].set_xlabel('HZ1 [rad/m]')
    ax[i,j].set_title('matrix element ({},{})'.format(i1,j1))

if __name__ == '__main__':
    mmap = load_madx_maps()
    cmap = load_cosy_maps()
    L = .7
    hz1 = .373991
    pp = np.array([.1,.2,.3,.4,.5,.6,1])
    hz1v = pp*hz1
    Lv = pp*L
    c = np.cos(hz1v/2*L)
    s = np.sin(hz1v/2*L)

    k = hz1/2 if LEN else hz1v/2 # https://uspas.fnal.gov/materials/13Duke/SCL_Chap3.pdf
    x = Lv if LEN else hz1v
    fig, ax = plt.subplots(2,2)
    add_plot(0,0,ax, x, c*c)
    add_plot(0,1,ax, x, s*c/k)
    add_plot(1,0,ax, x, -k*s*c)
    add_plot(1,1,ax, x, s*c, 1,3)
    

    KLmx = np.arccos(np.sqrt(mmap[:,0,0]))
    KLci = np.arccos(np.sqrt(cmap[:,0,0]))
    Kci = KLci/pp/L
    Kmx = KLmx/pp/L
    factor = Kci/Kmx # this factor equals e
    print(factor/np.e)
    
