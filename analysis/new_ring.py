import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from importlib import reload
import analysis as ana; reload(ana)

HOMEDIR, load_ps, load_sp, _read_header = ana.HOMEDIR, ana.load_ps, ana.load_sp, ana._read_header

DATADIR = 'data/NEW_RING/'

def plot(data, ndim=3):
    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(data['X']*1e3, data['A']*1e3, '.')
    ax[0,0].set_xlabel('X [mm]'); ax[0,0].set_ylabel('A [mrad]')
    ax[0,1].plot(data['Y']*1e3, data['B']*1e3, '.')
    ax[0,1].set_xlabel('Y [mm]'); ax[0,1].set_ylabel('B [mrad]')
    if ndim>2:
        ax[1,0].plot(data['T'], data['D'], '.')
        ax[1,0].set_xlabel('T [m]'); ax[1,0].set_ylabel('D')
        ax[1,1].plot(data['X']*1e3, label='X')
        ax[1,1].plot(data['Y']*1e3, label='Y')
        ax[1,1].set_ylabel('Y [mm]')

if __name__ == '__main__':
    ps = load_ps(HOMEDIR+DATADIR, ndim=2)
    plot(ps, 2)
    
