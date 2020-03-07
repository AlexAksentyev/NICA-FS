import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from importlib import reload
import analysis as ana
reload(ana)

HOMEDIR, load_ps, load_sp, _read_header = ana.HOMEDIR, ana.load_ps, ana.load_sp, ana._read_header

DATADIR = 'data/DECOH/'

def plot_spin(dat):
    fig, ax = plt.subplots(3,1,sharex=True)
    ax[0].plot(dat['iteration'], dat['S_X']); ax[0].set_ylabel('S_X')
    ax[1].plot(dat['iteration'], dat['S_Y']); ax[1].set_ylabel('S_Y')
    ax[2].plot(dat['iteration'], dat['S_Z']); ax[2].set_ylabel('S_Z')
    ax[2].set_xlabel('iteration')

def plot(dat):
    fig, ax = plt.subplots(4,1,sharex=True)
    ax[0].plot(dat['X']); ax[0].set_ylabel('X')
    ax[1].plot(dat['Y']); ax[1].set_ylabel('Y')
    ax[2].plot(dat['T']); ax[2].set_ylabel('T')
    ax[3].plot(dat['D']); ax[3].set_ylabel('D')

def plot_ps(data, varx, vary, turns):
    fig2, ax2 = plt.subplots(1,1)
    ax2.plot(data[varx][:turns], data[vary][:turns], '-.')
    ax2.set_ylabel(vary)
    ax2.set_xlabel(varx)
    ax2.ticklabel_format(axis='both', style='sci', scilimits=(0,0), useMathText=True)

def pol(dat):
    PX, PY, PZ = (dat['S_'+e].sum(axis=1) for e in ['X','Y','Z'])
    N = dat.shape[1]
    return np.sqrt(PX**2 + PY**2 + PZ**2)/N

if __name__ == '__main__':
    ps = load_ps(HOMEDIR+DATADIR)
    sp = load_sp(HOMEDIR+DATADIR)
    P = pol(sp)
    plt.plot(P)
    plot(ps)

