import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from importlib import reload
import analysis as ana; reload(ana)

HOMEDIR, load_ps, load_sp, _read_header = ana.HOMEDIR, ana.load_ps, ana.load_sp, ana._read_header

DATADIR = 'data/EL-BY-EL-TRACKING/'

ps = load_ps(HOMEDIR+DATADIR)
sp = load_sp(HOMEDIR+DATADIR)
ELNAMES = np.load(HOMEDIR+'analysis/nica_element_names.npy')

def plot_spin(rng):
    fig, ax = plt.subplots(3,1,sharex=True)
    ax[0].plot(sp['S_X'][rng]); ax[0].set_ylabel('S_X')
    ax[1].plot(sp['S_Y'][rng]); ax[1].set_ylabel('S_Y')
    ax[2].plot(sp['S_Z'][rng]); ax[2].set_ylabel('S_Z')
    ax[2].set_xticks(np.arange(len(ELNAMES)+1))
    ax[2].set_xticklabels(ELNAMES, rotation=45)

def plot(turn_num=1):
    fig, ax = plt.subplots(4,1,sharex=True)
    ax[0].plot(ps['X'][:471*turn_num]); ax[0].set_ylabel('X')
    ax[1].plot(ps['Y'][:471*turn_num]); ax[1].set_ylabel('Y')
    ax[2].plot(ps['T'][:471*turn_num]); ax[2].set_ylabel('T')
    ax[3].plot(ps['D'][:471*turn_num]); ax[3].set_ylabel('D')



if __name__ == '__main__':
    plot_spin(slice(0,471))
