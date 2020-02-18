import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from importlib import reload
import analysis as ana; reload(ana)

HOMEDIR, load_ps, load_sp, _read_header = ana.HOMEDIR, ana.load_ps, ana.load_sp, ana._read_header

DATADIR = 'data/EL-BY-EL-TRACKING/'

ps = load_ps(HOMEDIR+DATADIR)
sp = load_sp(HOMEDIR+DATADIR)


ELNAMES = np.insert(np.load('nica_element_names.npy'),0,'INJ')
ELNAMES = np.array([e+' ['+ str(i) + ']' for i,e in enumerate(ELNAMES)])

def plot_spin(dat):
    fig, ax = plt.subplots(3,1,sharex=True)
    ax[0].plot(dat['EID'], dat['S_X']); ax[0].set_ylabel('S_X')
    ax[1].plot(dat['EID'], dat['S_Y']); ax[1].set_ylabel('S_Y')
    ax[2].plot(dat['EID'], dat['S_Z']); ax[2].set_ylabel('S_Z')
    ax[2].set_xlabel('EID')
    plt.xticks(ticks=dat['EID'][:,0], labels=ELNAMES[dat['EID'][:,0]], rotation=60)

def plot(dat):
    fig, ax = plt.subplots(4,1,sharex=True)
    ax[0].plot(dat['X']); ax[0].set_ylabel('X')
    ax[1].plot(dat['Y']); ax[1].set_ylabel('Y')
    ax[2].plot(dat['T']); ax[2].set_ylabel('T')
    ax[3].plot(dat['D']); ax[3].set_ylabel('D')



if __name__ == '__main__':
    # plot_spin(sp[:,[0, 1, 28]])
    # plot(ps[:,[0,1,28]])
    plot(ps)
