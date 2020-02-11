import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import HOMEDIR, load_ps

DATADIR = 'data/EL-BY-EL-TRACKING/'

ps = load_ps(HOMEDIR+DATADIR)


def plot(turn_num=1):
    fig, ax = plt.subplots(4,1,sharex=True)
    ax[0].plot(ps['X'][:471*turn_num]); ax[0].set_ylabel('X')
    ax[1].plot(ps['Y'][:471*turn_num]); ax[1].set_ylabel('Y')
    ax[2].plot(ps['T'][:471*turn_num]); ax[2].set_ylabel('T')
    ax[3].plot(ps['D'][:471*turn_num]); ax[3].set_ylabel('D')



if __name__ == '__main__':
    plot(3)
