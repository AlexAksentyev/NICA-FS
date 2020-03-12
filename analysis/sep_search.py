import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import HOMEDIR, DAVEC, load_ps, load_sp

DIR  = 'data/TEST/'

# D_TYPE_BASE = list(zip(['iteration', 'PID'], [int]*2))
# VARS_PS = list(zip(['X','A','Y','B','T','D'], [float]*6))
# VARS_SP = list(zip(['X','Y','Z'], [float]*3))
# D_TYPE_PS = D_TYPE_BASE + VARS_PS
# D_TYPE_SP = D_TYPE_BASE + VARS_SP

def plot_spin(data, turns):
    fig1, ax1 = plt.subplots(4,1, sharex=True)
    for i, var in enumerate(['X','Y','Z']):
        ax1[i].plot(data[var][:turns])
        ax1[i].set_ylabel('S_'+var)
        ax1[i].ticklabel_format(axis='both', style='sci', scilimits=(0,0), useMathText=True)
    S_X, S_Y, S_Z = (data[lbl] for lbl in ['X','Y','Z'])
    Snorm = np.sqrt(S_X**2 + S_Y**2 + S_Z**2)
    ax1[3].plot(Snorm[:turns]-1)
    ax1[3].set_ylabel('|S|-1')
    ax1[3].ticklabel_format(axis='both', style='sci', scilimits=(0,0), useMathText=True)

def plot_ps(data, varx, vary, turns):
    fig2, ax2 = plt.subplots(1,1)
    ax2.plot(data[varx][:turns], data[vary][:turns], '-.')
    ax2.set_ylabel(vary)
    ax2.set_xlabel(varx)
    ax2.ticklabel_format(axis='both', style='sci', scilimits=(0,0), useMathText=True)


if __name__ == '__main__':
    ps0 =  load_ps(HOMEDIR+DIR, 'TRPRAY.dat', ndim=3)
    ps = ps0[:,4:7]
    # sp =  load_sp(HOMEDIR+DIR, 'TRPSPI.dat')
    # plot_ps(ps[:,:5], 'EID','X', 471*5)
    # plot_ps(ps[:,:5], 'EID','Y', 471*5)
    fig, ax = plt.subplots(3,1,sharex=False)
    ax[0].plot(ps['X'], ps['A'], '.'); ax[0].set_xlabel('X'); ax[0].set_ylabel('A')
    ax[1].plot(ps['Y'], ps['B'], '.'); ax[1].set_xlabel('Y'); ax[1].set_ylabel('B')
    ax[2].plot(ps['T'], ps['D'], '.'); ax[2].set_xlabel('T'); ax[2].set_ylabel('D')
    # ax[3].plot(ps['D']); ax[3].set_ylabel('D')

    # fig1, ax1 = plt.subplots(3,1,sharex=True)
    # ax1[0].plot(sp['S_X']); ax1[0].set_ylabel('S_X')
    # ax1[1].plot(sp['S_Y']); ax1[1].set_ylabel('S_Y')
    # ax1[2].plot(sp['S_Z']); ax1[2].set_ylabel('S_Z')
    ray = ps0[:,0]
