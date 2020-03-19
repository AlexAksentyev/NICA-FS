import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import HOMEDIR, DAVEC, load_ps, load_sp

DIR  = 'data/TEST/'

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

def plot_ps(data, varx, vary, turns=slice(0,None)):
    fig2, ax2 = plt.subplots(1,1)
    ax2.plot(data[varx][turns], data[vary][turns], '-.')
    ax2.set_ylabel(vary)
    ax2.set_xlabel(varx)
    ax2.ticklabel_format(axis='both', style='sci', scilimits=(0,0), useMathText=True)


if __name__ == '__main__':
    ps0 =  load_ps(HOMEDIR+DIR, 'TRPRAY.dat', ndim=3)
    ps = ps0[:, :17]
    plot_ps(ps, 'T','D')

