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
    ps =  load_ps(HOMEDIR+DIR, 'TRPRAY:COSY.dat')
    # sp =  load_sp(HOMEDIR+DIR, 'TRPSPI:COSY.dat')


    tmp = np.genfromtxt(HOMEDIR+DIR+'MAP', skip_footer = 1,
                        #dtype=DTYPE,
                        delimiter=(1, 14, 14, 14, 14, 14, 7),
                        usecols = range(1,6))
    MAP = np.zeros((6,6))
    MAP[:5,:] = tmp.T
    MAP[5,-1] = 1

    ntrn = 1000
    z = np.zeros((ntrn, 6,4))
    z[0,0,:] = [-1e-3, 0, 1e-3, 0]
    z[0, 2,:] = [0, -1e-3, 0, 1e-3]

    for i in range(1, ntrn):
        z[i] = np.matmul(MAP, z[i-1])
    

    f, ax = plt.subplots(2,1, sharex=True)
    ax[0].plot(z[:100,0,:]); ax[0].set_ylabel('X')
    ax[1].plot(z[:100,2,:]); ax[1].set_ylabel('Y')
