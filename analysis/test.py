import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import HOMEDIR, DAVEC, load_data

DIR  = '../data/TEST/FIRST-ST/NAVI-5deci/300000/'

def load_tss(path=HOMEDIR+DIR+'MU.dat'):
    d_type = [('EL', int), ('PID', int)] + list(zip(['NU', 'NX','NY','NZ'], [float]*4))
    dat = np.loadtxt(path, dtype=d_type)
    nray = len(np.unique(dat['PID']))
    dat.shape = (-1, nray)
    return dat[:, 1:]

def plot(dat, spdat, rng = slice(0,-1,50), pid = [1,2,3], fmt='.-'):
    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(dat[rng,pid]['X']*1000, dat[rng,pid]['A']*1000, fmt)
    ax[0,0].set_xlabel('X [mm]'); ax[0,0].set_ylabel('A [mrad]')
    ax[0,1].plot(dat[rng,pid]['Y']*100, dat[rng,pid]['B']*1000, fmt)
    ax[0,1].set_xlabel('Y [mm]'); ax[0,1].set_ylabel('B [mrad]')
    ax[1,0].plot(dat[rng,pid]['T'], dat[rng,pid]['D'], fmt)
    ax[1,0].set_xlabel('T'); ax[1,0].set_ylabel('D');
    ax[1,0].ticklabel_format(style='sci', scilimits=(0,0), useMathText=True, axis='both')
    ax[1,1].plot(spdat[rng, pid]['iteration']/1000, spdat[rng,pid]['S_Z'])
    ax[1,1].set_xlabel('turn [x1000]'); ax[1,1].set_ylabel('S_Z')
    #ax[1,1].ticklabel_format(style='sci', scilimits=(0,0), useMathText=True, axis='x')
    return fig,ax

def plot_seq(dat, spdat, pid = [1,2,3], itn=(0,1)):
    if type(itn)==int:
        ps1 = dat[dat[:,0]['iteration']<itn+1]
        sp1 = spdat[spdat[:,0]['iteration']<itn+1]
        eid = ps1['EID'][:, pid] if itn<2 else np.arange(ps1['EID'].max()*itn+1)
    else:
        itrow = dat[:,0]['iteration']
        ii = np.logical_and(itrow>itn[0], itrow<itn[1]+1)
        itrng = itn[1]-itn[0]
        ps1 = dat[ii]
        sp1 = spdat[ii]
        eid_max = ps1['EID'].max()
        eid = eid_max*itn[0] + np.arange(eid_max*itrng)
    fig, ax = plt.subplots(3,1)
    ax[0].plot(eid, ps1[:,pid]['X']*1000)
    ax[0].set_xlabel('EID'); ax[0].set_ylabel('X [mm]')
    ax[1].plot(eid, ps1[:,pid]['Y']*1000)
    ax[1].set_xlabel('EID'); ax[1].set_ylabel('Y[mm]')
    ax[2].plot(eid, sp1[:,pid]['S_Z'])
    ax[2].set_xlabel('EID'); ax[2].set_ylabel('S_Z')
    for i in range(3):
        ax[i].grid()
    return fig, ax
    

if __name__ == '__main__':
    dat = load_data(DIR, 'TRPRAY.dat')
    spdat = load_data(DIR, 'TRPSPI.dat')
    if DIR[-4:-1]=='SEQ':
        fig, ax = plot_seq(dat, spdat)
    else:
        fig, ax = plot(dat, spdat)
