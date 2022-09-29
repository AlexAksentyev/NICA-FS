from analysis import DAVEC, Polarization, load_data
import numpy as np
import matplotlib.pyplot as plt; plt.ion()


HOME = "../data/BYPASS/"

def load_trMap(fname):
    VARS  = ['X','A','Y','B','T','D']
    NVARS = len(VARS)
    VIN = ['X','A','Y','B','T']
    DTYPE = [('dummy', object)] + list(zip(VIN, [float]*5)) + [('EXP', int)]
    tmp = np.genfromtxt(fname, skip_footer = 1,
                        #dtype=DTYPE,
                        delimiter=(1, 14, 14, 14, 14, 14, 7),
                        usecols = range(1,NVARS))
    return tmp


def analysis_nu(fname):
    z = np.zeros(11, dtype=list(zip(['X','A','Y','B','T','D'], [float]*6)))
    z['X'] = np.linspace(-1e-3,1e-3,11)
    nu = DAVEC(HOME+"NU:FULL.da")
    plt.plot(z['X'], nu(z))

def plot_spin(spdat, rng=slice(0,-1,50),pid = [1,2,3], fmt='.-'):
    fig, ax = plt.subplots(3,1,sharex=True)
    ax[0].plot(spdat[rng, pid]['iteration']/1000, spdat[rng,pid]['S_X'])
    ax[0].set_ylabel('S_X')
    ax[1].plot(spdat[rng, pid]['iteration']/1000, spdat[rng,pid]['S_Y'])
    ax[1].set_ylabel('S_Y')
    ax[2].plot(spdat[rng, pid]['iteration']/1000, spdat[rng,pid]['S_Z'])
    ax[2].set_xlabel('turn [x1000]'); ax[2].set_ylabel('S_Z')
    return fig, ax


if __name__ == '__main__':
    TMDL = load_trMap(HOME+"WF2DL/TrMAP:FULL-1D")
    TMWF = load_trMap(HOME+"WF2WF/TrMAP:FULL-1D")
    datDL = load_data(HOME+"WF2DL/", "TRPRAY:FULL.dat")
    datWF = load_data(HOME+"WF2WF/", "TRPRAY:FULL.dat")
    rng = slice(0,-1,50); pid = [1,2,3]
    fig, ax = plt.subplots(3,2,sharey='row')
    ## left column for WF2WF
    ax[0,0].set_title('WF2WF')
    ax[0,0].plot(datWF[rng,pid]['X']*1000, datWF[rng,pid]['A']*1000)
    ax[0,0].set_xlabel('X [mm]'); ax[0,0].set_ylabel('A [mrad]')
    ax[1,0].plot(datWF[rng,pid]['Y']*1000, datWF[rng,pid]['B']*1000)
    ax[1,0].set_xlabel('Y [mm]'); ax[1,0].set_ylabel('B [mrad]')
    ax[2,0].plot(datWF[rng,pid]['T'], datWF[rng,pid]['D'])
    ax[2,0].set_xlabel('T'); ax[2,0].set_ylabel('D')
    ax[2,0].ticklabel_format(style='sci', scilimits=(0,0), useMathText=True, axis='both')
    ## right column for WF2DL
    ax[0,1].set_title('WF2DL')
    ax[0,1].plot(datDL[rng,pid]['X']*1000, datDL[rng,pid]['A']*1000)
    ax[0,1].set_xlabel('X [mm]'); ax[0,0].set_ylabel('A [mrad]')
    ax[1,1].plot(datDL[rng,pid]['Y']*1000, datDL[rng,pid]['B']*1000)
    ax[1,1].set_xlabel('Y [mm]'); ax[1,1].set_ylabel('B [mrad]')
    ax[2,1].plot(datDL[rng,pid]['T'], datDL[rng,pid]['D'])
    ax[2,1].set_xlabel('T'); ax[2,1].set_ylabel('D')
    ax[2,1].ticklabel_format(style='sci', scilimits=(0,0), useMathText=True, axis='both')
    
   
