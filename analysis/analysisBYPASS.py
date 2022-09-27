from analysis import DAVEC, Polarization, load_data
import numpy as np
import matplotlib.pyplot as plt; plt.ion()


HOME = "../data/CLEAN_BYPASS/"


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
    dat = load_data(HOME, 'TRPRAY:FULL.dat')
    spdat = load_data(HOME, 'TRPSPI:FULL.dat')
    P = Polarization.on_axis(spdat, [0,0,1])
    Pd = Polarization.on_axis(spdat[3:30:3], [0,0,1])
    Py = Polarization.on_axis(spdat[2:30:3], [0,0,1])
