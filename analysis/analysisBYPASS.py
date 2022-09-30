from analysis import DAVEC, Polarization, load_data
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from matplotlib import cm
from mpl_toolkits import mplot3d


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


def plot_spin(spdat, rng=slice(0,-1,50),pid = [1,2,3], title=''):
    fig, ax = plt.subplots(3,1,sharex=True)
    ax[0].set_title(title)
    ax[0].plot(spdat[rng, pid]['iteration']/1000, spdat[rng,pid]['S_X'])
    ax[0].set_ylabel('S_X')
    ax[1].plot(spdat[rng, pid]['iteration']/1000, spdat[rng,pid]['S_Y'])
    ax[1].set_ylabel('S_Y')
    ax[2].plot(spdat[rng, pid]['iteration']/1000, spdat[rng,pid]['S_Z'])
    ax[2].set_xlabel('turn [x1000]'); ax[2].set_ylabel('S_Z')
    return fig, ax


if __name__ == '__main__':
    z = np.zeros(11, dtype=list(zip(['X','A','Y','B','T','D'], [float]*6)))
    nu_pri = DAVEC(HOME+"NU:FULL.da")
    nu_opt = DAVEC(HOME+"NU:FULL-optSGxy.da")
    ## plot
    fig, ax = plt.subplots(1,2)
    ## vs X
    z['X'] = np.linspace(-1e-3,1e-3,11)
    ax[0].plot(z['X']*1000, nu_pri(z) - nu_pri.const, label=r"prior, $\nu_0$ = {:4.2e}".format(nu_pri.const))
    ax[0].plot(z['X']*1000, nu_opt(z) - nu_opt.const, label=r"optim, $\nu_0$ = {:4.2e}".format(nu_opt.const))
    ax[0].legend(); ax[0].set_xlabel('X [mm]'); ax[0].set_ylabel(r'$\nu$')
    ax[0].ticklabel_format(style='sci', scilimits=(0,0), useMathText=True, axis='y')
    ## vs Y
    z['Y'] = np.linspace(-1e-3,1e-3,11); z['X'] = np.zeros(11);
    ax[1].plot(z['Y']*1000, nu_pri(z) - nu_pri.const, label=r"prior, $\nu_0$ = {:4.2e}".format(nu_pri.const))
    ax[1].plot(z['Y']*1000, nu_opt(z) - nu_opt.const, label=r"optim, $\nu_0$ = {:4.2e}".format(nu_opt.const))
    ax[1].legend(); ax[1].set_xlabel('Y [mm]'); ax[1].set_ylabel(r'$\nu$')
    ax[1].ticklabel_format(style='sci', scilimits=(0,0), useMathText=True, axis='y')
   
    # look at the tracking data
    spdat_opt = load_data(HOME,'TRPSPI:FULL-optSGxy.dat')
    spdat = load_data(HOME,'TRPSPI:FULL.dat')
    plot_spin(spdat, title='NO SEXT')
    plot_spin(spdat_opt, title='W/SEXT')
