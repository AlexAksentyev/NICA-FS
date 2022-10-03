from analysis import DAVEC, Polarization, load_data
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from matplotlib import cm
from mpl_toolkits import mplot3d


HOME = "../data/BYPASS/"

BETA = 0.4641825 # injection at y = 1.129 (242.01975 MeV) Deuterons
CLIGHT = 3e8
Lacc = 503.04
v = CLIGHT*BETA
TAU = Lacc/v

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

def load_nbar(fname, mrkr):
    return [DAVEC(HOME+"NBAR{}:".format(i)+mrkr+".da") for i in range(1,4)]

def vecdisp(spdat):
    SX, SY, SZ = [spdat['S_'+lbl] for lbl in ('X','Y','Z')]
    Nray = SX.shape[1]-1 # bar the referecne ray
    SX0, SY0, SZ0 = [el[:,0].repeat(Nray).reshape(-1,Nray) for el in (SX, SY, SZ)]
    prod = SX[:,1:]*SX0 + SY[:,1:]*SY0 + SZ[:, 1:]*SZ0
    return prod

if __name__ == '__main__':
    z = np.zeros(11, dtype=list(zip(['X','A','Y','B','T','D'], [float]*6)))
    nu_pri = DAVEC(HOME+"NU:3M_psi45.da")
    nu_opt = DAVEC(HOME+"NU:3M-optSGxy_psi45.da")
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
    spdat_opt = load_data(HOME,'TRPSPI:3M-optSGxy_psi45.dat')
    spdat = load_data(HOME,'TRPSPI:3M_psi45.dat')
    plot_spin(spdat, title='NO SEXT')
    plot_spin(spdat_opt, title='W/SEXT')

    # polarization
    P_pri = Polarization.on_axis(spdat, axis=[0,-1,0])
    P_opt = Polarization.on_axis(spdat_opt, axis=[0,-1,0])
    # vector dispersion
    disp_pri = vecdisp(spdat)
    disp_opt = vecdisp(spdat_opt)
    t = spdat[:,0]['iteration']*TAU
    fig, ax = plt.subplots(1,1)
    ax.plot(disp_pri.std(1), label='unoptimized')
    ax.plot(disp_opt.std(1), label='optimized')
    ax.set_ylabel(r'$\sigma[cos(\vec s\cdot\vec s_0)]$')
    ax.set_xlabel('time [sec]')
    ax.legend()
    ax.ticklabel_format(style='sci', scilimits=(0,0), useMathText=True, axis='y')
