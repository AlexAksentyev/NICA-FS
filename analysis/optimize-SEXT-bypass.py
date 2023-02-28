import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import DAVEC, Polarization, load_data


DIR = '../data/BYPASS_SEX_CLEAR/optimize-SEXT/NO-DELTAP/'

GAMMA = 1.1279235
BETA = np.sqrt(GAMMA**2-1)/GAMMA # injection at y = 1.129 (242.01975 MeV) Deuterons
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

def vecdisp(spdat):
    SX, SY, SZ = [spdat['S_'+lbl] for lbl in ('X','Y','Z')]
    Nray = SX.shape[1]-1 # bar the referecne ray
    SX0, SY0, SZ0 = [el[:,0].repeat(Nray).reshape(-1,Nray) for el in (SX, SY, SZ)]
    prod = SX[:,1:]*SX0 + SY[:,1:]*SY0 + SZ[:, 1:]*SZ0
    return prod

def nu_analysis(dir):
    z = np.zeros(11, dtype=list(zip(['X','A','Y','B','T','D'], [float]*6)))
    nu_pre  = DAVEC(dir+"NU:pre-opt.da")
    nu_post = DAVEC(dir+"NU:post-opt.da")
    ## plot
    fig, ax = plt.subplots(1,3)
    ## vs X
    z['X'] = np.linspace(-1e-3,1e-3,11)
    ax[0].plot(z['X']*1000, nu_pre(z) - nu_pre.const, label=r"prior, $\nu_0$ = {:4.2e}".format(nu_pre.const))
    ax[0].plot(z['X']*1000, nu_post(z) - nu_post.const, label=r"optim, $\nu_0$ = {:4.2e}".format(nu_post.const))
    ax[0].legend(); ax[0].set_xlabel('X [mm]'); ax[0].set_ylabel(r'$\nu$')
    ax[0].ticklabel_format(style='sci', scilimits=(0,0), useMathText=True, axis='y')
    ## vs Y
    z['Y'] = np.linspace(-1e-3,1e-3,11); z['X'] = np.zeros(11);
    ax[1].plot(z['Y']*1000, nu_pre(z) - nu_pre.const, label=r"prior, $\nu_0$ = {:4.2e}".format(nu_pre.const))
    ax[1].plot(z['Y']*1000, nu_post(z) - nu_post.const, label=r"optim, $\nu_0$ = {:4.2e}".format(nu_post.const))
    ax[1].legend(); ax[1].set_xlabel('Y [mm]'); ax[1].set_ylabel(r'$\nu$')
    ax[1].ticklabel_format(style='sci', scilimits=(0,0), useMathText=True, axis='y')
    ## vs D
    z['D'] = np.linspace(-1e-3,1e-3,11); z['Y'] = np.zeros(11);
    ax[2].plot(z['D'], nu_pre(z) - nu_pre.const, label=r"prior, $\nu_0$ = {:4.2e}".format(nu_pre.const))
    ax[2].plot(z['D'], nu_post(z) - nu_post.const, label=r"optim, $\nu_0$ = {:4.2e}".format(nu_post.const))
    ax[2].legend(); ax[2].set_xlabel('D'); ax[2].set_ylabel(r'$\nu$')
    ax[2].ticklabel_format(style='sci', scilimits=(0,0), useMathText=True, axis='both')

def spin_analysis(dir):
    spdat_pre = load_data(dir,'TRPSPI:pre-opt.dat')
    spdat_post = load_data(dir,'TRPSPI:post-opt.dat')
    plot_spin(spdat_pre,  title='NO SEXT')
    plot_spin(spdat_post, title='W/SEXT')
    return spdat_pre, spdat_post

def polarization_analysis(spdat_pre, spdat_post):
    P_pre  = Polarization.on_axis(spdat_pre,  axis=[0,1,0])
    P_post = Polarization.on_axis(spdat_post, axis=[0,1,0])
    # vector dispersion
    disp_pre  = vecdisp(spdat_pre)
    disp_post = vecdisp(spdat_post)
    t = spdat_pre[:,0]['iteration']*TAU
    fig, ax = plt.subplots(1,1)
    ax.plot(t, disp_pre.std(1), label='unoptimized')
    ax.plot(t, disp_post.std(1), label='optimized')
    ax.set_ylabel(r'$\sigma[cos(\vec s\cdot\vec s_0)]$')
    ax.set_xlabel('time [sec]')
    ax.legend()
    ax.ticklabel_format(style='sci', scilimits=(0,0), useMathText=True, axis='y')
    return P_pre, P_post
    
if __name__ == '__main__':
    nu_analysis(DIR)
   
    spdat_pre, spdat_post = spin_analysis(DIR)   
    P_pre, P_post = polarization_analysis(spdat_pre, spdat_post)
    
