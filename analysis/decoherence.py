import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from importlib import reload
import analysis as ana
reload(ana)
from scipy.optimize import curve_fit

HOMEDIR, load_ps, load_sp = ana.HOMEDIR, ana.load_ps, ana.load_sp

DATADIR = 'data/DECOH/'

Fcyc = 286166.4863010005 # Hz

def guess_freq(time, signal): # estimating the initial frequency guess
    zci = np.where(np.diff(np.sign(signal)))[0] # find indexes of signal zeroes
    delta_phase = np.pi*(len(zci)-1)
    delta_t = time[zci][-1]-time[zci][0]
    guess = delta_phase/delta_t/2/np.pi
    return guess

def guess_phase(time, sine):
    ds = sine[1]-sine[0]
    dt = time[1]-time[0]
    sg = np.sign(ds/dt)
    phase0 = np.arcsin(sine[0]) if sg>0 else np.pi-np.arcsin(sine[0])
    return phase0


def plot_spin(dat):
    fig, ax = plt.subplots(3,1,sharex=True)
    ax[0].plot(dat['iteration'], dat['S_X']); ax[0].set_ylabel('S_X')
    ax[1].plot(dat['iteration'], dat['S_Y']); ax[1].set_ylabel('S_Y')
    ax[2].plot(dat['iteration'], dat['S_Z']); ax[2].set_ylabel('S_Z')
    ax[2].set_xlabel('iteration')

def plot(dat):
    fig, ax = plt.subplots(4,1,sharex=True)
    ax[0].plot(dat['X']); ax[0].set_ylabel('X')
    ax[1].plot(dat['Y']); ax[1].set_ylabel('Y')
    ax[2].plot(dat['T']); ax[2].set_ylabel('T')
    ax[3].plot(dat['D']); ax[3].set_ylabel('D')

def plot_ps(data, varx, vary, turns):
    fig2, ax2 = plt.subplots(1,1)
    ax2.plot(data[varx][:turns], data[vary][:turns], '.')
    ax2.set_ylabel(vary)
    ax2.set_xlabel(varx)
    ax2.ticklabel_format(axis='both', style='sci', scilimits=(0,0), useMathText=True)

def plot_dm_angle(dat, same_axis=True, deg=True):
    phih = angles(dat)
    phiv = angles(dat, 'V')
    phit = angles(dat, 'T')
    title = 'RMS deviation angle from (0, 0, 1)'
    if deg:
        phih, phiv, phit = (np.rad2deg(e) for e in [phih, phiv, phit])
        ylabel_app = ' [deg]'
    else:
        ylabel_app = ' [rad]'
    hd_meas = phih.std(axis=1)
    vd_meas = phiv.std(axis=1)
    td_meas = phit.std(axis=1)
    it = dat['iteration'][:,0]
    if same_axis:
        fig, ax = plt.subplots(1,1)
        ax.plot(it, hd_meas, label=r'$\sigma(\theta_{xz})$')
        ax.plot(it, vd_meas, label=r'$\sigma(\theta_{zy})$')
        ax.plot(it, td_meas, label=r'$\sigma(\theta_{xy})$')
        ax.legend()
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        ax.set_xlabel('turn')
        ax.set_ylabel('Statistic' + ylabel_app)
        ax.set_title(title)
    else:
        fig, ax = plt.subplots(3,1,sharex=True)
        ax[0].plot(it, hd_meas, label='hor')
        ax[0].set_ylabel(r'$\sigma(\theta_{xz})$' + ylabel_app)
        ax[1].plot(it, vd_meas, label='vert')
        ax[1].set_ylabel(r'$\sigma(\theta_{zy})$' + ylabel_app)
        ax[2].plot(it, td_meas, label='tran')
        ax[2].set_ylabel(r'$\sigma(\theta_{zy})$' + ylabel_app)
        ax[2].set_xlabel('turn')
        for i in range(3):
            ax[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)

def plot_pol(dat):
    P = pol(dat)
    it = dat['iteration'][:,0]
    fig, ax = plt.subplots(1,1)
    ax.plot(it, P)
    ax.set_xlabel('turn')
    ax.set_ylabel('P')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)

def pol(dat):
    PX, PY, PZ = (dat['S_'+e].sum(axis=1) for e in ['X','Y','Z'])
    N = dat.shape[1]
    return np.sqrt(PX**2 + PY**2 + PZ**2)/N

def angles(s1, plane='H'):
    pdict = {'H':('S_X', 'S_Z'), 'V':('S_Z','S_Y'), 'T':('S_X','S_Y')}
    c1, c2 = pdict[plane.upper()]
    if plane!='T':
        s0 = np.array([(0, 0, 1)], dtype=list(zip(['S_X','S_Y','S_Z'], [float]*3)))
    else:
        s0 = np.array([(0, 1, 0)], dtype=list(zip(['S_X','S_Y','S_Z'], [float]*3)))
    s1n, s0n = (np.sqrt(e[c1]**2 + e[c2]**2) for e in (s1,s0))
    # if np.all((s1n==0) + (s0n == 0)):
    #     return np.zeros(s1.shape)
    dp = s1[c1]*s0[c1] + s1[c2]*s0[c2]
    cos_phi = np.divide(dp/s0n, s1n, where=s1n!=0, out=np.zeros(s1n.shape))
    return np.arccos(cos_phi)

def synchrotron_osc(dat, plot=False):
    fun = lambda x, a,b,p: a*np.sin(b*x + p)
    nray = dat.shape[1]
    stat = np.empty((nray, 3), dtype = list(zip(['EST','SE'], [float]*2)))
    fig, ax = plt.subplots(1,1)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    ax.set_xlabel('turn')
    ax.set_ylabel('D')
    for i,ray in enumerate(dat.T):
        it = ray['iteration']
        d = ray['D']
        a0 = d.max()
        f0 = guess_freq(it, d/a0)
        p0 = guess_phase(it, d/a0)
        popt, pcov = curve_fit(fun, it, d, [a0, f0, p0])
        perr = np.sqrt(np.diag(pcov))
        stat[i]['EST'] = popt
        stat[i]['SE']  = perr
        ax.plot(it, d, '.')
        ax.plot(it, fun(it, *popt), '--r')
    return stat

if __name__ == '__main__':
    ps0 = load_ps(HOMEDIR+DATADIR)
    sp0 = load_sp(HOMEDIR+DATADIR)
    ps = ps0[:,2::3]
    sp = sp0[:,2::3]
    plot_ps(ps,'T','D',-1)
    plot_pol(sp)
    plot_dm_angle(sp)
    plot_spin(sp)
