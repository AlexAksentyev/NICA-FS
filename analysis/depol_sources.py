import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from importlib import reload
import analysis as ana
reload(ana)
from scipy.optimize import curve_fit

HOMEDIR, load_ps, load_sp = ana.HOMEDIR, ana.load_ps, ana.load_sp

DATADIR = 'data/DEPOL_SOURCES/WITHRF/LONGITUDINAL_SPIN/100turns/'

ELNAMES = np.insert(np.load('nica_element_names.npy'),0,'INJ')
ELNAMES = np.insert(ELNAMES, 1,'RF')
#ELNAMES = np.array([e+' ['+ str(i) + ']' for i,e in enumerate(ELNAMES)])

def pick_elems(name, dat):
    lbls = tick_labels(dat)
    sub_ii = [i for i, e in enumerate(lbls) if name in e]
    return sub_ii, np.array(lbls)[sub_ii]

def tick_labels(dat):
    it = dat['iteration'][:,0]
    nit = np.unique(it[1:])
    eid = dat['EID'][:,0]
    name = ELNAMES[eid]
    return ['{} [{}:{}]'.format(*e) for e in list(zip(name, it, eid))]

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
    ax[0].plot(dat['S_X']); ax[0].set_ylabel('S_X')
    ax[0].grid(axis='x')
    ax[1].plot(dat['S_Y']); ax[1].set_ylabel('S_Y')
    ax[1].grid(axis='x')
    ax[2].plot(dat['S_Z']); ax[2].set_ylabel('S_Z')
    ax[2].grid(axis='x')
    ax[2].set_xlabel('(TURN, EID)')
    plt.xticks(ticks=np.arange(dat.shape[0]), labels=tick_labels(dat), rotation=60)
    

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

def plot_dm_angle2d(dat, same_axis=True, deg=True, elem='all'):
    phih = np.arccos(dot2d(dat))
    phiv = np.arccos(dot2d(dat, 'V'))
    phit = np.arccos(dot2d(dat, 'T'))
    if deg:
        phih, phiv, phit = (np.rad2deg(e) for e in [phih, phiv, phit])
        ylabel_app = ' [deg]'
    else:
        ylabel_app = ' [rad]'
    hd_meas = phih.std(axis=1)
    vd_meas = phiv.std(axis=1)
    td_meas = phit.std(axis=1)
    #it = dat['EID'][:,0]
    if elem=='all':
        jj = np.arange(len(hd_meas))
        lbls = tick_labels(dat)
        lab_pref = ''
        ylabel = lab_pref + r'$\sigma$' + ylabel_app
    else:
        jj, lbls = pick_elems(elem, dat)
        hd_meas, vd_meas, td_meas = (np.diff(np.insert(e,0,0)) for e in [hd_meas, vd_meas, td_meas])
        lab_pref = r'$\Delta$'
        ylabel = lab_pref+ r'$\sigma$' + ylabel_app
    if same_axis:
        fig, ax = plt.subplots(1,1)
        ax.plot(hd_meas[jj], label=r'$\theta_{xz}$')
        ax.plot(vd_meas[jj], label=r'$\theta_{zy}$')
        ax.plot(td_meas[jj], label=r'$\theta_{xy}$')
        ax.legend()
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        ax.set_xlabel('EID')
        ax.set_ylabel(ylabel)
        ax.grid(axis='x')
    else:
        fig, ax = plt.subplots(3,1,sharex=True)
        ax[0].plot(hd_meas[jj], label='hor')
        ax[0].set_ylabel(lab_pref + r'$\sigma(\theta_{xz})$' + ylabel_app)
        ax[1].plot(vd_meas[jj], label='vert')
        ax[1].set_ylabel(lab_pref + r'$\sigma(\theta_{zy})$' + ylabel_app)
        ax[2].plot(td_meas[jj], label='tran')
        ax[2].set_ylabel(lab_pref + r'$\sigma(\theta_{xy})$' + ylabel_app)
        ax[2].set_xlabel('turn')
        for i in range(3):
            ax[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
            ax[i].grid(axis='x')
    plt.xticks(ticks=np.arange(len(jj)), labels=lbls, rotation=60)
    

def plot_dm_angle3d(dat, deg=True, ii=slice(1,None), elem='all'):
    dp = dot3d(dat, ii)
    phi = np.arccos(dp)
    title = 'RMS deviation angle from reference (1st injected ray)'
    if deg:
        phi = np.rad2deg(phi)
        ylabel_app = ' [deg]'
    else:
        ylabel_app = ' [rad]'
    dm = phi.std(axis=1)
    if elem=='all':
        jj = np.arange(len(dm))
        lbls = tick_labels(dat)
        ylabel = r'$\sigma[\arccos(\vec s_1\cdot\vec s_0)]$' + ylabel_app
    else:
        jj, lbls = pick_elems(elem, dat)
        dm = np.diff(np.insert(dm, 0, 0))
        ylabel = r'$\Delta\sigma[\arccos(\vec s_1\cdot\vec s_0)]$' + ylabel_app
    fig, ax = plt.subplots(1,1)
    ax.plot(dm[jj])
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    ax.set_xlabel('EID')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(ticks=np.arange(len(jj)), labels=lbls, rotation=60)
    plt.grid(axis='x')
    

def plot_pol(dat, diff=False):
    P = pol(dat)
    if diff:
        P = np.diff(P)
    #it = dat['EID'][:,0]
    fig, ax = plt.subplots(1,1)
    ax.plot(P)
    ax.set_xlabel('EID')
    if diff:
        ax.set_ylabel(r'$\Delta P$')
    else:
        ax.set_ylabel('P')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    plt.xticks(ticks=np.arange(dat.shape[0]), labels=tick_labels(dat), rotation=60)
    plt.grid(axis='x')

def pol(dat):
    PX, PY, PZ = (dat['S_'+e].sum(axis=1) for e in ['X','Y','Z'])
    N = dat.shape[1]
    return np.sqrt(PX**2 + PY**2 + PZ**2)/N

def dot2d(s1, plane='H'):
    pdict = {'H':('S_X', 'S_Z'), 'V':('S_Z','S_Y'), 'T':('S_X','S_Y')}
    c1, c2 = pdict[plane.upper()]
    if plane=='H':
        s0 = np.array([(0, 0, 1)], dtype=list(zip(['S_X','S_Y','S_Z'], [float]*3)))
    elif plane=='T':
        s0 = np.array([(0, 1, 0)], dtype=list(zip(['S_X','S_Y','S_Z'], [float]*3)))
    else: # plane == 'V'
        s0 = np.array([(0, 1/np.sqrt(2), 1/np.sqrt(2))], dtype=list(zip(['S_X','S_Y','S_Z'], [float]*3)))
    s1n = norm3d(s1) #np.sqrt(s1[c1]**2 + s1[c2]**2)
    s1 = {e:np.divide(s1[e], s1n, where=s1n!=0, out=np.zeros(s1.shape)) for e in [c1,c2]}
    ## |s1| = |s0| = 1
    dp = s1[c1]*s0[c1] + s1[c2]*s0[c2]
    return dp

def norm3d(svec):
    sx, sy, sz = [svec['S_'+e] for e in ['X','Y','Z']]
    return np.sqrt(sx**2 + sy**2 + sz**2)
    
def dot3d(spdat, ii=slice(1,None)):
    s0 = spdat[:,0] # reference ray is at index 0
    s1 = spdat[:,ii]
    s0n = norm3d(s0)
    s1n = norm3d(s1)
    s0 = {'S_'+e: s0['S_'+e]/s0n for e in ['X','Y','Z']}
    s1 = {'S_'+e:s1['S_'+e]/s1n for e in ['X','Y','Z']}
    ## |s0| = |s1| = 1
    dp = s1['S_X'].T*s0['S_X'] + s1['S_Y'].T*s0['S_Y'] + s1['S_Z'].T*s0['S_Z']
    ## cos (s1, s2) = dp/|s1|/|s2| = dp
    return dp.T

def synchrotron_osc(dat, plot=False):
    fun = lambda x, a,b,p: a*np.sin(b*x + p)
    nray = dat.shape[1]
    stat = np.empty((nray, 3), dtype = list(zip(['EST','SE'], [float]*2)))
    fig, ax = plt.subplots(1,1)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    ax.set_xlabel('turn')
    ax.set_ylabel('D')
    for i,ray in enumerate(dat.T):
        #it = ray['iteration']
        d = ray['D']
        a0 = d.max()
        f0 = guess_freq(it, d/a0)
        p0 = guess_phase(it, d/a0)
        popt, pcov = curve_fit(fun, it, d, [a0, f0, p0])
        perr = np.sqrt(np.diag(pcov))
        stat[i]['EST'] = popt
        stat[i]['SE']  = perr
        ax.plot(d, '.')
        ax.plot(fun(it, *popt), '--r')
    plt.xticks(ticks=np.arange(dat.shape[0]), labels=tick_labels(dat), rotation=60)
    return stat

if __name__ == '__main__':
    ps0 = load_ps(HOMEDIR+DATADIR, 'TRPRAY:TREL.dat')
    sp0 = load_sp(HOMEDIR+DATADIR, 'TRPSPI:TREL.dat')
    ps = ps0[:1+472*10,:]#3::3]
    sp = sp0[:1+472*10,:]#3::3]
    plot_dm_angle3d(sp)
    plot_spin(sp)
    plot_dm_angle3d(sp, elem='SOL')
    # plot_dm_angle3d(sp, elem='QUAD')
    # plot_dm_angle3d(sp, elem='RB')
