import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from importlib import reload
import analysis as ana
reload(ana)
from pandas import DataFrame


HOMEDIR, load_ps, load_sp = ana.HOMEDIR, ana.load_ps, ana.load_sp, ELNAMES, tick_labels

DATADIR = 'data/DEPOL_SOURCES/WITHRF/LONGITUDINAL_SPIN/'


#ELNAMES = np.array([e+' ['+ str(i) + ']' for i,e in enumerate(ELNAMES)])

def norm3d(svec):
    sx, sy, sz = [svec['S_'+e] for e in ['X','Y','Z']]
    return np.sqrt(sx**2 + sy**2 + sz**2)

def pick_elems(name, dat):
    lbls = tick_labels(dat)
    sub_ii = [i for i, e in enumerate(lbls) if name in e]
    return sub_ii, np.array(lbls)[sub_ii]    

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

def plot_dm_angle2d(particle, same_axis=True, deg=True, elem='all'):
    dat = particle.sp
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
        ax[2].set_xlabel('EID')
        for i in range(3):
            ax[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
            ax[i].grid(axis='x')
    plt.xticks(ticks=np.arange(len(jj)), labels=lbls, rotation=90)
    

def plot_dm_angle3d(particle, deg=True, pids=slice(1,None), elem='all'):
    dat = particle.sp
    dp = dot3d(dat, pids)
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
    plt.xticks(ticks=np.arange(len(jj)), labels=lbls, rotation=90)
    plt.grid(axis='x')

def decoherence_derivative(spdat, psdat): # this function computes the derivative
                               # delta-spin-vector-deviation-angle/ delta-initial-ps-offset
                               # from the CO
    def fit_line(x,y): # this is used for evaluating the derivative
        line = lambda x,a,b: a + b*x
        popt, pcov = curve_fit(line, x, y)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    ylab_fun = lambda p,v: r'$\Delta\Theta_{}/\Delta {}$'.format(p,v)
    plane_dict = dict(A='{3d}', H='{xz}', V='{zy}', T='{xy}')
    var_dict = dict(X='x', Y='y', D=r'\delta')
    spc = spdat.copy()[[0,-1]] # initial and final spin vectors for delta-angle evaluation
    psc = psdat.copy()[0] # initial phase space offset from CO
    dphia = np.insert(np.diff(np.arccos(dot3d(spc)), axis=0).flatten(), 0, 0) # computing the 3D dot product and deviation angle
                # zero is inserted at the front represents the deviation angle of the CO ray's spin vector from itself
    dphih, dphiv, dphit = (np.diff(np.arccos(dot2d(spc, e)), axis=0).flatten() for e in ['H','V','T'])
                # here nothing is inserted at the front b/c the vectors i compute the dot products against are other than the
                # CO particle's spin vector
    dphia, phih, dphiv, dphit = (e - e[0] for e in [dphia, dphih, dphiv, dphit]) # here take the difference between the non-CO particles'
                                        # spin rotation angle and that of the CO one <=> deviation angle; for 3D it changes nothing
    dict_i = {e:slice((i+1), None, 3) for i,e in enumerate(['X','Y','D'])} # picking bunch (X,Y,D) particles
    # tbl = {}
    tbl_short = np.zeros((3,4))
    fig, ax = plt.subplots(3,4)
    for i, it1 in enumerate(dict_i.items()):
        vn, vi = it1 # pname name and bunch ray indices
        v = psc[vi][vn] # pick the initial phase space coordinates for the relevant indices
        for j, it2 in enumerate(dict(A=dphia, H=phih, V=dphiv, T=dphit).items()):
            fn, f = it2 # angle plane (A = 'all' = 3D) name and delta-deviation-angle
            print(vn, fn)
            par, err = fit_line(abs(v), abs(f[vi])) # computing the derivative delta-dev-angle/delta-offset
            ax[i,j].plot(v, f[vi], '.')
            xlab = '${}$'.format(var_dict[vn]); ylab = ylab_fun(plane_dict[fn], var_dict[vn])
            ax[i,j].set_xlabel(xlab); ax[i,j].set_ylabel(ylab)
            ax[i,j].ticklabel_format(style='sci', scilimits=(0,0), axis='both', useMathText=True)
            slp, slp_err = par[1], err[1]
            # tbl.update({(fn,vn):(slp, slp_err, slp_err/slp)})
            tbl_short[i,j] = slp
    return tbl_short        

def plot_pol(particle, diff=False):
    dat = particle.sp
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

class DecohMeasure:
    def __init__(self, particle):
        dat = particle.sp
        self._phi = dict(A=np.arccos(dot3d(dat)), H=np.arccos(dot2d(dat, 'H')), V=np.arccos(dot2d(dat, 'V')), T=np.arccos(dot2d(dat, 'T')))
        self._dm  = {name:angle.std(axis=1) for name, angle in phi.items()}
    @property
    def dm(self):
        return self._dm
    @property
    def angle(self):
        return self._phi
    def plot_net(self, ):
        pass

if __name__ == '__main__':
    path = lambda name: HOMEDIR+DATADIR+name.upper()+'/'
    # pro = Particle(path('proton'), 'proton')
    deu = Particle(path('deuteron'), 'deuteron')
    
    # pro.plot_spin(slice(1, None, 3), slice(0, None, 15), name='X')
    # deu.plot_spin(slice(1, None, 3), slice(0, None, 15), name='X')

