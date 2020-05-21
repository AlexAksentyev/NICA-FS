import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from matplotlib import ticker
from analysis import HOMEDIR, Data, TSS, Polarization, fit_line, TAU, guess_freq, guess_phase, DAVEC
from pandas import DataFrame, ExcelWriter
from numpy.linalg import norm
from scipy.optimize import curve_fit
from statsmodels.nonparametric.smoothers_lowess import lowess
from mpl_toolkits.mplot3d import Axes3D
import os
from glob import glob

class NBAR:
    def __init__(self, path):
        var_dict = {'X':'NBAR1', 'Y':'NBAR2', 'Z':'NBAR3'}
        self._da = {lbl: DAVEC(path+var_dict[lbl]+'.da') for lbl in ['X','Y','Z']}

    def __getitem__(self, lbl):
        return self._da[lbl]

    def __call__(self, psvec, fun=lambda x:x):
        VARS = ['X','A','Y','B','T','D']
        return {lbl: fun(self[lbl](psvec[VARS])) for lbl in ['X','Y','Z']}
    def mean(self, psvec):
        return self(psvec, lambda x:np.mean(x))

def fit_model(x, y):
    i0 = y.mean()
    s0 = 0
    a0 = abs(y.max()-y.min())/2
    lam0 = 0
    f0 = guess_freq(x, y-y.mean())
    p0 = guess_phase(x, y)
    # print('guess freq: ', f0)
    # print('guess phase: ', p0)
    # print('guess amplitude', a0)
    model = lambda x, i,s,a,lam,f,p: i + s*x + a*np.exp(lam*x)*np.sin(2*np.pi*f*x + p)
    pest, pcov = curve_fit(model, x, y, p0=[i0,s0,a0,lam0,f0,p0])
    perr = np.sqrt(np.diag(pcov))
    # fig, ax = plt.subplots(1,1)
    # ax.plot(x, y, '-.')
    # ax.plot(x, model(x, *pest), '-r')
    pairs = list(zip(pest, perr))
    names = ['icpt','slp','ampl','pow', 'freq','phase']
    df = DataFrame(dict(zip(names, pairs)), index=['est','se'])
    return df#, fig, ax

def fitpar_analysis(spdata, psdata, fitpar='slp'):
    SPC = ['S_X','S_Y','S_Z']
    dm = psdata['D'].mean(axis=0)
    dms_total = dm.std()
    dm_total = dm.mean()
    fig, ax = plt.subplots(3,1, sharex=True)
    # ax[0].set_title('fit parameter plotted: '+fitpar)
    ax[2].set_xlabel(r'$\langle\delta\rangle$')
    for i in range(3):
        ax[i].set_ylabel(r'$\langle$'+ r'${}$'.format(SPC[i]) + r'$\rangle$: '+ fitpar)
        ax[i].ticklabel_format(axis='both', style='sci', scilimits=(0,0), useMathText=True)
        ax[i].grid()
    for j in range(spdata.data.shape[1]):
        ray = spdata[:,j]
        rayd = psdata[:,j]['D']
        dm = rayd.mean()
        if (dm-dm_total)>3*dms_total:
            print('delta energy outlier; pid ', j,
                      'delta excess: {:4.2e}'.format(dm),
                      'in sigmas: {:4.2e}'.format((dm-dm_total)/dms_total))
            continue
        ds = rayd.std()/np.sqrt(rayd.shape[0])
        t = ray['iteration']*TAU
        try:
            par = {lbl: fit_model(t, ray[lbl]) for lbl in SPC}
            for i, lbl in enumerate(SPC):
                if np.divide(par[lbl][fitpar]['se'], par[lbl][fitpar]['est'])>.5:
                    continue
                ax[i].errorbar(dm, par[lbl][fitpar]['est'], yerr=par[lbl][fitpar]['se'], xerr=ds, fmt='.b')
                # print('pid', j, 'component', lbl)
                # print(par[lbl])
        except:
            print('failed at pid', j)
    return fig, ax

def plot_spin(spdata):
    t = spdata['iteration'][:,0]*TAU
    mean_sp = {lbl: spdata[lbl].mean(axis=0) for lbl in ['S_X','S_Y','S_Z']}
    fig, ax = plt.subplots(3,1,sharex=True)
    ax[2].set_xlabel('t [sec]')
    for i, v in enumerate(['S_X','S_Y','S_Z']):
        lines = ax[i].plot(t, spdata[v])
        ax[i].set_ylabel(r'${}$'.format(v))
        ax[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    for i, v in enumerate(['S_X','S_Y','S_Z']):
        ax[i].hlines(mean_sp[v], t[0], t[-1], linewidths=2, ls='dashed')
    return fig, ax, lines

def plot_spin_1turn(spdata):
    fig, ax = plt.subplots(3,1,sharex=True)
    ax[2].set_xlabel('EID')
    for i, v in enumerate(['S_X', 'S_Y' , 'S_Z']):
        ax[i].set_ylabel(r'${}$'.format(v))
        ax[i].plot(spdata[v])
        ax[i].grid()
        ax[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    return fig, ax

def dotprod(s1, s2):
    prod = [s1[lbl]*s2[lbl] for lbl in ['S_X','S_Y','S_Z']]
    return np.sum(prod)

def norm(s):
    return np.sqrt(dotprod(s,s))

def analyze_spin(spdata):
    t = spdata['iteration'][:,0]*TAU
    fig, ax = plt.subplots(3,1,sharex=True)
    ax[2].set_xlabel('sec')
    for i, lab in enumerate(['S_X','S_Y','S_Z']):
        ax[i].set_ylabel(lab)
        ax[i].ticklabel_format(axis='y',style='sci', scilimits=(0,0),useMathText=True)
        ax[i].grid()
    for s in spdata.T:
        for i, v in enumerate(['S_X','S_Y','S_Z']):
            sc = s[v]
            fit = lowess(sc, t)
            par, err = fit_line(t, sc)
            ax[i].plot(t, sc, '.',
                           label='({:4.2e} $\pm$ {:4.2e}, {:4.2e} $\pm$ {:4.2e})'.format(par[0],
                                                                                             err[0],
                                                                                             par[1],
                                                                                             err[1]))
            ax[i].plot(t, par[0] + t*par[1], '-r')
            ax[i].plot(fit[:,0], fit[:,1], '-k')
            ax[i].legend()
    return fig, ax
        

def decoherence_derivative(spdata, psdata, tssdata, eid): # this is the same thing as the decoherence_derivative in analysis.py
                                        # but the angle is comouted in the axes: nbar--nbar-orthogonal
    # first of all, check that I have sufficient data for comupation (I need at least two turns)
    def dot3d(vdat, v0):
        return np.array([vdat[lbl]*v0[i] for i,lbl in enumerate(['S_X','S_Y','S_Z'])]).sum(axis=0)
    if len(np.unique(spdata['iteration']))<3: # need AT LEAST [0, 1, 2]
        print('insufficient data')
        return 1;
    else:
        # picking the nbar at EID location (single turn)
        jj = tssdata['EID'][:,0]==eid
        nbar = np.array([tssdata[lbl][jj,0] for lbl in ['NX', 'NY', 'NZ']]).flatten() # pick CO nbar
        nnbar = norm(nbar)
        if abs(nnbar-1) > 1e-6:
            print('nbar norm suspect! {}'.format(nnbar))
        # picking tracking data at EID location (multiple turns)
        jj = spdata['EID']==eid
        ntrn = np.unique(spdata['iteration'][:,0])[-1]
        spc = spdata[jj].reshape(ntrn, -1)
        psc = psdata[jj].reshape(ntrn, -1)
        print(' spc shape: {}'.format(spc.shape))
        ps0c = psdata[0].copy() # initial offsets
        #### here I compute the change to the angle
        #### between the particle's spin vector and the invariant spin axis after n-th turn
        ## this is deviation angle relative to the PREVIOUS deviation angle
        #dphi = np.diff(np.arccos(dot3d(spc, nbar)), axis=0) # computing the 3D dot product and deviation angle
        ## this is relative INITIAL angle
        phi = np.arccos(dot3d(spc, nbar)) # current angle between spin vector and nbar
        dphi = phi[1:] - phi[0] # how much it changed since 1st turn
        ##
        # minus the reference ray's angle change
        dphi -= dphi[:,0].repeat(dphi.shape[1]).reshape(-1,dphi.shape[1])
        print('dphi shape: {}'.format(dphi.shape))
        # dictionary containing the indices for the X,Y,D bunch particles
        #dict_i = {e:slice((i+1), None, 3) for i,e in enumerate(['X','Y','D'])}
        dict_i = {e: ps0[e]!=0 for e in ['X','Y','D']}
        var_dict = dict(X='x', Y='y', D=r'\delta')
        ylab_fun = lambda p,v: r'$\Delta\Theta_{}/\Delta {}$'.format(p,v)
        fig, ax = plt.subplots(1,3)
        for i, it1 in enumerate(dict_i.items()):
            vn, vi = it1 # pname name and bunch ray indices
            v = ps0c[vi][vn] # pick the initial phase space coordinates for the relevant indices
            #par, err = fit_line(abs(v), abs(dphi[vi])) # computing the derivative delta-dev-angle/delta-offset
            ax[i].plot(v, dphi[:,vi].T, '.') #psc[vn][1:,vi].T
            xlab = '${}$'.format(var_dict[vn]); ylab = ylab_fun('{3d}', var_dict[vn])
            ax[i].set_xlabel(xlab); ax[i].set_title(ylab)
            ax[i].grid()
            ax[i].ticklabel_format(style='sci', scilimits=(0,0), axis='both', useMathText=True)
        return dphi

def analysis(path, eid, name='', axis=[1,0,0]):
    def get_spin_labels():
        bunch = (path.split('/')[-3])[0]
        lbl = []
        for pid in pids:
            str_ = r'${}_0$ = {:4.2e}'.format(bunch.lower(), pray[0, pid][bunch])
            lbl.append(str_)
        return lbl
    def get_polarization():
        if axis!='nbar':
            pol = Polarization.on_axis(sp, axis)
        else:
            pol = Polarization.on_nbar(sp, tss)
        return pol
    def plot_spd():
        t = pol['iteration']*TAU
        spd = pol.spin_proj.std(axis=1)
        fit = lowess(spd[0:None:100], t[0:None:100])
        par, err = fit_line(t, spd)
        fig, ax = plt.subplots(1,1)
        ax.plot(t, spd, '.')
        ax.plot(fit[:,0], fit[:, 1], '-k')
        ax.plot(t, par[0]+par[1]*t, '-r', label=r'slp = {:4.2e} $\pm$ {:4.2e} [u/sec]'.format(par[1], err[1]))
        ax.set_xlabel('sec')
        ax.set_ylabel(r'$\sigma(\vec s_i, \bar n_{})$'.format(eid))
        ax.grid()
        ax.legend()
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
        return fig, ax
    ### loading data
    print("data from",  path)
    tss = TSS(path, 'MU.dat')
    pray = Data(path, 'PRAY.dat')
    sp = Data(path, 'TRPSPI.dat')
    # sp1 = Data(path, 'TRPSPI:ONE_TURN.dat')
    ps = Data(path, 'TRPRAY.dat')
    pol = get_polarization()
    ### making the spin one-turn plot
    # print("plotting spin for one turn")
    # sp1fig, sp1ax = plot_spin_1turn(sp1)
    # sp1ax[0].set_title(name)
    # plt.savefig(path+'img/spin_1turn.png', dpi=450, bbox_inches='tight', pad_inches=.1)
    # plt.close()
    ### making the spin_analysis plot
    print("spin analysis")
    fsa, axsa = analyze_spin(sp[:,1:150:50])
    axsa[0].set_title(name)
    plt.savefig(path+'img/spin_analysis.png', dpi=450, bbox_inches='tight', pad_inches=.1)
    plt.close()
    ### making the fitpar_analysis plots
    print("plotting spin vector component fitpar plots")
    for parname in ['icpt' ,'slp', 'freq', 'pow']:
        fsp, axsp = fitpar_analysis(sp, ps, parname)
        axsp[0].set_title(name)
        plt.savefig(path+'img/fitpar_{}.png'.format(parname), dpi=450, bbox_inches='tight', pad_inches=.1)
        plt.close()
    ### making the polarization plot
    print("plotting polarization")
    fpol, axpol = pol.plot(eid, 'sec')
    axpol.grid(axis='y')
    axpol.set_title(name)
    plt.savefig(path+'img/polarization.png', dpi=450, bbox_inches='tight', pad_inches=.1)
    plt.close()
    ### plotting spin projection dispersion
    spdfig, spdax = plot_spd()
    spdax.set_title(name)
    plt.savefig(path+'img/spin_proj_disp.png', dpi=450, bbox_inches='tight', pad_inches=.1)
    plt.close()
    ### making the TSS plot
    print("plotting TSS")
    ftss, axtss = tss.plot()
    axtss[0].set_title(name.split(":")[1].strip())
    plt.savefig(path+'img/tss.png', dpi=450, bbox_inches='tight', pad_inches=.1)
    plt.close()

def one_turn_analysis(path, name):
    sp = Data(path, 'TRPSPI.dat')
    fsp, axsp = plt.subplots(3,1,sharex=True)
    axsp[0].set_title(name)
    for i, v in enumerate(['S_X','S_Y','S_Z']):
        axsp[i].plot(sp[:,0][v])
        axsp[i].set_ylabel(v)
        axsp[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

def pol_analysis(pol, eid):
    jj = pol['EID']==eid
    t = pol['iteration'][jj]*TAU
    p = pol['Value'][jj]
    fit = lowess(p[0:None:100], t[0:None:100])
    f, ax = pol.plot(eid)
    ax.grid(axis='y')
    ax.plot(fit[:,0], fit[:, 1], '-k')
    return f, ax

def spin_offset_analysis(spdata, psdata, eid):
    try:
        jj = spdata['EID'][:,0]==eid # SPD eid now is 1, b/c I didn't split the lattice map anywhere else
    except:
        jj = slice(0, None)
    dm = psdata[jj]['D'].mean(0)
    sm = {lbl: spdata[jj][lbl].mean(0) for lbl in ['S_X','S_Y','S_Z']}
    fig, ax = plt.subplots(3,1, sharex=True)
    ax[2].set_xlabel(r'$\langle\delta\rangle$')
    for i, it in enumerate(sm.items()):
        name, data = it
        par, err = fit_line(dm, data)
        ax[i].plot(dm, data, '.')
        ax[i].plot(dm, par[0] + par[1]*dm, '--r')
        ax[i].grid()
        ax[i].set_ylabel(r'$\langle$'+r'${}$'.format(name) + r'$\rangle$')
        ax[i].ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    return fig, ax

def plot_pol_3D(spdat, eid, zero=True):
    num_of_subsets = 5
    try:
        jj = spdat['EID'][:,0]==eid
        sub_idx_rng = int(jj.sum()/num_of_subsets)
    except:
        jj = slice(0, None)
        sub_idx_rng = int(spdat.data.shape[0]/num_of_subsets)
    title = 'at SPD' #if eid==2 else 'at MPD' # data only taken at SPD
    axis = dict(X=[1,0,0], Y=[0,1,0], Z=[0,0,1])
    P = {n: Polarization.on_axis(spdat, x)['Value'][jj] for n, x in axis.items()}
    P0 = {n: P[n].mean() for n in ['X','Y','Z']}
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if zero:
        ax.plot([0, P0['X']], [0, P0['Z']], [0, P0['Y']], '--k')
        ax.plot([0,0], [0,0], [0,0], '*r')
    for i in range(num_of_subsets):
        ii = slice(i*(sub_idx_rng+1), sub_idx_rng*(i+1))
        ax.plot(P['X'][ii], P['Z'][ii], P['Y'][ii], label=i)
    plt.xlim((-.1, .1))
    plt.ylim((-1, 1))
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(r'$P_x$')
    ax.set_ylabel(r'$P_z$')
    ax.set_zlabel(r'$P_y$')

def ensemble_mean_svec(spdat):
    nray = spdat.shape[1]
    mean_sv = np.zeros(nray, dtype = list(zip(['X','Y','Z'], [float]*3)))
    for i, ray in enumerate(spdat.T):
        mean_sv[i] = ray['S_X'].mean(), ray['S_Y'].mean(), ray['S_Z'].mean()
    return mean_sv
    
def plot_svec_means_3D(spdat, pray, var='X', eid=1):
    num_vec = spdat.shape[1]
    pids = [int(e) for e in spdat['vector'][0]]
    x0 = pray[0, pids][var]
    try:
        jj = spdat['EID'][:,0]==eid
    except:
        jj = slice(0, None)
    title = 'at SPD'
    P0 = ensemble_mean_svec(spdat)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot([0,0], [0,0], [0,0], '*r')
    for i in range(num_vec):
        ax.plot([0, P0['X'][i]], [0, P0['Z'][i]], [0, P0['Y'][i]], '--k')
        ax.plot([0, P0['X'][i]], [0, P0['Z'][i]], [0, P0['Y'][i]], '.',
                    label='{:4.2f} mm'.format(x0[i]*1e3))
    # plt.xlim((-.1, .1))
    # plt.ylim((-1, 1))
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(r'$S_x$')
    ax.set_ylabel(r'$S_z$')
    ax.set_zlabel(r'$S_y$')

def plot_spin_3D(spdat, eid, zero=True):
    def plot_one(turn):
        if zero:
            ax.plot([0, S['S_X'][turn,0]], [0, S['S_Z'][turn,0]], [0, S['S_Y'][turn,0]], '--k')
            ax.plot([0,0], [0,0], [0,0], '*r')
        ax.plot(S['S_X'][turn,:], S['S_Z'][turn,:], S['S_Y'][turn,:], '.')
    title = 'at SPD' #if eid==2 else 'at MPD' # only outputting data at SPD
    try:
        jj = spdat['EID'][:,0]==eid
    except:
        jj = slice(0, None)
    S = {n: spdat[jj][n] for n in ['S_X','S_Y','S_Z']}
    nturn = S['S_X'].shape[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel(r'$S_x$')
    ax.set_ylabel(r'$S_z$')
    ax.set_zlabel(r'$S_y$')
    turn = 0
    while (turn>=0)*(turn<nturn):
        plot_one(turn)
        turn = int(input('next turn '))
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))

def ensemble_nbar(nbar, psdat):
    enbar = np.zeros(psdat.shape, dtype = list(zip(['X','Y','Z'], [float]*3)))
    for i, ray in enumerate(psdat.T):
        x = nbar(ray)
        enbar[:,i]['X'] = x['X']
        enbar[:,i]['Y'] = x['Y']
        enbar[:,i]['Z'] = x['Z']
    return enbar

def plot_ensemble_nbar(enbar, zero=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('at SPD')
    ax.set_xlabel(r'$\bar n_x$')
    ax.set_ylabel(r'$\bar n_z$')
    ax.set_zlabel(r'$\bar n_y$')
    for n0 in enbar.T:
        ax.plot(n0['X'], n0['Z'], n0['Y'], '.')
    if zero:
        n0m = {lbl: enbar[lbl].mean(axis=0) for lbl in ['X','Y','Z']}
        for i in range(n0m['X'].shape[0]):
            ax.plot([0, n0m['X'][i]], [0, n0m['Z'][i]], [0, n0m['Y'][i]], '--k')
            ax.plot([0, n0m['X'][i]], [0, n0m['Z'][i]], [0, n0m['Y'][i]], '.k')
        

    
def ensemble_mean_nbar(nbar, psdat):
    nray = psdat.shape[1]
    mean_nbar = np.zeros(nray, dtype = list(zip(['X','Y','Z'], [float]*3)))
    for i, ray in enumerate(psdat.T):
        x = nbar.mean(ray)
        mean_nbar[i] = x['X'], x['Y'], x['Z']
    return mean_nbar

def plot_nbar_3D(mean_nbar):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('at SPD')
    ax.set_xlabel(r'$\bar n_x$')
    ax.set_ylabel(r'$\bar n_z$')
    ax.set_zlabel(r'$\bar n_y$')
    for N0 in mean_nbar:
        ax.plot([0, N0['X']], [0, N0['Z']], [0, N0['Y']], '--k')
        ax.plot([0, N0['X']], [0, N0['Z']], [0, N0['Y']], '.')

def plot_3D_both(mean_nbar, mean_svec, svdata=None, zero=True):
    if svdata==None:
        zero = True
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('at SPD')
    if svdata != None:
        for ray in svdata.T:
            ax.plot(ray['S_X'], ray['S_Z'], ray['S_Y'], '.')
    if zero == True:
        ax.set_xlabel(r'$\bar n_x~[s_x]$')
        ax.set_ylabel(r'$\bar n_z~[s_z]$')
        ax.set_zlabel(r'$\bar n_y~[s_y]$')
        for N0 in mean_nbar:
            ax.plot([0, N0['X']], [0, N0['Z']], [0, N0['Y']], '-r')
            ax.plot([0, N0['X']], [0, N0['Z']], [0, N0['Y']], '.r')
        for S0 in mean_svec:
            ax.plot([0, S0['X']], [0, S0['Z']], [0, S0['Y']], '-k')
            ax.plot([0, S0['X']], [0, S0['Z']], [0, S0['Y']], '.k')
    else:
        ax.set_xlabel(r'$s_x$')
        ax.set_ylabel(r'$s_z$')
        ax.set_zlabel(r'$s_y$')

def plot_mu_3D(path, psdat, zvar, xvar='T', yvar='D', fmt='-.'):
    var_dict = {'NU':'NU', 'NX':'NBAR1', 'NY':'NBAR2', 'NZ':'NBAR3'}
    zlab = {'NU':r'$\nu_s$', 'NX':r'$\bar n_x$', 'NY': r'$\bar n_y$', 'NZ': r'$\bar n_z$'}
    xylab = {'X': 'X [mm]', 'A': 'A [mrad]',
                'Y': 'Y [mm]', 'B': 'B [mrad]',
                'T': r'$\ell$ [mm]', 'D': r'$\delta$ [%]'}
    xftr = 1e3 if xvar!='D' else 1e2
    yftr = 1e3 if yvar!='D' else 1e2
    davec = DAVEC(path+var_dict[zvar]+'.da')
    VARS = ['X','A','Y','B','T','D']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(xylab[xvar],labelpad=10)
    ax.set_ylabel(xylab[yvar],labelpad=10)
    ax.set_zlabel(zlab[zvar], labelpad=20, rotation=90)
    for ray in psdat.T:
        mudat = davec(ray[VARS])
        ax.plot(ray[xvar]*xftr, ray[yvar]*yftr, mudat, fmt)
    formatter = ticker.ScalarFormatter(useMathText=True, useOffset=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-0,0))
    ax.w_zaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax.w_yaxis.set_major_formatter(formatter)
    ax.w_xaxis.set_major_formatter(formatter)
    return fig, ax

def mu_3D_analysis(path, name):
    ps = Data(path, 'TRPRAY:MAIN.dat')
    for var in ['NU', 'NZ']:
        fig, ax = plot_mu_3D(path, ps[0:None:45, 1:150:25], var, fmt='-')
        ax.set_title(name)
        plt.savefig(path+'img/%s_3D_TD.png' %var, dpi=450)#, bbox_inches='tight', pad_inches=.3)
        plt.close()

def depol_sources_analysis(path, name):
    sp1 = Data(path, 'TRPSPI:ONE_TURN.dat')
    dispS = {lbl: sp1[lbl].std(axis=1) for lbl in ['S_X','S_Y','S_Z']}
    eid = sp1['EID'][:,0]
    fig, ax = plt.subplots(3,1,sharex=True)
    ax[2].set_xlabel('EID')
    for i, lbl in enumerate(['S_X','S_Y', 'S_Z']):
        ax[i].plot(eid, dispS[lbl], '--.')
        ax[i].set_ylabel(r'$%s$' % lbl)
        ax[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    return fig, ax


def main(root):
    def get_axis(psi):
        try:
            sign = int(psi[-1]+'1')
            psi = np.deg2rad(float(psi[-1]+psi[4:-1]))
        except:
            return [0,0,1]
        if psi!=0:
            axis = [0, np.sin(psi), np.cos(psi)]
        else:
            axis = [0, 0, sign]
        return axis
    dir_list = glob(root+'*/')
    for d in dir_list:
        if not os.path.exists(d+'img/'):
            os.makedirs(d+'img/')
        bunch, psi = d.split('/')[-3:-1]
        name = ': '.join([bunch, psi])
        print(name)
        axis = get_axis(psi)
        analysis(d, 1, name, axis)
        # fig, ax = depol_sources_analysis(d, name)
        # plt.savefig(d+'img/spin_disp_1turn.png', dpi=450, bbox_inches='tight', pad_inches=.1)
        # plt.close()
        # mu_3D_analysis(d, name)
        
    
if __name__ == '__main__':
    common = HOMEDIR+'data/REPORT/PROTON/BENDS24/3MTURN/'
    # main(common+'X-bunch/')
    # main(common+'Y-bunch/')
    # main(common+'D-bunch/')
    # common = HOMEDIR+'data/REPORT/DEUTERON/BENDS24/12MTURN/'
    # main(common+'X-bunch/')
    # main(common+'Y-bunch/')
    # main(common+'D-bunch/')
    
    
