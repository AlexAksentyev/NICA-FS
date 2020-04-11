import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import HOMEDIR, Data, TSS, Polarization, fit_line
from pandas import DataFrame, ExcelWriter
from numpy.linalg import norm
#from scipy.optimize import curve_fit
from statsmodels.nonparametric.smoothers_lowess import lowess
from mpl_toolkits.mplot3d import Axes3D
import os
from glob import glob

def plot_spin(spdata, L=503, gamma=1.14):
    beta = np.sqrt(1 - 1/gamma**2)
    v = beta*3e8
    tau = L/v
    t = spdata['iteration'][:,0]*tau
    fig, ax = plt.subplots(3,1,sharex=True)
    ax[2].set_xlabel('t [sec]')
    for i, v in enumerate(['S_X','S_Y','S_Z']):
        lines = ax[i].plot(t, spdata[v])
        ax[i].set_ylabel(v)
        ax[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    return fig, ax, lines

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
    # loading data
    print("data from",  path)
    tss = TSS(path, 'MU.dat')
    pray = Data(path, 'PRAY.dat')
    sp = Data(path, 'TRPSPI.dat')
    ps = Data(path, 'TRPRAY.dat')
    pol = get_polarization()
    # making the spin plot
    jj = sp['EID'][:,0]==eid
    pids = [0, 1, 4]
    labels = get_spin_labels()
    print("plotting spin vector components")
    fsp, axsp, linsp = plot_spin(sp[jj][:, pids])
    axsp[0].set_title(name)
    plt.legend(linsp, labels)
    plt.savefig(path+'img/spin.png', dpi=450, bbox_inches='tight', pad_inches=.1)
    plt.close()
    # making the polarization plot
    print("plotting polarization")
    fpol, axpol = pol.plot(eid, 'sec')
    axpol.grid(axis='y')
    axpol.set_title(name)
    plt.savefig(path+'img/polarization.png', dpi=450, bbox_inches='tight', pad_inches=.1)
    plt.close()
    # making the TSS plot
    print("plotting TSS")
    ftss, axtss = tss.plot()
    axtss[0].set_title(name)
    plt.savefig(path+'img/tss.png', dpi=450, bbox_inches='tight', pad_inches=.1)
    plt.close()
    # making the plot of spin vector cpmponents' dependence on particle mean energy offset
    print("plotting spin vs gamma effective")
    spofffig, spoffax = spin_offset_analysis(sp, ps, eid)
    spoffax[0].set_title(name)
    plt.savefig(path+'img/mean_spin_vs_gamma.png', dpi=450, bbox_inches='tight', pad_inches=.1)
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
    gamma = 1.14
    L = 503
    beta = np.sqrt(1 - 1/gamma**2)
    v = beta*3e8
    tau = L/v
    jj = pol['EID']==eid
    t = pol['iteration'][jj]*tau
    p = pol['Value'][jj]
    fit = lowess(p[0:None:100], t[0:None:100])

    f, ax = pol.plot(eid)
    ax.grid(axis='y')
    ax.plot(fit[:,0], fit[:, 1], '-k')
    return f, ax

def spin_offset_analysis(spdata, psdata, eid):
    jj = spdata['EID'][:,0]==eid # SPD eid now is 1, b/c I didn't split the lattice map anywhere else
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
    jj = spdat['EID'][:,0]==eid
    sub_idx_rng = int(jj.sum()/num_of_subsets)
    title = 'at SPD' if eid==2 else 'at MPD'
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
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(r'$P_x$')
    ax.set_ylabel(r'$P_z$')
    ax.set_zlabel(r'$P_y$')


def plot_spin_3D(spdat, eid, zero=True):
    def plot_one(turn):
        if zero:
            ax.plot([0, S['S_X'][turn,0]], [0, S['S_Z'][turn,0]], [0, S['S_Y'][turn,0]], '--k')
            ax.plot([0,0], [0,0], [0,0], '*r')
        ax.plot(S['S_X'][turn,:], S['S_Z'][turn,:], S['S_Y'][turn,:], '.')
    title = 'at SPD' if eid==2 else 'at MPD'
    jj = spdat['EID'][:,0]==eid
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


def main(root):
    def get_axis(psi):
        sign = int(psi[-1]+'1')
        psi = np.deg2rad(float(psi[-1]+psi[4:-1]))
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
        axis = get_axis(psi)
        analysis(d, 1, name, axis)
        
    
if __name__ == '__main__':
    common = HOMEDIR+'data/REPORT/NON-FS/100kTURN/'
    main(common+'X-bunch/')
    main(common+'Y-bunch/')
    main(common+'D-bunch/')
    
    
