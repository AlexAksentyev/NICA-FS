import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import HOMEDIR, Data, TSS, Polarization
from pandas import DataFrame, ExcelWriter
from numpy.linalg import norm
from scipy.optimize import curve_fit

def plot_spin(spdata, L=503, gamma=1.14):
    beta = np.sqrt(1 - 1/gamma**2)
    v = beta*3e8
    tau = L/v
    t = spdata['iteration'][:,0]*tau
    fig, ax = plt.subplots(3,1,sharex=True)
    ax[2].set_xlabel('t [sec]')
    for i, v in enumerate(['S_X','S_Y','S_Z']):
        ax[i].plot(t, spdata[v])
        ax[i].set_ylabel(v)
        ax[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
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
        dict_i = {e:slice((i+1), None, 3) for i,e in enumerate(['X','Y','D'])}
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
            ax[i].ticklabel_format(style='sci', scilimits=(0,0), axis='both', useMathText=True)
        return dphi

def analysis(path, eid, name=''):
    print("data from",  path)
    tss = TSS(path, 'MU.dat')
    sp = Data(path, 'TRPSPI.dat')
    pol = Polarization.on_axis(sp, [1,0,0])#tss)
    jj = sp['EID'][:,0]==eid
    fsp, axsp = plot_spin(sp[jj][:, [0, 1]])
    fpol, axpol = pol.plot(eid, 'turn')
    ftss, axtss = tss.plot()
    axsp[0].set_title(name)
    axpol.set_title(name)
    axtss[0].set_title(name)

def one_turn_analysis(path, name):
    sp = Data(path, 'TRPSPI.dat')
    fsp, axsp = plt.subplots(3,1,sharex=True)
    axsp[0].set_title(name)
    for i, v in enumerate(['S_X','S_Y','S_Z']):
        axsp[i].plot(sp[:,0][v])
        axsp[i].set_ylabel(v)
        axsp[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

def main(dirs):        
    for name, path in dirs.items():
        analysis(path, 2, name)

def main_one_turn(dirs):
    for name, path in dirs.items():
        one_turn_analysis(path, name)
    
if __name__ == '__main__':
    common = HOMEDIR+'data/SPINTUNE_VARIED/100kTURN/RADIAL/'
    dirs = {'NO-NAVI': common+'NO_NAVIG/'
                ,'SPD-5': common+'NAVIG-SPD-5/'
                ,'SPD-4': common+'NAVIG-SPD-4/'
                }
    main(dirs)
    
    
