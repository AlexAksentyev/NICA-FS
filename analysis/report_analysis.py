import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import HOMEDIR, Data, TSS
from pandas import DataFrame, ExcelWriter
from numpy.linalg import norm
from scipy.optimize import curve_fit

def fit_line(x,y): # this is used for evaluating the derivative
    line = lambda x,a,b: a + b*x
    popt, pcov = curve_fit(line, x, y)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

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

def analysis(path, name=''):
    print("data from",  path)
    tss = TSS(path, 'MU.dat')
    sp = Data(path, 'TRPSPI.dat')
    sp_nb_proj = project_spin_nbar(sp, tss)
    pol = polarization(sp_nb_proj)
    fsp, axsp = plot_spin(sp[0:None:1000, [0, 1]])
    fpol, axpol = plot_pol(abs(pol[0:None:1000]))
    ftss, axtss = tss.plot()
    axsp[0].set_title(name)
    axpol.set_title(name)
    axtss[0].set_title(name)

def main():
    common = HOMEDIR+'data/SPINTUNE/100kTURN/'
    dirs = {'NO-NAVI': common+'NO_NAVIG/'
                ,'SPD-0': common+'NAVIG-SPD-0/'
                ,'SPD-90': common+'NAVIG-SPD-90/'
                }
        
    for name, path in dirs.items():
        analysis(path, name)
    
if __name__ == '__main__':
    main()
    
    
