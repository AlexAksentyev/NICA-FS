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

def project_spin_nbar(spdata, tssdata, ftype='CO'): # either CO, or mean
                                        # DON'T USE MEAN, I only generate data for one turn here
   def make_nbar_seq(component, repnum):
       num_el = len(component)
       pick_i = np.unique(spdata['EID'][:,0])%num_el
       x = component[pick_i]
       x0 = x[0]
       x = x[1:] #np.append(x[1:], x[:2])[:-1]
       x = np.tile(x, repnum) # x is 1d; [1,2,3,..., 1, 2, 3, ..., 1, 2, 3,... ]
       return np.insert(x, 0, x0)
   def normalize(nbar):
       norm_nbar = np.sqrt(nbar['X']**2 + nbar['Y']**2 + nbar['Z']**2)
       if np.any(abs(norm_nbar-1))>1e-6:
           print('********** nbar norm is suspect! {}, {}'.format(norm_nbar.min(), norm_nbar.max()))
       return {lbl: nbar[lbl]/norm_nbar for lbl in ['X','Y','Z']}
   s = {lbl:spdata['S_'+lbl] for lbl in ['X','Y','Z']}
   ntrn = np.unique(spdata['iteration'][1:,0])[-1]
   if ftype=='CO':
       n = {lbl:make_nbar_seq(tssdata['N'+lbl][:,0], ntrn) for lbl in ['X','Y','Z']}
   elif ftype=='mean':
       n = {lbl:make_nbar_seq(np.mean(tssdata['N'+lbl], axis=1), ntrn) for lbl in ['X','Y','Z']}
   n = normalize(n)
   prod = {lbl: (s[lbl].T*n[lbl]).T for lbl in ['X','Y','Z']}
   it = spdata['iteration'][:,0]
   proj = prod['X']+prod['Y']+prod['Z']
   return proj

def polarization(sp_nb_proj):
    nray = sp_nb_proj.shape[1]
    return sp_nb_proj.sum(axis=1)/nray

class Polarization(Data):
    def __init__(self, spdata, tssdata):
        sp_proj = project_spin_nbar(spdata, tssdata, ftype='CO')
        nray = sp_proj.shape[1]
        pol = sp_proj.sum(axis=1)/nray
        it = spdata['iteration'][:,0]
        eid = spdata['EID'][:,0]
        self._data = np.array(list(zip(it, eid, pol)), dtype = [('iteration', int), ('EID', int), ('Value', float)])

    @property
    def co(self):
        return self._data
    def plot(self, eid, L=503, gamma=1.14):
        beta = np.sqrt(1 - 1/gamma**2)
        v = beta*3e8
        tau = L/v
        jj = self['EID']==eid
        y = self['Value'][jj]
        x = self['iteration'][jj]*tau
        par, err = fit_line(x, y)
        fig, ax = plt.subplots(1,1)
        ax.plot(x,y, '.')
        ax.plot(x, par[0] + x*par[1], '-r', label=r'$slp = {:4.2e} \pm {:4.2e}$ [u/sec]'.format(par[1], err[1]))
        ax.set_ylabel(r'$\sum_i(\vec s_i, \bar n_{})$'.format(eid))
        ax.set_xlabel('t [sec]')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
        ax.legend()
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

def plot_spin(spdata):
    fig, ax = plt.subplots(3,1,sharex=True)
    for i, var in enumerate(['S_X','S_Y','S_Z']):
        ax[i].plot(spdata[var], '--.')
        ax[i].set_ylabel(var)
        ax[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    return fig, ax

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
    
    
