import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import HOMEDIR, Particle, tick_labels
from pandas import DataFrame, ExcelWriter
import spintune as st
from numpy.linalg import norm
from scipy.optimize import curve_fit

def fit_line(x,y): # this is used for evaluating the derivative
    line = lambda x,a,b: a + b*x
    popt, pcov = curve_fit(line, x, y)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

class ExtPart(Particle): # includes TSS data
    def __init__(self, path, name):
        Particle.__init__(self, path, name) # loading tracking data, enabling its plotting
        self._mu, _ = st.load_tss(path+'MU.dat') # loading TSS data
        self.__spin_nbar_prod() # computing the spin vectors projections on the CO nbar
        self._nbar_angle = np.arccos(self._sp_nb_proj)
        self._pol = self._sp_nb_proj.sum(axis=1)/self._sp.shape[1]

    @property
    def nu(self):
        return self._mu['NU']
    @property
    def nx(self):
        return self._mu['NX']
    @property
    def ny(self):
        return self._mu['NY']
    @property
    def nz(self):
        return self._mu['NZ']
    @property
    def polarization(self):
        return self._pol
    
    def plot_pol(self, L=503, gamma=1.14):
        beta = np.sqrt(1 - 1/gamma**2)
        v = beta*3e8
        tau = L/v
        y = self._pol
        x = np.arange(len(y))
        par, err = fit_line(x, y)
        fig, ax = plt.subplots(1,1)
        ax.plot(x,y, '--.')
        ax.plot(x, par[0] + x*par[1], '-r', label=r'$slp = {:4.2e} \pm {:4.2e}$'.format(par[1], err[1]))
        ax.set_ylabel(r'$\sum_i(\vec s_i, \bar n)$')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
        ax.legend()
        return fig, ax

    def plot_tss(self, fun=lambda x: x[:, 0]):
        norm1 = np.sqrt(self.ny**2 + self.nz**2)
        sin_psi = self.ny/norm1
        psi = np.rad2deg(np.arcsin(sin_psi))
        fig, ax = plt.subplots(3,1, sharex=True)
        ax[0].plot(fun(self._mu['NU']))
        ax[0].set_ylabel(r'$f(\nu_s)$')
        ax[1].set_ylabel(r'$f(\bar n_{\alpha})$')
        for v in ['NX','NY','NZ']:
            ax[1].plot(fun(self._mu[v]), label=v)
        ax[1].legend()
        ax[2].plot(fun(psi))
        ax[2].set_ylabel(r'$\angle(\bar n,\vec v)$ [deg]')
        for i in range(3):
            ax[i].ticklabel_format(axis='both', style='sci', scilimits=(0,0), useMathText=True)
            ax[i].grid(axis='x')
        return fig, ax

    def __spin_nbar_prod(self, ftype='CO'): # either CO, or mean
                                        # DON'T USE MEAN, I only generate data for one turn here
        def make_nbar_seq(component, repnum):
            num_el = len(component)
            pick_i = np.unique(self._sp['EID'][:,0])%num_el
            global x
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
        s = {lbl:self._sp['S_'+lbl] for lbl in ['X','Y','Z']}
        global ntrn, n
        ntrn = np.unique(self._sp['iteration'][1:,0])[-1]
        global n
        if ftype=='CO':
            n = {lbl:make_nbar_seq(self._mu['N'+lbl][:,0], ntrn) for lbl in ['X','Y','Z']}
        elif ftype=='mean':
            n = {lbl:make_nbar_seq(np.mean(self._mu['N'+lbl], axis=1), ntrn) for lbl in ['X','Y','Z']}
        n = normalize(n)
        prod = {lbl: (s[lbl].T*n[lbl]).T for lbl in ['X','Y','Z']}
        self._sp_nb_proj = prod['X']+prod['Y']+prod['Z']
        

    def decoherence_derivative(self, eid): # this is the same thing as the decoherence_derivative in analysis.py
                                        # but the angle is comouted in the axes: nbar--nbar-orthogonal
    # first of all, check that I have sufficient data for comupation (I need at least two turns)
        def dot3d(vdat, v0):
            return np.array([vdat[lbl]*v0[i] for i,lbl in enumerate(['S_X','S_Y','S_Z'])]).sum(axis=0)
        if len(np.unique(self._sp['iteration']))<3: # need AT LEAST [0, 1, 2]
            print('insufficient data')
            return 1;
        else:
            # picking the nbar at EID location (single turn)
            jj = self._mu['EID'][:,0]==eid
            nbar = np.array([self._mu[lbl][jj,0] for lbl in ['NX', 'NY', 'NZ']]).flatten() # pick CO nbar
            nnbar = norm(nbar)
            if abs(nnbar-1) > 1e-6:
                print('nbar norm suspect! {}'.format(nnbar))
            # picking tracking data at EID location (multiple turns)
            jj = self._sp['EID']==eid
            ntrn = np.unique(self._sp['iteration'][:,0])[-1]
            spc = self._sp[jj].reshape(ntrn, -1)
            psc = self._ps[jj].reshape(ntrn, -1)
            print(' spc shape: {}'.format(spc.shape))
            ps0c = self._ps[0].copy() # initial offsets
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
            

def analyze_decoherence(data_dir, particle_name):
    path = HOMEDIR + data_dir
    dummy = data_dir.split("/")[-2]
    pcl = Particle(path, particle_name)
    der = decoherence_derivative(pcl.sp, pcl.ps)
    plt.savefig('img/decoherence_table-{}-{}'.format(particle_name, dummy),
                       bbox_inches = 'tight', pad_inches = 0.1,
                       dpi=600)
    return der

def make_frame(data_table): # cf. analysis.py: decoherence_derivative for 'columns,' 'index'
    return DataFrame(data_table, columns=['A','H','V','T'], index = ['X','Y','D'])

def tss_analysis(data_dir, name):
    data, case = st.load_tss(HOMEDIR+data_dir+'MU.dat')
    fig, ax = st.plot(data, fun=lambda x: x[:,0])
    ax[0].set_title(name)


def main(dirs):
    # getting the decoherence derivatives tables when the spin navogators are off and when they are on
    ders = {}
    for name, directory in dirs.items():
        ders.update({name: make_frame(analyze_decoherence(directory, 'deuteron'))})
        tss_analysis(directory, name)

    # outputting data
    with ExcelWriter('tbl/derivatives_table.xlsx') as writer:
        for name, frame in ders.items():
            frame.to_excel(writer, sheet_name = name)

if __name__ == '__main__':
    common = 'data/SPINTUNE/1MTURN/'
    dirs = {'NO-NAVI': common+'NO_NAVIG/',
                'SPD-0': common+'NAVIG-SPD-0/',
                'SPD-90': common+'NAVIG-SPD-90/'}
        
    deu = {name: ExtPart(HOMEDIR+d, 'deuteron') for name, d in dirs.items()}
    for name, d in deu.items():
        fig, ax = d.plot_tss()
        ax[0].set_title(name)
        fig, ax = d.plot_spin(pcl_ids=[0, 1], name=name)
        ax[0].set_title(name)
    
    
