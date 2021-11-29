import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import load_data, Polarization, guess_freq, guess_phase
import numpy.lib.recfunctions as rfn
from scipy.optimize import curve_fit

LATTICE = 'SECOND-ST'
ENERGY = '130' # MeV
DATDIR = 'NAVI-VARI-continuous'
Wcyc = .28494828977e6 # cyclotron frequency [Hz = rev/sec]
spin_tune = 1.5e-2 # spin rotations per beam revolution
st_rps = 2*np.pi*spin_tune*Wcyc # spin tune [rad/sec]
st_dps = np.rad2deg(st_rps) # spin tune [deg/sec]

deg_per_sec = lambda deg_per_turn: deg_per_turn*Wcyc

def dotprod(s0, s1):
    return np.sum([s0['S_'+lbl]*s1['S_'+lbl] for lbl in ['X','Y','Z']], axis=0)

def spin_disp(spdat):
    nray = spdat.shape[1]
    s0 = spdat[:,0].repeat(nray-1)
    s0.shape = (-1, nray-1)
    cos_phi = dotprod(s0, spdat[:,1:]) # spdat columns are unit-vectors
    disp = cos_phi.std(axis=1)
    return disp

def load(case, mrkr='PSI0spin=PSInavi'):
    folder = '../data/'+LATTICE+'/'+ENERGY+'MeV/'+DATDIR+'/RATE_'+case+'/'
    dat = load_data(folder, 'TRPRAY:{}.dat'.format(mrkr))
    spdat = load_data(folder, 'TRPSPI:{}.dat'.format(mrkr))
    muarr = load_data(folder, 'TRMUARR:{}.dat'.format(mrkr))
    # rate = np.diff(np.rad2deg(np.arcsin(muarr['NY'][:,0])))
    # print('psi <rate-of-change> = ', rate.mean(), '[deg/turn]')
    return dat, spdat, muarr

def plot_muarr(muarr):
    fig1, ax1 = plt.subplots(4,1,sharex=True)
    ax1[3].set_xlabel('TURN #')
    for i, var in enumerate(['U', 'X', 'Y', 'Z']):
        ax1[i].plot(muarr['TURN'], muarr['N'+var], '-')
        ax1[i].ticklabel_format(style='sci',scilimits=(0,0),useMathText=True)
        if i>0:
            ax1[i].set_ylabel(r'$\bar n_{}$'.format(var))
        else:
            ax1[i].set_ylabel(r'$\nu$')
    return fig1, ax1

def plot_spin(spdat):
    fig3, ax3 = plt.subplots(3,1,sharex=True)
    s0 = spdat[:,0] # reference spin vector
    ax3[2].set_xlabel('TURN #')
    for i, var in enumerate(['X', 'Y', 'Z']):
        ax3[i].plot(spdat[:,1:]['TURN'], spdat[:,1:]['S_'+var], '--')
        ax3[i].plot(s0['TURN'], s0['S_'+var], '-k')
        ax3[i].set_ylabel(r'$\vec s_{}$'.format(var))
    return fig3, ax3

def plot_phase(dat):
    fig2, ax2 = plt.subplots(3,1)
    ax2[0].plot(dat[:,:3]['X']*1e3, dat[:,:3]['A']*1e3)
    ax2[0].set_xlabel('X [mm]'); ax2[0].set_ylabel('A [mrad]')
    ax2[1].plot(dat[:,:3]['Y']*1e3, dat[:,:3]['B']*1e3)
    ax2[1].set_xlabel('Y [mm]'); ax2[1].set_ylabel('B [mrad]')
    ax2[2].plot(dat[:,:3]['T']*1e3, dat[:,:3]['D'])
    ax2[2].set_xlabel('T [mm]'); ax2[2].set_ylabel('D [unit]')
    for i in range(3):
        ax2[i].ticklabel_format(style='sci',scilimits=(0,0),useMathText=True)
    return fig2, ax2

def plot_spin_nbar(spdat, muarr):
    proj = pol_on_nbar(spdat, muarr)
    fig1, ax1 = plt.subplots(3,1, sharex=True)
    muarr0 = muarr[:,0] # muarr on the closed orbit
    ax1[2].set_xlabel('time [sec]')
    for i, var in enumerate(['X', 'Y', 'Z']):
        # svec
        ax1[i].plot(spdat['TURN']/Wcyc, spdat['S_'+var], '-k')
        # nbar (on the CO)
        ax1[i].plot(muarr0['TURN']/Wcyc, muarr0['N'+var], '--r', label=r'$\bar n^{CO}$')
        # metadata
        ax1[i].set_ylabel(r'$\vec s_{}$'.format(var))
        ax1[i].ticklabel_format(style='sci',scilimits=(0,0),useMathText=True)
        ax1[i].legend()
    fig2, ax2 = plt.subplots(1,1)
    ax2.plot(spdat['S_Z'], spdat['S_Y'], '-k')
    ax2.plot(muarr0['NZ'],muarr0['NY'], '--r',label=r'$\bar n^{CO}$')
    ax2.legend()
    ax2.set_xlabel(r'$\vec s_Z$'); ax2.set_ylabel(r'$\vec s_Y$')
    return fig1, ax1, fig2, ax2

def spin_analysis(case, mrkr):  # produces spin precession frequency estimations for further spin-tune-dispersion analysis
    folder = '../data/'+LATTICE+'/'+ENERGY+'MeV/'+DATDIR+'/RATE_'+case+'/'
    spdat = load_data(folder, 'TRPSPI:{}.dat'.format(mrkr))
    nray = spdat.shape[1]
    omega = np.zeros(nray, dtype = [('xpct', float), ('se', float)])
    fun = lambda x, w,p: np.sin(w*x+p)
    t = spdat[:,0]['TURN']/Wcyc
    figspa, axspa = plt.subplots(1,1)
    for j, p in enumerate(spdat.T):
        sig = p['S_Y']
        try:
            w0 = guess_freq(t, sig)*2*np.pi # in [rad/sec]
            ph0 = guess_phase(t, sig)       # in [rad]
        except:
            w0 = 0; ph0 = 0
        print(w0,ph0)
        popt, pcov = curve_fit(fun, t, sig, p0=[w0, ph0])
        perr = np.sqrt(np.diag(pcov))
        omega[j] = popt[0], perr[0]
        axspa.plot(t, p['S_Y'], '-k')
        axspa.plot(t, fun(t, *popt), '--r')
    return omega # in [rad/sec]

def pol_on_nbar(spdat, muarr): # compute beam polarization as spin-projection on n-bar
    def shape_up(elem):
        res = np.repeat(elem, nray)
        res.shape = (-1, nray)
        return res
    nray = spdat.shape[1]
    nturn = muarr.shape[0]
    s = {lbl:spdat['S_'+lbl] for lbl in ('X','Y','Z')}
    n0 = {lbl:shape_up(muarr['N'+lbl][:,0]) for lbl in ('X','Y','Z')}
    proj = np.zeros(nturn, dtype = [('TURN',int)]+list(zip(['X','Y','Z','TOT', 'TIME'], [float]*5)))
    for var in ['X','Y','Z']:
        proj[var] = np.sum(s[var][1:,:]*n0[var], axis=1)/nray
    proj['TOT'] = proj['X']+proj['Y']+proj['Z']
    proj['TURN'] = muarr['TURN'][:,0]
    proj['TIME'] = proj['TURN']/Wcyc # Wcyc -- the cyclotron frequency -- a constant defined above
    return proj

def process(case_name, mrkr='PSI0spin=PSInavi'):
    dat, spdat, muarr = load(case_name, mrkr)
    step = int(dat[1,0]['TURN'])
    if True:      # svec + nbar plot
        fig1, ax1, fig2, ax2  = plot_spin_nbar(spdat, muarr)
        ax1[0].set_title(r'$\dot \psi = $ {} [deg/{}-switch]'.format(case_name,step))
        fig1.savefig('../img/'+case_name+'-SVEC+NBAR-plot.png', dpi=450, bbox_inches='tight', pad_inches=.1)
        ax2.set_title(r'$\dot \psi = $ {} [deg/{}-switch]'.format(case_name,step))
        fig2.savefig('../img/'+case_name+'-SVEC+NBAR-circ-plot.png', dpi=450, bbox_inches='tight', pad_inches=.1)
    if False:       # phase space plot
        fig3, ax3 = plot_phase(dat)
        ax3[0].set_title(r'$\dot \psi = $ {} [deg/{}-switch]'.format(case_name,step))
        fig3.savefig('../img/'+case_name+'-PHASESPACE-plot.png', dpi=450, bbox_inches='tight', pad_inches=.1)
    if True:     # polarization plot
        P = pol_on_nbar(spdat, muarr)
        fig4, ax4 = plt.subplots(1,1)
        ax4.plot(P['TIME'], P['TOT']); ax4.grid()
        ax4.set_title(r'$\dot \psi = $ {} [deg/{}-switch]'.format(case_name,step))
        ax4.set_xlabel('time [sec]'); ax4.set_ylabel(r'$\sum\vec s\cdot \bar n^{CO}$')
        ax4.ticklabel_format(style='sci',scilimits=(0,0),useMathText=True)
        fig4.savefig('../img/'+case_name+'-POLARIZATION-plot.png', dpi=450, bbox_inches='tight', pad_inches=.1)
    return P

def main(case_names, mrkr='PSI0spin=PSInavi'):
    proj_dict = {}
    W_dict = {}
    for i, case in enumerate(case_names):
        proj = process(case, mrkr)
        omega = spin_analysis(case, mrkr)
        proj_dict.update({case:proj})
        W_dict.update({case:omega})
    return proj_dict, W_dict

if __name__ == '__main__':
    vals = ['1']
    pows = ['1/LONG']
    case_names = np.array([[x+'E'+y for x in vals] for y in pows]).flatten()
    # case_names = ['30', '60', '90']
    proj_dict, W_dict = main(case_names, 'nu-free')
    rates = [deg_per_sec(float(e)) for e in case_names]
    adiabaticity = st_dps/rates
