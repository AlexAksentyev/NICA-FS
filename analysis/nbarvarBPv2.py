import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt; plt.ion()
from analysis import load_data, Polarization, guess_freq, guess_phase
import numpy.lib.recfunctions as rfn
from scipy.optimize import curve_fit

mpl.rcParams['font.size']=14

LATTICE = 'BYPASS_SEX_wRC'
DATDIR = 'continuous'
Wcyc = .2756933208648683e6 # cyclotron frequency [Hz = rev/sec]
TAU = 1/Wcyc

rate_per_sec = lambda rate_per_turn: rate_per_turn*Wcyc

def dotprod(s0, s1):
    return np.sum([s0['S_'+lbl]*s1['S_'+lbl] for lbl in ['X','Y','Z']], axis=0)

def spin_disp(spdat):
    nray = spdat.shape[1]
    s0 = spdat[:,0].repeat(nray-1)
    s0.shape = (-1, nray-1)
    cos_phi = dotprod(s0, spdat[:,1:]) # spdat columns are unit-vectors
    disp = cos_phi.std(axis=1)
    return disp

def load(case, mrkr='resflip'):
    folder = '../data/'+LATTICE+'/ADIABATICITY/'+DATDIR+'/RATE_'+case+'/SHORT/'
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

def plot_SPN(spdat, pid=0):
    sp=spdat[:,pid]
    t=sp['TURN']*TAU
    
    fig, ax = plt.subplots(3,1,sharex=True)
    ax[0].set_title('pid = {:d}'.format(pid))
    for i, var in enumerate(['S_X','S_Y','S_Z']):
        ax[i].plot(t, sp[var])
        ax[i].set_ylabel(var)
    ax[2].set_xlabel('t [sec]')
    return fig

def plot_PS(phdat, pid=0):
    ph=phdat[:,pid]
    t = ph['TURN']*TAU

    fig = plt.figure()
    gs = fig.add_gridspec(2,3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_title('pid = {:d}'.format(pid))
    ax1.plot(ph['X']*1000, ph['A']*1000, '.')
    ax1.set_xlabel('X [mm]'); ax1.set_ylabel('A [mrad]')
    ax2.plot(ph['Y']*1000, ph['B']*1000, '.')
    ax2.set_xlabel('Y [mm]'); ax2.set_ylabel('B [mrad]')
    ax3.plot(ph['T'], ph['D'], '.')
    ax3.set_xlabel('T'); ax3.set_ylabel('D')
    for ax in (ax1, ax2, ax3):
        ax.ticklabel_format(style='sci', scilimits=(0,0),useMathText=True,axis='both')
    ax4.plot(t, ph['X']*1000, label='X')
    ax4.plot(t, ph['Y']*1000, label='Y')
    ax4.set_xlabel('t [sec]'); ax4.set_ylabel('trans. coord')
    ax4.legend()
    return fig, ax4

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
        ax1[i].ticklabel_format(style='sci',axis='x', scilimits=(0,0),useMathText=True)
        ax1[i].legend()
    fig2, ax2 = plt.subplots(1,1)
    ax2.plot(spdat['S_Z'], spdat['S_Y'], '-k')
    ax2.plot(muarr0['NZ'],muarr0['NY'], '--r',label=r'$\bar n^{CO}$')
    ax2.legend()
    ax2.set_xlabel(r'$\vec s_Z$'); ax2.set_ylabel(r'$\vec s_Y$')
    return fig1, ax1, fig2, ax2

def spin_analysis(case, mrkr):  # produces spin precession frequency estimations for further spin-tune-dispersion analysis
    folder = '../data/'+LATTICE+'/ADIABITICITY/'+DATDIR+'/RATE_'+case+'/SHORT/'
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

def pol_on_z(spdat):
    def shape_up(elem):
        res = np.repeat(elem, nray)
        res.shape = (-1, nray)
        return res
    nturn, nray = spdat.shape
    s = {lbl:spdat['S_'+lbl] for lbl in ('X','Y','Z')}
    ez = np.zeros(spdat.shape, dtype=list(zip(['X','Y','Z'],[float]*3)))
    ez['Z'] = np.ones(spdat.shape)
    proj = np.zeros(nturn, dtype = [('TURN',int)]+list(zip(['X','Y','Z','TOT', 'TIME'], [float]*5)))
    for var in ['X','Y','Z']:
        proj[var] = np.sum(s[var]*ez[var], axis=1)/nray
    proj['TOT'] = proj['X']+proj['Y']+proj['Z']
    proj['TURN'] = spdat['TURN'][:,0]
    proj['TIME'] = proj['TURN']/Wcyc # Wcyc -- the cyclotron frequency -- a constant defined above
    return proj

def process(case_name, mrkr='resflip'):
    dat, spdat, muarr = load(case_name, mrkr)
    step = int(dat[1,0]['TURN'])
    if True:      # svec + nbar plot
        fig1, ax1, fig2, ax2  = plot_spin_nbar(spdat, muarr)
        ax1[0].set_title(r'$\dot \psi = $ {} [kV/cm/{}-switch]'.format(case_name,step))
        fig1.savefig('../img/BYPASS/ADIABATICITY/'+case_name+'-SVEC+NBAR-plot.png', dpi=450, bbox_inches='tight', pad_inches=.1)
        ax2.set_title(r'$\dot \psi = $ {} [kV/cm/{}-switch]'.format(case_name,step))
        fig2.savefig('../img/BYPASS/ADIABATICITY/'+case_name+'-SVEC+NBAR-circ-plot.png', dpi=450, bbox_inches='tight', pad_inches=.1)
    if True:       # phase space plot
        fig3, ax3 = plot_PS(dat)
        ax3.set_title(r'$\dot \psi = $ {} [kV/cm/{}-switch]'.format(case_name,step))
        fig3.savefig('../img/BYPASS/ADIABATICITY/'+case_name+'-PHASESPACE-plot.png', dpi=450, bbox_inches='tight', pad_inches=.1)
    if True:     # polarization on nbar plot
        P = pol_on_nbar(spdat, muarr)
        fig4, ax4 = plt.subplots(1,1)
        ax4.plot(P['TIME'], P['TOT']); ax4.grid()
        ax4.set_title(r'$\dot \psi = $ {} [kV/cm/{}-switch]'.format(case_name,step))
        ax4.set_xlabel('time [sec]'); ax4.set_ylabel(r'$\sum\vec s\cdot \bar n^{CO}$')
        ax4.ticklabel_format(style='sci',scilimits=(0,0),useMathText=True)
        fig4.savefig('../img/BYPASS/ADIABATICITY/'+case_name+'-POLARIZATION-plot.png', dpi=450, bbox_inches='tight', pad_inches=.1)
    if True:      # polarization on ez plot
        P = pol_on_z(spdat)
        fig5, ax5 = plt.subplots(1,1)
        ax5.plot(P['TIME'], P['TOT']); ax5.grid()
        ax5.set_title(r'$\dot \psi = $ {} [kV/cm/{}-switch]'.format(case_name,step))
        ax5.set_xlabel('time [sec]'); ax5.set_ylabel(r'$\sum\vec s\cdot \hat e_z$')
        ax5.ticklabel_format(style='sci',scilimits=(0,0),useMathText=True)
        fig5.savefig('../img/BYPASS/ADIABATICITY/'+case_name+'-POLARIZATION_z-plot.png', dpi=450, bbox_inches='tight', pad_inches=.1)
    return P

def main(case_names, mrkr='resflip'):
    proj_dict = {}
    W_dict = {}
    for i, case in enumerate(case_names):
        proj = process(case, mrkr)
        #omega = spin_analysis(case, mrkr)
        proj_dict.update({case:proj})
        #W_dict.update({case:omega})
    return proj_dict, W_dict

def fig_four(spdat, muarr):
    P = pol_on_nbar(spdat, muarr)
    muarr0 = muarr[:,0] # muarr on the closed orbit
    fig, ax = plt.subplots(2,1, sharex=True)
    ax[1].set_xlabel('time [sec]')
    # svec
    ax[0].plot(spdat['TURN']/Wcyc, spdat['S_Z'], '-k')
    # nbar (on the CO)
    ax[0].plot(muarr0['TURN']/Wcyc, muarr0['NZ'], '--r', label=r'$\bar n^{CO}$')
    # metadata
    ax[0].set_ylabel(r'$\vec s_z$')
    ax[0].ticklabel_format(style='sci',axis='x', scilimits=(0,0),useMathText=True)
    ax[0].legend()
    ax[1].plot(P['TIME'], P['TOT']); ax[1].grid()
    ax[1].set_ylabel(r'$\sum\vec s\cdot \bar n^{CO}$')
    ax[1].set_yticks([-.5,0,.5,1])
    ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)
    fig.savefig('../img/WEPOPT006_f2.png', dpi=450, bbox_inches='tight', pad_inches=.1)
    return fig, ax

def fig_five(spdat1, spdat10, muarr1, muarr10):
    P = pol_on_nbar(spdat10, muarr10)
    muarr1 = muarr1[:,0] 
    muarr10 = muarr10[:,0]
    fig, ax = plt.subplots(3,1, sharex=True)
    ax[2].set_xlabel('time [sec]')
    ax[0].plot(spdat1['TURN']/Wcyc, spdat1['S_Z'], '-k')
    ax[0].plot(muarr1['TURN']/Wcyc, muarr1['NZ'], '--r', label=r'$\bar n^{CO}$')
    ax[0].set_ylabel(r'$\vec s_z$')
    ax[0].ticklabel_format(style='sci',axis='x', scilimits=(0,0),useMathText=True)
    ax[0].legend()
    ax[1].plot(spdat10['TURN']/Wcyc, spdat10['S_Z'], '-k')
    ax[1].plot(muarr10['TURN']/Wcyc, muarr10['NZ'], '--r', label=r'$\bar n^{CO}$')
    ax[1].set_ylabel(r'$\vec s_z$')
    ax[1].ticklabel_format(style='sci',axis='x', scilimits=(0,0),useMathText=True)
    #ax[1].legend()
    ax[2].plot(P['TIME'], P['TOT']); ax[2].grid()
    #ax[2].set_yticks([.925,.95,.975,1])
    ax[2].set_yticks([.93,.95,1])
    ax[2].set_ylabel(r'$\sum\vec s\cdot \bar n^{CO}$')
    ax[2].ticklabel_format(style='sci', scilimits=(0,0),useMathText=True)
    fig.savefig('../img/WEPOPT006_f3.png', dpi=450, bbox_inches='tight', pad_inches=.1)
    return fig, ax

if __name__ == '__main__':
    vals = ['5']
    pows = ['-4']
    case_names = np.array([[x+'E'+y for x in vals] for y in pows]).flatten()
    # case_names = ['30', '60', '90']
    proj_dict, W_dict = main(case_names, 'resflip')
    #rates = [deg_per_sec(float(e)) for e in case_names]
    #adiabaticity = st_dps/rates
