import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import HOMEDIR, DAVEC, load_data, Polarization, fit_line
from load_SpTrMAP import load_SpTrMAP, euler_angles
import matplotlib.cm as cmx
import matplotlib.colors as colors

CLIGHT = 3e8
PSVARS = list(zip(['X','A','Y','B','T','D'],[float]*6))

LATTICE = 'SECOND-ST'
LLENGTH = 503 # meters
ENERGY = '3059'
GAMMA = 1 + float(ENERGY)/938 # injecting PROTONS with rest mass 938 MeV
BETA = np.sqrt(1 - 1/GAMMA**2)
v = CLIGHT*BETA
TAU = LLENGTH/v # reference particle's time-of-flight
NTURN = '3000000'
DATDIR = '../data/'+LATTICE+'/'+ENERGY+'MeV/'+NTURN

SEQMAP = { #indexes of cornerstone elements (in COSY indexing, SEQFULL.fox file [i.e., no RF (which is at index 0 anyway)])
    'SPD1':21,  'ARC1s':43, 'ARC1f': 236,
    'MDP1':257, 'MPD2':293, # straight section
    'ARC2s':318, 'ARC2f':511, 'SPD2':530
    }


def load_tss(path=HOMEDIR+DATDIR+'MU.dat'):
    d_type = [('EL', int), ('PID', int)] + list(zip(['NU', 'NX','NY','NZ'], [float]*4))
    dat = np.loadtxt(path, dtype=d_type)
    nray = len(np.unique(dat['PID']))
    dat.shape = (-1, nray)
    return dat[:, 1:]

def plot(dat, spdat, rng = slice(0,-1,50), pid = [1,2,3], fmt='.-'):
    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(dat[rng,pid]['X']*1000, dat[rng,pid]['A']*1000, fmt)
    ax[0,0].set_xlabel('X [mm]'); ax[0,0].set_ylabel('A [mrad]')
    ax[0,1].plot(dat[rng,pid]['Y']*100, dat[rng,pid]['B']*1000, fmt)
    ax[0,1].set_xlabel('Y [mm]'); ax[0,1].set_ylabel('B [mrad]')
    ax[1,0].plot(dat[rng,pid]['T'], dat[rng,pid]['D'], fmt)
    ax[1,0].set_xlabel('T'); ax[1,0].set_ylabel('D');
    ax[1,0].ticklabel_format(style='sci', scilimits=(0,0), useMathText=True, axis='both')
    ax[1,1].plot(spdat[rng, pid]['iteration']/1000, spdat[rng,pid]['S_X'])
    ax[1,1].set_xlabel('turn [x1000]'); ax[1,1].set_ylabel('S_X')
    #ax[1,1].ticklabel_format(style='sci', scilimits=(0,0), useMathText=True, axis='x')
    return fig,ax

def plot_spin(spdat, rng=slice(0,-1,50),pid = [1,2,3], fmt='.-'):
    fig, ax = plt.subplots(3,1,sharex=True)
    ax[0].plot(spdat[rng, pid]['iteration']/1000, spdat[rng,pid]['S_X'])
    ax[0].set_ylabel('S_X')
    ax[1].plot(spdat[rng, pid]['iteration']/1000, spdat[rng,pid]['S_Y'])
    ax[1].set_ylabel('S_Y')
    ax[2].plot(spdat[rng, pid]['iteration']/1000, spdat[rng,pid]['S_Z'])
    ax[2].set_xlabel('turn [x1000]'); ax[2].set_ylabel('S_Z')
    return fig, ax

def plot_spin2(spdat, rng=slice(0,-1,50),pid = [1,2,3]):
    fig, ax = plt.subplots(2,1)
    SX, SY, SZ = (spdat[lbl][rng,pid] for lbl in ['S_X','S_Y','S_Z'])
    ax[0].plot(SX, SZ); ax[0].set_xlabel('S_X'); ax[0].set_ylabel('S_Z')
    ax[1].plot(SZ, SY); ax[1].set_xlabel('S_Z'); ax[1].set_ylabel('S_Y')
    return fig, ax

def plot_seq(dat, spdat, pid = [1,2,3], itn=(0,1), show_elems=[21, 43, 236, 257, 293, 318, 511, 530]):
    if type(itn)==int:
        ps1 = dat[dat[:,0]['iteration']<itn+1]
        sp1 = spdat[spdat[:,0]['iteration']<itn+1]
        eid = ps1['EID'][:, pid] if itn<2 else np.arange(ps1['EID'].max()*itn+1)
    else:
        itrow = dat[:,0]['iteration']
        ii = np.logical_and(itrow>itn[0], itrow<itn[1]+1)
        itrng = itn[1]-itn[0]
        ps1 = dat[ii]
        sp1 = spdat[ii]
        eid_max = ps1['EID'].max()
        eid = eid_max*itn[0] + np.arange(eid_max*itrng)
    fig, ax = plt.subplots(5,1,sharex=True)
    ax[0].plot(eid, ps1[:,pid]['X']*1000)
    ax[0].set_ylabel('X [mm]')
    ax[1].plot(eid, ps1[:,pid]['Y']*1000)
    ax[1].set_ylabel('Y[mm]')
    ax[2].plot(eid, sp1[:,pid]['S_X'])
    ax[2].set_ylabel('S_X')
    ax[3].plot(eid, sp1[:,pid]['S_Z'])
    ax[3].set_ylabel('S_Z')
    ax[4].plot(eid, sp1[:,pid]['S_Y'])
    ax[4].set_xlabel('EID'); ax[4].set_ylabel('S_Y')
    for i in range(5):
        ax[i].grid()
    if itn==1:
        fname = '../src/setups/'+LATTICE+'/FULL.npy'
        elnames = list(np.load(fname))
        elnames.insert(0, 'RF') # need this only if RF is inserted, which is most times but still -- not necessarily true
        elnames.insert(0, 'INJ') # **
        elnames=np.array(elnames)
        elnames = np.array([e+' ['+ str(i+1) + ']' for i,e in enumerate(elnames)]) # add the ordinal
        eid = eid[:,0] if eid.ndim>1 else eid
        eid_max = eid.max()
        show_elems=np.array(show_elems)+1 # +1 because of the added INJ **
                            # (the added RF is taken care of due to python indexing starting at 0 while cosy's at 1)
        plt.xticks(ticks=eid[show_elems], labels=elnames[show_elems], rotation=60)
    return fig, ax

def load_nbar(folder, spin_psi):
    nbar = {}
    for i, lbl in [(1,'X'),(2,'Y'),(3,'Z')]:
        nbar.update({lbl:DAVEC(folder+'NBAR({:d}):PSI0spin{}'.format(i, spin_psi))})
    return nbar

def main(navi_psi, spin_psi='-0'):
    folder  = DATDIR+'/NAVI-ON/NAVIPSI-{}/'.format(navi_psi)
    print(folder)
    dat = load_data(folder, 'TRPRAY:PSI0spin{}.dat'.format(spin_psi))
    spdat = load_data(folder, 'TRPSPI:PSI0spin{}.dat'.format(spin_psi))
    try:
        spintune = DAVEC(folder+'MU:PSI0spin{}'.format(spin_psi))
        nbar = load_nbar(folder, spin_psi)
    except:
        spintune = None; nbar = None # all components zero
        print("error in trying to load spin-tune/nbar")
    navi_psi_rad = np.deg2rad(180-navi_psi) # 180 b/c the navigators set psi in the SPD as 180 - psi
    print(navi_psi, navi_psi_rad)
    axis = [0, np.sin(navi_psi_rad), np.cos(navi_psi_rad)]
    print(axis)
    P = Polarization.on_axis(spdat, axis)
    P.plot(1)
    plt.savefig(folder+spin_psi+'-pol.png', bbox_inches='tight', pad_inches=.1)
    Px = Polarization.on_axis(spdat[1:-1:3], axis)
    Px.plot(1)
    plt.savefig(folder+spin_psi+'-pol-X-bunch.png', bbox_inches='tight', pad_inches=.1)
    fig, ax = plot(dat, spdat)
    plt.savefig(folder+spin_psi+'-plots.png', bbox_inches='tight', pad_inches=.1)
    fig2, ax2 = plot_spin(spdat)
    plt.savefig(folder+spin_psi+'-spin.png', bbox_inches='tight', pad_inches=.1)
    plt.close('all')
    return spintune, nbar

def polarization_analysis(navi_psi, spin_psi='-0'):
    folder  = DATDIR+'/NAVI-ON/NAVIPSI-{}/'.format(navi_psi)
    print(folder)
    spdat = load_data(folder, 'TRPSPI:PSI0spin{}.dat'.format(spin_psi))
    navi_psi_rad = np.deg2rad(180-navi_psi) # 180 b/c the navigators set psi in the SPD as 180 - psi
    axis = [0, np.sin(navi_psi_rad), np.cos(navi_psi_rad)]
    print(axis)
    P = Polarization.on_axis(spdat, axis)
    t = P['iteration']*TAU
    parr, errarr = fit_line(t, P['Value'])
    mean_P = P['Value'].mean()
    std_P = P['Value'].std()
    return parr, errarr, mean_P, std_P

def depolarization_analysis(psi_rng):
    n = len(psi_rng)
    d_type = [('xpct', float), ('serr', float)] # x-pectation and standard error
    mP = np.zeros(n, dtype=d_type)
    slp  = np.zeros(n, dtype=d_type)
    for i, psi in enumerate(psi_rng):
        parr, errarr, mean_P, std_P = polarization_analysis(psi,'=PSInavi')
        mP[i] = mean_P, std_P
        slp[i]  = parr[1], errarr[1]
    fig, ax = plt.subplots(2,1,sharex=True)
    ax[0].errorbar(psi_rng, mP['xpct']*100, yerr=mP['serr']*100)
    ax[1].errorbar(psi_rng, slp['xpct']*100, yerr=slp['serr']*100)
    ax[1].set_xlabel(r'$\psi_{navi}$ [deg]')
    ax[0].set_ylabel(r'$\langle P\rangle$ [%]')
    ax[1].set_ylabel(r'$\langle\beta\rangle$ [%/sec]')
    ax[0].set_title(r'Polarization data linear fit: $P\sim \alpha + \beta\cdot t$')
    ax[1].ticklabel_format(style='sci',scilimits=(0,0),useMathText=True,axis='y')
    for i in range(2):
        ax[i].grid()
    return mP, slp


def tss_analysis_mu(mu):
    def _tune_parabola(ax, i, j, var='X'): # used in tss_analysis()
        if var in ['X','Y']:
            ftr = 1e3
            xlbl = var + ' [mm]'
            rng = 3e-3
        else:
            ftr = 1
            xlbl = r'$\Delta K/K$'
            rng = 3e-4
            ## form phase space vectors
        npart = 11
        z = np.zeros(npart, dtype=PSVARS)
        z0 = np.zeros(1, dtype=PSVARS)
        z[var] = np.linspace(-rng, rng, npart)
        for lbl, itm in mu.items():
            vals = itm(z); vals = vals-itm(z0)
            ax[i,j].plot(z[var]*ftr, vals, label=lbl)
        ax[i,j].set_xlabel(xlbl)
        ax[i,j].set_ylabel(r'$\nu(\vec z; \psi) - \nu(\vec 0; \psi)$ for different $\psi$')
        ax[i,j].ticklabel_format(style='sci',scilimits=(0,0),useMathText=True)
        ax[i,j].grid()
    ## form mu0
    mu0 = np.zeros(len(mu), dtype=[('psi', float), ('const', float)])
    for i, (lbl, itm) in enumerate(mu.items()):
        try:
            mu0[i] = lbl, itm.const
        except:
            mu0[i] = lbl, 0
    ## plot spin tune
    figst, axst = plt.subplots(2,2)
    _tune_parabola(axst, 0, 0, 'X')
    _tune_parabola(axst, 1, 0, 'Y')
    _tune_parabola(axst, 1, 1, 'D')
    axst[1,1].legend(title=r'$\psi_{navi}$', bbox_to_anchor=(1,2))
    #
    axst[0,1].plot(mu0['psi'], mu0['const'], '.-')
    axst[0,1].set_xlabel(r'navigator-set $\psi$ [deg]')
    axst[0,1].set_ylabel(r'$\nu(\vec 0; \psi)$')
    axst[0,1].ticklabel_format(style='sci',scilimits=(0,0), axis='y', useMathText=True)
    axst[0,1].grid()
    return figst, axst

def tss_analysis_nbar(nbar):
    npart = 11; rng = 3e-3
    fig, ax = plt.subplots(3,3, sharex='col')
    ftr = 1e3
    for j, var in enumerate(['X','Y','D']):
        if var=='D':
            ftr = 1; rng = 3e-4;
        z = np.zeros(npart, dtype=PSVARS)
        z[var] = np.linspace(-rng, rng, npart)
        for i, coo in enumerate(['X','Y','Z']):
            ax[i,0].set_ylabel(r'$\bar n_{} - \bar n_{}^0$'.format(coo, coo))
            ax[i,j].ticklabel_format(style='sci', scilimits=(0,0), useMathText=True)
            for lbl, itm in nbar.items():
                ax[i,j].plot(z[var]*ftr, itm[coo](z)-itm[coo].const, label=lbl)
            ax[i,j].grid()
        var = r'$\Delta K/K$' if var=='D' else var + ' [mm]'
        ax[2,j].set_xlabel(var)
    ax[2,2].legend(title=r'$\psi_{navi}$', bbox_to_anchor=(1,2))
    return fig, ax
        
def spin_matrix_analysis(navi_psi, spin_psi='-0'):
    folder  = DATDIR+'/NAVI-ON/NAVIPSI-{:d}/'.format(navi_psi)
    stm = load_SpTrMAP(folder+'SpTrMAP:PSI0spin'+spin_psi)
    angles = euler_angles(stm)
    return stm, angles

def polar_nbar(nbar0):
    fig, ax = plt.subplots()
    psi_arr = nbar0['psi']
    n0z = nbar0['Z']
    n0y = nbar0['Y']
    cmap = plt.cm.jet
    cNorm  = colors.Normalize(vmin=np.min(n0z), vmax=np.max(n0z))
    scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=cmap)
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    for i, psi in enumerate(psi_arr):
        colorVal = scalarMap.to_rgba(n0z[i])
        plt.arrow(0,0,n0z[i],n0y[i], width=.005, color=colorVal)
        plt.text(n0z[i], n0y[i], r'$\psi={}^\circ$'.format(psi))
    plt.grid()

if __name__ == '__main__':
    mu = {}
    nbar = {}
    psi_rng = [180, 135, 90]#range(0,380,20)
    ndir = len(psi_rng)
    STM = np.zeros(ndir, dtype=[('psi', float),('stm', object)])
    euang = np.zeros(ndir, dtype=list(zip(['psi','X','Y','Z'],[float]*4)))
    nbar0 = np.zeros(ndir, dtype=list(zip(['psi', 'X','Y','Z'], [float]*4)))
    for i, psi in enumerate(psi_rng):
        mu_, nbar_ = main(psi,'-'+str(psi))
        mu.update({psi:mu_})
        nbar.update({psi:nbar_})
        stm, ang = spin_matrix_analysis(psi,'-'+str(psi))
        STM[i] = psi, stm
        euang[i] = psi, *ang
        try:
            nbar0[i] = psi, *(nbar_[coo].const for coo in ['X','Y','Z'])
        except:
            nbar0[i] = psi, 0, 0, 0
