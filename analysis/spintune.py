import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import HOMEDIR, DAVEC, load_ps, load_sp

ELNAMES = np.load('nica_element_names.npy')
ELNAMES = np.insert(ELNAMES, 0,'RF')

DIR = 'data/SPINTUNE/'

def pick_elems(name, dat):
    lbls = tick_labels(dat)
    sub_ii = np.array([i for i, e in enumerate(lbls) if name in e])
    return sub_ii, np.array(lbls)[sub_ii]

def tick_labels(dat):
    eid = dat['EID'][:,0]
    name = ELNAMES[eid-1]
    return ['{} [{}]'.format(*e) for e in list(zip(name, eid))]

def plot(dat, fun=lambda x: x.mean(1), elem='all'):
    dc = dat.copy()
    if elem=='all':
        jj = np.arange(dc.shape[0])
        lbls = tick_labels(dc)
        ylab_pref = ''
    else:
        jj, lbls = pick_elems(elem, dc)
        ylab_pref = r'$\Delta$'
        dc1 = np.append(dc, dc[:2], axis=0)[:-1]
        for name in dc.dtype.names[2:]:
            dc[name] = np.diff(dc1[name], axis=0)
        del dc1
    fig, ax = plt.subplots(2,1, sharex=True)
    ax[0].plot(fun(dc[jj,:]['NU']))
    ax[0].set_ylabel(ylab_pref + r'$f(\nu_s)$')
    ax[1].set_ylabel(ylab_pref + r'$f(\bar n_{\alpha})$')
    for v in ['NX','NY','NZ']:
        ax[1].plot(fun(dc[jj,:][v]), label=v)
    ax[1].legend()
    for i in range(2):
        ax[i].ticklabel_format(axis='both', style='sci', scilimits=(0,0), useMathText=True)
        ax[i].grid(axis='x')
    plt.xticks(ticks=np.arange(len(jj)), labels=lbls, rotation=90)
    return fig, ax


def load_tss(path=HOMEDIR+DIR+'MU.dat'):
    case = path.split('/')[-2]
    d_type = [('EID', int), ('PID', int)] + list(zip(['NU', 'NX','NY','NZ'], [float]*4))
    dat = np.loadtxt(path, dtype=d_type)
    nray = len(np.unique(dat['PID']))
    dat.shape = (-1, nray)
    return dat[:, 1:], case

if __name__ == '__main__':
    bunch_num = 2 # 0 (X), 1 (Y), 2 (D)
    dat1, case1 = load_tss(HOMEDIR+'data/SPINTUNE/NO_NAVIG/'+'MU.dat')
    dat2, case2 = load_tss(HOMEDIR+'data/SPINTUNE/NAVIG-DECI-SAME/'+'MU.dat')
    dat3, case3 = load_tss(HOMEDIR+'data/SPINTUNE/NAVIG-DECI-OPPOSITE/'+'MU.dat')
    pcls = [0,1,2,3]
    fun = lambda x: x[:,0]
    fig1, ax1 = plot(dat1[:, pcls], fun=fun)
    ax1[0].set_title(case1)
    fig2, ax2 = plot(dat2[:, pcls], fun=fun)
    ax2[0].set_title(case2)
    fig3, ax3 = plot(dat3[:, pcls], fun=fun)
    ax3[0].set_title(case3)
