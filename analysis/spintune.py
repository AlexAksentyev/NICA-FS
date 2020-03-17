import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import HOMEDIR, DAVEC, load_ps, load_sp

DIR  = 'data/SPINTUNE/'

ELNAMES = np.load('nica_element_names.npy')
ELNAMES = np.insert(ELNAMES, 0,'RF')

def pick_elems(name, dat):
    lbls = tick_labels(dat)
    sub_ii = [i for i, e in enumerate(lbls) if name in e]
    return sub_ii, np.array(lbls)[sub_ii]

def tick_labels(dat):
    eid = dat['EID'][:,0]
    name = ELNAMES[eid-1]
    return ['{} [{}]'.format(*e) for e in list(zip(name, eid))]

def plot(dat, fun=lambda x: x, elem='all'):
    if elem=='all':
        jj = np.arange(dat.shape[0])
        lbls = tick_labels(dat)
    else:
        jj, lbls = pick_elems(elem, dat)
    fig, ax = plt.subplots(2,1, sharex=True)
    ax[0].plot(fun(dat[jj,:]['NU']))
    ax[0].set_ylabel(r'$f(\nu_s)$')
    for v in ['NX','NY','NZ']:
        ax[1].plot(fun(dat[jj,:][v]), label=v)
    ax[1].legend()
    ax[1].set_ylabel(r'$f(\bar n_{\alpha})$')
    for i in range(2):
        ax[i].ticklabel_format(axis='both', style='sci', scilimits=(0,0), useMathText=True)
        ax[i].grid(axis='x')
    plt.xticks(ticks=np.arange(len(jj)), labels=lbls, rotation=90)


def load_tss(path=HOMEDIR+DIR+'MU.dat'):
    d_type = [('EID', int), ('PID', int)] + list(zip(['NU', 'NX','NY','NZ'], [float]*4))
    dat = np.loadtxt(path, dtype=d_type)
    nray = len(np.unique(dat['PID']))
    dat.shape = (-1, nray)
    return dat[:, 1:]

if __name__ == '__main__':
    dat = load_tss()
    plot(dat[:, 2::3])
