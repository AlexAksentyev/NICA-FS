import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import load_data

DATADIR = '../data/BYPASS_SEX_CLEAR/EL-BY-EL-TRACKING/'

ps = load_data(DATADIR, 'TRPRAY:SEQ.dat')
sp = load_data(DATADIR, 'TRPSPI:SEQ.dat')

def prep_elnames(raw=False):
    elnames =  list(np.load('../src/setups/BYPASS/FULL_SEX_CLEAR-element_names.npy', allow_pickle=True))
    elnames.insert(0, 'RF')
    elnames.insert(0, 'inj')
    elnames = np.array(elnames)
    elnames = np.array([e.strip('\n').strip('\{').strip('\}') for e in elnames])
    if not raw:
        elnames = np.array([e +' ['+ str(i+1) + ']' for i,e in enumerate(elnames)])
    return elnames


ELNAMES = prep_elnames()
#ELNAMES1 = np.array([re.sub('ra','EB', e) for e in ELNAMES])
ELNAMES_RAW=prep_elnames(True) # raw names
iEB = ELNAMES_RAW == 'ra'
iBH = ELNAMES_RAW == 'BH'
show_elems = iEB + iBH

def plot_seq(dat, spdat, pid = [1,2,3], itn=(0,1), show_elems=None):
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
        elnames=prep_elnames() # element names
        eid = eid[:,0] if eid.ndim>1 else eid
        eid_max = eid.max()
        if not np.any(show_elems)==None:
            plt.xticks(ticks=np.arange(len(ELNAMES))[show_elems], labels=ELNAMES[show_elems], rotation=90)
    return fig, ax



if __name__ == '__main__':
    plot_seq(ps,sp, itn=1, show_elems=show_elems)
