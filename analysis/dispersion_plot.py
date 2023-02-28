import numpy as np
import matplotlib.pyplot as plt; plt.ion()

def prep_elnames(raw=False):
    elnames =  np.load('../src/setups/BYPASS/FULL_SEX_CLEAR-element_names.npy', allow_pickle=True)
    elnames = np.array([e.strip('\n').strip('\{').strip('\}') for e in elnames])
    if not raw:
        elnames = np.array([e +' ['+ str(i+1) + ']' for i,e in enumerate(elnames)])
    return elnames


DIR = '../data/BYPASS/FULL_SEX_CLEAR/'
ELEMENTS = prep_elnames()
ELEMENTS_RAW = prep_elnames(True)
iSF = ELEMENTS_RAW == 'SF'
iSD = ELEMENTS_RAW == 'SD'
iSEX = iSF + iSD
TICK_STP=20

disp_x = np.loadtxt(DIR+'DISPX', dtype=[('DISP', float), ('len', float)], skiprows=1)
disp_y = np.loadtxt(DIR+'DISPY', dtype=[('DISP', float), ('len', float)], skiprows=1)
beta_x = np.loadtxt(DIR+'BETAX', dtype=[('BETA', float), ('len', float)], skiprows=1)
beta_y = np.loadtxt(DIR+'BETAY', dtype=[('BETA', float), ('len', float)], skiprows=1)
S = np.cumsum(disp_x['len']) # longitudinal coordinate

fig, ax = plt.subplots(3,1,sharex=True)
ax[0].plot(S, beta_x['BETA'], label='beta_x')
ax[0].plot(S, beta_y['BETA'], label='beta_y')
ax[1].plot(S, disp_x['DISP'], label='disp_x')
ax[1].plot(S, disp_y['DISP'], label='disp_y')
ax[2].plot(S, disp_x['DISP'], label='disp_x')
ax[2].plot(S, beta_x['BETA'], label='beta_x')
ax[2].plot(S, beta_y['BETA'], label='beta_y')
for i in range(3):
    ax[i].legend()
    ax[i].grid()
# plt.xticks(ticks=range(len(ELEMENTS))[::TICK_STP], labels=ELEMENTS[::TICK_STP], rotation=90)
plt.xticks(ticks=np.arange(len(ELEMENTS))[iSEX], labels=ELEMENTS[iSEX], rotation=90)
