import numpy as np
import matplotlib.pyplot as plt; plt.ion()

def prep_elnames():
    elnames =  np.load('../src/setups/BYPASS/FULL_SEX_CLEAR-element_names.npy', allow_pickle=True)
    elnames = np.array([e.strip('\n').strip('\{').strip('\}')+' ['+ str(i+1) + ']' for i,e in enumerate(elnames)])
    return elnames


DIR = '../data/BYPASS/FULL_SEX_CLEAR/'
ELEMENTS = prep_elnames()
TICK_STP=20

disp_x = np.loadtxt(DIR+'DISPX', dtype=[('DISP', float), ('len', float)], skiprows=1)
disp_y = np.loadtxt(DIR+'DISPY', dtype=[('DISP', float), ('len', float)], skiprows=1)
S = np.cumsum(disp_x['len']) # longitudinal coordinate

fig, ax = plt.subplots(1,1)
ax.plot(S, disp_x['DISP'], label='disp_x')
ax.plot(S, disp_y['DISP'], label='disp_y')
ax.legend()
ax.grid()
plt.xticks(ticks=range(len(ELEMENTS))[::TICK_STP], labels=ELEMENTS[::TICK_STP], rotation=90)
