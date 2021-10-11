import numpy as np
from os import listdir

HOME = '/Users/alexaksentyev/REPOS/NICA-FS/'
DIR  = 'data/ELEMENT_MAPS/SECOND-ST/'
VARS  = ['X','A','Y','B','T','D']
NVARS = len(VARS)
VIN = ['X','A','Y','B','T']
DTYPE = [('dummy', object)] + list(zip(VIN, [float]*5)) + [('EXP', int)]


map_files = listdir(HOME+DIR)
map_files.sort(key = lambda x: int(x[5:]))

nfile = len(map_files)

elmap = np.empty(nfile, dtype=object)

for i, map_file in enumerate(map_files):
    tmp = np.genfromtxt(HOME+DIR+map_file, skip_footer = 1,
                        #dtype=DTYPE,
                        delimiter=(1, 14, 14, 14, 14, 14, 7),
                        usecols = range(1,NVARS))
    try:
        elmap[i] = np.zeros((NVARS, NVARS)) #np.zeros(NVARS, dtype=list(zip(VARS, [float]*NVARS)))
        #elmap[i][VIN] = tmp[VIN]
        #elmap[i]['D'][-1] = 1
        elmap[i][:5,:] = tmp.T
        elmap[i][5,-1] = 1
    except:
        print(map_file)
