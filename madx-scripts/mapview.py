import numpy as np

### maps to compare
mad_name = 'BH'
cosy_name = '50'


HOME = '/Users/alexaksentyev/REPOS/NICA-FS/'
DIR  = 'data/ELEMENT_MAPS/'
VARS  = ['X','A','Y','B','T','D']
NVARS = len(VARS)


DTYPE = [('name', object), ('coef', float)]

def parse_names(names):
    conv = lambda x: [int(e) for e in list(x)]
    names = np.array([e.strip('",C').split("_") for e in names])
    fin = np.array([int(e) for e in names[:,0]])-1
    ini = np.array([conv(e) for e in names[:,1]])
    return np.vstack((fin, ini.T)).T


## getting the MAD-X transfer map definition
maptable = np.loadtxt("elmap/map_table_"+mad_name, skiprows=8, usecols=(0,1), dtype=DTYPE)

transfer_MADX = np.zeros((6,6))
pmat = parse_names(maptable['name'])
carr = maptable['coef']

for k, c  in enumerate(carr):
    i = pmat[k][0]
    try:
        j = list(pmat[k][1:]).index(1)
    except:
        print(k, i, j, c, '***** constant coefficient')
    print(k, i, j, c)
    transfer_MADX[i,j] = c
    

## same in COSY Infinity
tmp = np.genfromtxt(HOME+DIR+'ELEM_'+cosy_name, skip_footer = 1,
                        #dtype=DTYPE,
                        delimiter=(1, 14, 14, 14, 14, 14, 7),
                        usecols = range(1,NVARS))
transfer_COSY = np.zeros((NVARS, NVARS)) 
transfer_COSY[:5,:] = tmp.T; transfer_COSY[5,-1] = 1


## printing
print('********************')
print('COSY[:2, :2]')
print(transfer_COSY[:2, :2])
print('MADX[:2, :2]')
print(transfer_MADX[:2, :2])
print('COSY[2:4, 2:4]')
print(transfer_COSY[2:4, 2:4])
print('MADX[2:4, 2:4]')
print(transfer_MADX[2:4, 2:4])
