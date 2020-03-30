import numpy as np


## functions loading transfer maps produced by
## MAD-X PTC module
def load_madx(address):
    def parse_names(names):
        conv = lambda x: [int(e) for e in list(x)]
        names = np.array([e.strip('",C').split("_") for e in names])
        fin = np.array([int(e) for e in names[:,0]])-1
        ini = np.array([conv(e) for e in names[:,1]])
        return np.vstack((fin, ini.T)).T
    maptable = np.loadtxt(address, skiprows=8, usecols=(0,1), dtype=[('name', object), ('coef', float)])

    trans_map = np.zeros((6,6))
    pmat = parse_names(maptable['name'])
    carr = maptable['coef']

    for k, c  in enumerate(carr):
        i = pmat[k][0]
        try:
            j = list(pmat[k][1:]).index(1)
            trans_map[i,j] = c
        except:
            print(k, i, c, '***** constant coefficient')
    return trans_map

## COSY PM procedure
def load_cosy(address):
    tmp = np.genfromtxt(address, skip_footer = 1,
                        delimiter=(1, 14, 14, 14, 14, 14, 7),
                        usecols = range(1,6))
    trans_map = np.zeros((6, 6)) 
    trans_map[:5,:] = tmp.T; trans_map[5,-1] = 1
    return trans_map

def pm(map_): ## outputs maps loaded by the above two functions into console in a human-readable form
    conv = lambda x: 0 if abs(x)<1e-12 else x
    for i in range(6):
        row = list(map(conv, map_[i,:]))
        print('{: 4.6e} {: 4.6e} {: 4.6e} {: 4.6e} {: 4.6e} {: 4.6e}'.format(*row))

def compare(m1, m2): # compares maps loaded by load_cosy and load_madx
    print('x-a')
    print('{: 4.6e}|{: 4.6e}    {: 4.6e}|{: 4.6e}'.format(m1[0,0], m1[0,1], m1[1,0], m1[1,1]))
    print('{: 4.6e}|{: 4.6e}    {: 4.6e}|{: 4.6e}'.format(m2[0,0], m2[0,1], m2[1,0], m2[1,1]))
    print('{: 4.6e}|{: 4.6e}    {: 4.6e}|{: 4.6e}'.format(
        m2[0,0]-m1[0,0], m2[0,1]-m1[0,1],
        m2[1,0]-m1[1,0], m2[1,1]-m1[1,1])
              )
    print('=============')
    print('y-b')
    print('{: 4.6e}|{: 4.6e}    {: 4.6e}|{: 4.6e}'.format(m1[2,2], m1[2,3], m1[3,2], m1[3,3]))
    print('{: 4.6e}|{: 4.6e}    {: 4.6e}|{: 4.6e}'.format(m2[2,2], m2[2,3], m2[3,2], m2[3,3]))
    print('{: 4.6e}|{: 4.6e}    {: 4.6e}|{: 4.6e}'.format(
        m2[2,2]-m1[2,2], m2[2,3]-m1[2,3],
        m2[3,2]-m1[3,2], m2[3,3]-m1[3,3]))
    print('=============')
    print('t-d')
    print('{: 4.6e}|{: 4.6e}    {: 4.6e}|{: 4.6e}'.format(m1[4,4], m1[4,5], m1[5,4], m1[5,5]))
    print('{: 4.6e}|{: 4.6e}    {: 4.6e}|{: 4.6e}'.format(m2[4,4], m2[4,5], m2[5,4], m2[5,5]))
    print('{: 4.6e}|{: 4.6e}    {: 4.6e}|{: 4.6e}'.format(
        m2[4,4]-m1[4,4], m2[4,5]-m1[4,5],
        m2[5,4]-m1[5,4], m2[5,5]-m1[5,4]))
    print('=============')
