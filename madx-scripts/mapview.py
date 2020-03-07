import numpy as np

### maps to compare
mad_name = 'BV2'
cosy_name = '12'


HOME = '/Users/alexaksentyev/REPOS/NICA-FS/'
DIR  = 'data/ELEMENT_MAPS/AUTO/'

O = np.zeros((3,3))
I = np.eye(3)
S = np.block([[O, I], [-I, O]])

def symplecticity_test(mat):
    return abs(S - mat.T@(S@mat))<1e-12

def pm(map_):
    conv = lambda x: 0 if abs(x)<1e-12 else x
    for i in range(6):
        row = list(map(conv, map_[i,:]))
        print('{: 4.6e} {: 4.6e} {: 4.6e} {: 4.6e} {: 4.6e} {: 4.6e}'.format(*row))

def compare(m1, m2):
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

def parse_names(names):
    conv = lambda x: [int(e) for e in list(x)]
    names = np.array([e.strip('",C').split("_") for e in names])
    fin = np.array([int(e) for e in names[:,0]])-1
    ini = np.array([conv(e) for e in names[:,1]])
    return np.vstack((fin, ini.T)).T

def load_madx(name):
    maptable = np.loadtxt("elmap/map_table_"+name, skiprows=8, usecols=(0,1), dtype=[('name', object), ('coef', float)])

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

def load_cosy(name, dir_=DIR):
    tmp = np.genfromtxt(HOME+dir_+'ELEM_'+name, skip_footer = 1,
                        delimiter=(1, 14, 14, 14, 14, 14, 7),
                        usecols = range(1,6))
    trans_map = np.zeros((6, 6)) 
    trans_map[:5,:] = tmp.T; trans_map[5,-1] = 1
    return trans_map

def solmap(L, KS):
    c = np.cos(KS*L)
    s = np.sin(KS*L)
    M = np.eye(4)
    M[0,0] = c*c
    M[0,1] = s*c/KS/2
    M[0,2] = s*c
    M[0,3] = s*s/KS/s
    M[1,0] = -KS/2*c
    M[1,1] = c*c
    M[1,2] = -KS/2*s*s
    M[1,3] = s*c
    M[2,0] = -s*c
    M[2,1] = -s*s/KS/2
    M[2,2] = c*c
    M[2,3] = s*c/KS/2
    M[3,0] = KS/2*s*s
    M[3,1] = -s*c
    M[3,2] = -KS/2*s*c
    M[3,3] = c*c
    return M
if __name__ == '__main__':
    
## getting the MAD-X transfer map definition
    transfer_MADX = load_madx(mad_name)

## same in COSY Infinity
    transfer_COSY = load_cosy(cosy_name)


## printing
    print('********************')
    print('NAME:', cosy_name, '(COSY)/', mad_name, '(MADX)')
    print('********************')
    print('COSY[:2, :2]')
    print(transfer_COSY[:2, :2])
    print('MADX[:2, :2]')
    print(transfer_MADX[:2, :2])
    print('++ Difference')
    print(transfer_COSY[:2,:2]-transfer_MADX[:2,:2])
    print('COSY[2:4, 2:4]')
    print(transfer_COSY[2:4, 2:4])
    print('MADX[2:4, 2:4]')
    print(transfer_MADX[2:4, 2:4])
    print('++ Difference')
    print(transfer_COSY[2:4,2:4]-transfer_MADX[2:4,2:4])

    eldict = {'BH':'50','BV1':'10','BV2':'12','MQD1':'22','MQF1':'20','QD1':'2','QD2':'6','QF2':'4','SNK1':'26'}
    maparr = np.empty(len(eldict), dtype=object)
    for i, names in enumerate(eldict.items()):
        maparr[i] = (load_madx(names[0]), load_cosy(names[1]))

    for i, names in enumerate(eldict.items()):
        print('')
        print('{} (MADX) / {} (COSY)'.format(*names))
        compare(maparr[i][0], maparr[i][1])

    BH = load_madx('BH'); E50 = load_cosy('50')
    BV1 = load_madx('BV1'); E10 = load_cosy('10')
    BV2 = load_madx('BV2'); E12 = load_cosy('12')
    MQD1 = load_madx('MQD1'); E22 = load_cosy('22')
    MQF1 = load_madx('MQF1'); E20 = load_cosy('20')
    QD1 = load_madx('QD1'); E2 = load_cosy('2')
    QD1 = load_madx('QD2'); E6 = load_cosy('6')
    QF2 = load_madx('QF2'); E4 = load_cosy('4')
    SNK1 = load_madx('SNK1'); E26 = load_cosy('26')

    hz1 = 0.3739991

