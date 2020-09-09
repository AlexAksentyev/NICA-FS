import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import HOMEDIR, Data, TAU, Polarization, fit_line, TSS
import os
from glob import glob

def plot_TD(case, pid=1, num_seg=5):
    fig, ax = plt.subplots(1,1)
    N = int(case['iteration'].shape[0]/num_seg)
    for i in range(num_seg):
        rng = slice(i*N, (i+1)*N)
        sec0, sec1 = (case['iteration'][j*N,0]*TAU for j in [i,i+1])
        ax.plot(case['T'][rng,pid]*1e3, case['D'][rng,pid]*1e2, '.',
                    label='{:3.1f}--{:3.1f} sec'.format(sec0, sec1)
                    )
    ax.set_xlabel(r'$\ell$ [mm]')
    ax.set_ylabel(r'$\delta$ [%]')
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0), useMathText=True)
    ax.legend()
    return fig, ax

def main1():
    path = lambda case: HOMEDIR + 'data/REPORT/DEUTERON/BENDS24/24MTURN/Y-bunch/SPD-{}/'.format(case)
    CASES = ['0+', '90+', '90-']
    
    sp = {case: Data(path(case), 'TRPSPI:MAIN.dat') for case in CASES}
    ps = {case: Data(path(case), 'TRPRAY:MAIN.dat') for case in CASES}
    pol = {case: Polarization.on_axis(sp[case], [0,1,0]) for case in CASES}

    for case in CASES:
        print(case)
        d = path(case)
        if not os.path.exists(d+'img/'):
            os.makedirs(d+'img/')
        fig, ax = plot_TD(ps[case][3000:],num_seg=4)
        plt.savefig(d+'img/PS_TD_{}.png'.format(case), dpi=450, bbox_inches='tight', pad_inches=.1)
        plt.close()

def analysis(path):
    def plot_pol():
        fpol, axpol = pol.plot(1, 'sec')
        axpol.grid(axis='y')
        plt.savefig(path+'polarization.png', dpi=450, bbox_inches='tight', pad_inches=.1)
        plt.close()
    def plot_td():
        fig, ax = plot_TD(ps[3000:],num_seg=4)
        plt.savefig(path+'PS_TD.png', dpi=450, bbox_inches='tight', pad_inches=.1)
        plt.close()
    def plot_tss():
        print("plotting TSS")
        ftss, axtss = tss.plot()
        # axtss[0].set_title(name.split(":")[1].strip())
        plt.savefig(path+'tss.png', dpi=450, bbox_inches='tight', pad_inches=.1)
        plt.close()
    tss = TSS(path, 'MU.dat')
    sp = Data(path, 'TRPSPI:MAIN.dat')
    axis = sp[0,0][['S_X','S_Y','S_Z']]
    ps = Data(path, 'TRPRAY:MAIN.dat')
    pol = Polarization.on_axis(sp, axis)
    plot_pol()
    plot_td()
    plot_tss()
    t = pol['iteration']*TAU
    par, err = fit_line(t, pol['Value'])
    return par[1], err[1]

def meta_analysis(root):
    dir_list = glob(root+'*/')
    ncase = len(dir_list)
    result = np.zeros(ncase, dtype = list(zip(['psi','Xslp','SEslp'],[float]*3)))
    for i, d in enumerate(dir_list):
        bunch, psi = d.split('/')[-3:-1]
        print(psi)
        psi = np.deg2rad(float(psi[-1]+psi[4:-1]))
        result[i] = psi, *analysis(d)
    return result

def plot_meta(res):
    angle = abs(np.rad2deg(res['psi']))
    Xslp = abs(res['Xslp'])*100
    SEslp = res['SEslp']*100
    fig, ax = plt.subplots(1,1)
    ax.errorbar(angle, Xslp, yerr=SEslp, fmt='.')
    ax.set_xlabel('nbar tilt angle [deg]')
    ax.set_ylabel('polarization line fit slope [%/sec]')
    #plt.yscale('log')
    return fig, ax

if __name__ == '__main__':
    path = HOMEDIR+'data/REPORT/DEUTERON/BENDS24/18MTURN/X-bunch/'
    mres = meta_analysis(path)
    fig, ax = plot_meta(mres)
    plt.savefig(path+'meta_analysis.png', dpi=450, bbox_inches='tight', pad_inches=.1)
