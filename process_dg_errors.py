#!/usr/bin/env python
import pylab
import numpy as np
import matplotlib as mpl
import os
import errno
import pickle
from itertools import cycle 
styles = cycle(['b-', 'r-', 'b-.', 'r-.']) 
mpl.rc_file(r'./mpl.rc')

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return

# load errors from already-run simulations
dg_errors = pickle.load(open('dg_error_pickle.p', 'rb'))

# reconstruct keys
pdescs = dg_errors.keys()
ETdescs = map(lambda x: x[1], sorted(map(lambda x: (-len(x.split(' ')), x), dg_errors[pdescs[0]].keys()), reverse=True))
NumElements = sorted(dg_errors[pdescs[0]][ETdescs[0]].keys())

# output errors in a latex-friendly table using siunitx's '\num'
for pd in pdescs:
    print r'\begin{table}[!h]'
    print r'\centering'
    print r'\caption{'+pd+r': $L_2$ Error (DG method)}'
    print r'\begin{tabular}{r | ' + ' '.join(['r']*len(ETdescs)) + r'}'
    print r'Number of Elements & ' + ' & '.join(ETdescs) + r' \\'
    print r'\midrule'
    for Ne in NumElements:
        print r'\num{{{:d}}} &'.format(Ne),
        print ' & '.join(map(lambda ET: r'\num{{{:.4e}}}'.format(dg_errors[pd][ET][Ne][1]), ETdescs)) + r' \\'
    print r'\end{tabular}'
    print r'\end{table}'

# same but for H1 error
for pd in pdescs:
    print r'\begin{table}[!h]'
    print r'\centering'
    print r'\caption{'+pd+r': $H_1$ Error (DG method)}'
    print r'\begin{tabular}{r | ' + ' '.join(['r']*len(ETdescs)) + r'}'
    print r'Number of Elements & ' + ' & '.join(ETdescs) + r' \\'
    print r'\midrule'
    for Ne in NumElements:
        print r'\num{{{:d}}} &'.format(Ne),
        print ' & '.join(map(lambda ET: r'\num{{{:.4e}}}'.format(dg_errors[pd][ET][Ne][1]), ETdescs)) + r' \\'
    print r'\end{tabular}'
    print r'\end{table}'

# L2 and H1 convergence rates
for pd in pdescs:
    print r'\begin{table}[!h]'
    print r'\centering'
    print r'\caption{'+pd+r': Convergence Rates (DG method)}'
    print r'\begin{tabular}{r | r r}'
    print r'Element & $L_2$ & $H_1$ \\'
    print r'\midrule'
    for ET in ETdescs:
        upper = dg_errors[pd][ET][80][0]
        lower = dg_errors[pd][ET][100][0]
        l2_rate = np.log10(upper/lower)/np.log10(100.0/80.0)
        upper = dg_errors[pd][ET][80][1]
        lower = dg_errors[pd][ET][100][1]
        h1_rate = np.log10(upper/lower)/np.log10(100.0/80.0)
        print ET+r' & ',
        print r'\num{{{:.3f}}} & '.format(l2_rate),
        print r'\num{{{:.3f}}} \\'.format(h1_rate)
    print r'\end{tabular}'
    print r'\end{table}'

# generate plots for L2 error
for pd in pdescs:
    f = pylab.figure(figsize=(3,3))
    for et in ETdescs:
        h = map(lambda x: 1.0/x, NumElements)
        L2_errors = map(lambda x: dg_errors[pd][et][x][0], NumElements)
        pylab.loglog(h, L2_errors, styles.next())
    pylab.legend(ETdescs, loc='best', prop={'size': 8})
    pylab.title('$L_2$ error for ' + pd)
    pylab.ylabel('$L_2$ error')
    pylab.xlabel('Element size $h$ [1/$N_e$]')
    pylab.tight_layout(pad=0.2)
    make_sure_path_exists('report/figs')
    pylab.savefig('report/figs/dg_l2_error_'+pd.split(' ')[0]+'.png')
    pylab.close(f)
        
# generate plots for L2 error
for pd in pdescs:
    f = pylab.figure(figsize=(3,3))
    for et in ETdescs:
        h = map(lambda x: 1.0/x, NumElements)
        H1_errors = map(lambda x: dg_errors[pd][et][x][1], NumElements)
        pylab.loglog(h, H1_errors, styles.next())
    pylab.legend(ETdescs, loc='best', prop={'size': 8})
    pylab.title('$H_1$ error for ' + pd)
    pylab.ylabel('$H_1$ error')
    pylab.xlabel('Element size $h$ [1/$N_e$]')
    pylab.tight_layout(pad=0.2)
    make_sure_path_exists('report/figs')
    pylab.savefig('report/figs/dg_h1_error_'+pd.split(' ')[0]+'.png')
    pylab.close(f)

