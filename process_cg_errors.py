#!/usr/bin/env python
import pylab
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
cg_errors = pickle.load(open('cg_error_pickle.p', 'rb'))

# reconstruct keys
pdescs = cg_errors.keys()
ETdescs = map(lambda x: x[1], sorted(map(lambda x: (-len(x.split(' ')), x), cg_errors[pdescs[0]].keys()), reverse=True))
NumElements = sorted(cg_errors[pdescs[0]][ETdescs[0]].keys())

# output errors in a latex-friendly table using siunitx's '\num'
for pd in pdescs:
    print r'\begin{table}[!h]'
    print r'\centering'
    print r'\caption{'+pd+r': $L_2$ Error (CG methods)}'
    print r'\begin{tabular}{r | ' + ' '.join(['r']*len(ETdescs)) + r'}'
    print r'Number of Elements & ' + ' & '.join(ETdescs) + r' \\'
    print r'\midrule'
    for Ne in NumElements:
        print r'\num{{{:d}}} &'.format(Ne),
        print ' & '.join(map(lambda ET: r'\num{{{:.4e}}}'.format(cg_errors[pd][ET][Ne][1]), ETdescs)) + r' \\'
    print r'\end{tabular}'
    print r'\end{table}'

# same but for H1 error
for pd in pdescs:
    print r'\begin{table}[!h]'
    print r'\centering'
    print r'\caption{'+pd+r': $H_1$ Error (CG methods)}'
    print r'\begin{tabular}{r | ' + ' '.join(['r']*len(ETdescs)) + r'}'
    print r'Number of Elements & ' + ' & '.join(ETdescs) + r' \\'
    print r'\midrule'
    for Ne in NumElements:
        print r'\num{{{:d}}} &'.format(Ne),
        print ' & '.join(map(lambda ET: r'\num{{{:.4e}}}'.format(cg_errors[pd][ET][Ne][1]), ETdescs)) + r' \\'
    print r'\end{tabular}'
    print r'\end{table}'

# generate plots for L2 error
for pd in pdescs:
    f = pylab.figure(figsize=(3,3))
    for et in ETdescs:
        h = map(lambda x: 1.0/x, NumElements)
        L2_errors = map(lambda x: cg_errors[pd][et][x][0], NumElements)
        pylab.loglog(h, L2_errors, styles.next())
    pylab.legend(ETdescs, loc='best', prop={'size': 8})
    pylab.title('$L_2$ error for ' + pd)
    pylab.ylabel('$L_2$ error')
    pylab.xlabel('Element size $h$ [1/$N_e$]')
    pylab.tight_layout(pad=0.2)
    make_sure_path_exists('report/figs')
    pylab.savefig('report/figs/l2_error_'+pd.split(' ')[0]+'.png')
    pylab.close(f)
        
# generate plots for L2 error
for pd in pdescs:
    f = pylab.figure(figsize=(3,3))
    for et in ETdescs:
        h = map(lambda x: 1.0/x, NumElements)
        H1_errors = map(lambda x: cg_errors[pd][et][x][1], NumElements)
        pylab.loglog(h, H1_errors, styles.next())
    pylab.legend(ETdescs, loc='best', prop={'size': 8})
    pylab.title('$H_1$ error for ' + pd)
    pylab.ylabel('$H_1$ error')
    pylab.xlabel('Element size $h$ [1/$N_e$]')
    pylab.tight_layout(pad=0.2)
    make_sure_path_exists('report/figs')
    pylab.savefig('report/figs/h1_error_'+pd.split(' ')[0]+'.png')
    pylab.close(f)

