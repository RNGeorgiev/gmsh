"""Collection of functions for mesh reading, writing, and re-ordering
by proximity.

"""
import os
import numpy as np
import pandas as pd
from io import StringIO
from csv import QUOTE_NONNUMERIC as quote

PATH = os.path.split(__file__)

def fast_2D_where(a, rows=True, cols=True):
    w = a.shape[1]
    a = a.ravel()
    ls = np.bincount(a)
    idxs = np.argsort(a, kind='mergesort')
    cs = np.hstack((0, np.cumsum(ls)))
    if rows and cols:
        out = np.transpose(np.divmod(idxs, w)).ravel()
        cs = 2 * cs
    elif rows:
        out = idxs // w if rows else idxs % w
    cs = np.vstack((cs[:-1], cs[1:])).T
    out = [out[slice(*i)] for i in cs]
    return ls, out

def node_in_element_at_position(es):
    ls, idxs = fast_2D_where(es)
    ndInAt = - np.ones((len(ls), ls.max() * 2 + 1), int)
    ndInAt[:, 0] = ls
    ls = 2 * ls + 1
    for i in range(len(ls)):
        ndInAt[i, 1:ls[i]] = idxs[i]
    return ndInAt

def mesh_reorder(ps, es, ndInAt):
    idxPs = [np.argmax(ndInAt[:, 0])]
    last = [np.argmax(ndInAt[:, 0])]
    while len(idxPs) < len(ndInAt):
        newEs = pd.unique(ndInAt[last][:, 1::2].ravel())
        newEs = np.delete(newEs, newEs == -1)
        last = pd.Index(pd.unique(es[newEs].ravel()))
        last = last.difference(pd.Index(idxPs), sort=False)
        idxPs = np.hstack((idxPs, last))
    idxEs = pd.unique(ndInAt[idxPs, 1::2].ravel())
    idxEs = np.delete(idxEs, idxEs == -1)
    idxE2s = np.argsort(pd.unique(es[idxEs].ravel()))
    return ps[idxPs], idxE2s[es[idxEs]], idxEs

def data2txt(df, key, N=True):
    head, tail = [i + key for i in ['$', '$End']]
    settings = {'header': None, 'index': None, 'quoting': quote}
    fdf = df.to_csv(None, ' ', **settings)
    mid = str(len(df)) if N else ''
    out = [i for i in [head, mid, fdf[:-1], tail] if i]
    return '\n'.join(out) + '\n'

def mwrite(fileName, ps, es, gs=None):
    fGs = ''
    nPs, nEs = [i.shape[0] for i in [ps, es]]
    iPs, iEs = [range(1, i + 1) for i in [nPs, nEs]]
    tEs, dEs, eEs, gEs = np.tile([9, 2, 1, 0], (nEs, 1)).T
    dfPs = pd.DataFrame(np.vstack((iPs, *ps.T)).T)
    fPs = data2txt(dfPs.astype({0: int}), 'Nodes')
    if gs:
        enum = enumerate(gs.keys())
        pdGs = pd.DataFrame([(2, i + 1, j) for i,j in enum])
        fGs = data2txt(pdGs, 'PhysicalNames')
        for i,j in enumerate(gs.keys()):
            gEs[gs[j]] = i + 1
    dfEs = np.vstack((iEs, tEs, dEs, gEs, gEs, es.T + 1)).T
    fEs = data2txt(pd.DataFrame(dfEs), 'Elements')
    dfM = pd.DataFrame([2.2, 0, 8]).T
    dfM = dfM.astype({1: int, 2: int})
    fM = data2txt(dfM, 'MeshFormat', False)
    with open(fileName, 'w') as f:
        f.write(fM + fGs + fPs + fEs)
    return

def f_via_float(f, ps, es, gEs):
    fPs = np.repeat(f, ps.shape[0])
    fEs = np.mean(fPs[es], 1)
    return fPs, fEs

def f_via_str(f, ps, es, gEs):
    xs, ys, zs = ps.T
    fPs = np.array(eval(f))
    fEs = np.mean(fPs[es], 1)
    return fPs, fEs

def f_via_dict(f, ps, es, gEs):
    _, rows = fast_2D_where(es, cols=False)
    fEs = np.zeros(es.shape[0])
    m = f.pop('method') if 'method' in f.keys() else None
    [np.put(fEs, gEs[i], f[i]) for i in f.keys()]
    fPs = np.array([fEs[i].mean() for i in rows])
    fPs = m(fPs) if m else fPs
    return fPs, fEs

def f_from_var(rule, ps, es, gEs):
    if type(rule) is int: rule = float(rule)
    if type(rule) not in [str, float, dict]:
        msg = 'The variable must be an int, a float'
        raise Exception(msg + ', a str, or a dict.')
    f = 'f_via_' + type(rule).__name__
    return eval(f)(rule, ps, es, gEs)

def txt2data(msh, key):
    j0 = msh.index('$' + key) + 2
    nJs = int(msh[j0-1])
    js = StringIO('\n'.join(msh[j0:j0+nJs]))
    js = pd.read_csv(js, sep=' ', header=None, )
    return js, nJs

def mread(mshFile, acts, mobs):
    physGs = 'PhysicalNames'
    with open(mshFile) as f:
        msh = f.read().splitlines()
    strings = ['Nodes', 'Elements']
    if '$' + physGs in msh: strings.append(physGs)
    data = [txt2data(msh, i) for i in strings]
    nPs, nEs = [i[1] for i in data[:-1]]
    ps = data[0][0].to_numpy()[:, 1:]
    e_flags = data[1][0][1] == 9
    es = data[1][0].to_numpy()[e_flags, -6:] - 1
    msg = 'Some elements removed (not curved triangles).'
    if nEs != es.shape[0]: print(msg)
    if nEs != es.shape[0]: nEs = es.shape[0]
    ndInAt = node_in_element_at_position(es)
    if not np.all(ndInAt[:, 0]):
        errMsg = 'Point(s) in ' + mshFile
        raise Exception(errMsg + ' not part of any element.')
    if len(strings) == 3:
        nGs = data[2][1]
        tags = data[1][0].to_numpy()[e_flags, 3]
        gs = dict(zip(*[data[2][0][i] for i in [2, 1]]))
        gEs = {i: np.where(tags == gs[i])[0] for i in gs.keys()}
    ps, es, idxEs = mesh_reorder(ps, es, ndInAt)
    args = idxEs.argsort()
    gEs = {i: args[gEs[i]] for i in gEs.keys()} if gEs else {}
    aPs, aEs = f_from_var(acts, ps, es, gEs)
    mPs, mEs = f_from_var(mobs, ps, es, gEs)
    vals = [ps, es, nPs, nEs, gEs, aPs, mPs, aEs, mEs]
    msh = abbreviated_keys_dict(vals, 'mesh')
    return msh

def basic_read(mshFile):
    with open(mshFile) as f:
        msh = f.read().splitlines()
    strings = ['Nodes', 'Elements']
    data = [txt2data(msh, i) for i in strings]
    ps = data[0][0].to_numpy()[:, 1:]
    e_flags = data[1][0][1] == 9
    es = data[1][0].to_numpy()[e_flags, -6:] - 1
    return ps, es

### Docstrings for functions in this package ###
funcs = [fast_2D_where, node_in_element_at_position, mesh_reorder]
funcs.extend([data2txt, mwrite, f_via_float, f_via_str, f_via_dict])
funcs.extend([f_from_var, txt2data, mread, basic_read])
path = os.path.join(PATH[0], 'docs', PATH[1][:-3], '')
for i in funcs:
    with open(path + i.__name__ + '.txt', 'r') as f:
        i.__doc__ = f.read()