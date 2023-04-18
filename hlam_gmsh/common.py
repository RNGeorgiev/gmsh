"""Common functions used by the other modules.

"""
import os
import gmsh
import trimesh
import numpy as np
import multiprocessing as mp
from matplotlib import pyplot as plt
from .tools import basic_read

PATH = os.path.split(__file__)

def abbreviated_keys_dict(vals, dict_name):
    vals = sum(zip(vals, vals), ())
    path = os.path.split(__file__)
    path = os.path.join(path[0], 'src', 'dict_keys.txt')
    with open(path, 'r') as file:
        txt = file.read().splitlines()
    heads = [i.split(':')[0] for i in txt]
    keys = txt[heads.index(dict_name)]
    keys = keys.split(':')[1].split('//')
    long, short = [i.split(',') for i in keys]
    keys = sum(zip(long, short), ())
    return dict(zip(keys, vals))

def restart_GMSH():
    if gmsh.is_initialized(): gmsh.finalize()
    gmsh.initialize()
    return

def new_geometry(occ=True):
    restart_GMSH()
    model = gmsh.model
    geo = gmsh.model.occ if occ else gmsh.model.geo
    mesh = gmsh.model.mesh
    opts = gmsh.option
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count())
    opts.setNumber("General.NumThreads", mp.cpu_count())
    if occ: opts.setNumber("Geometry.OCCParallel", 1)
    return model, geo, opts

def check_CAD(model):
    occs = [model.occ.getMaxTag(i) for i in range(4)]
    geos = [model.geo.getMaxTag(i) for i in range(4)]
    if sum(occs + geos):
        flag = np.argwhere(np.any([occs, geos], 1))[0, 0]
        geo = model.geo if flag else model.occ
    else:
        geo = model.occ
    return geo

def is_part_of(model):
    if model:
        geo = check_CAD(model)
    else:
        model, geo, _ = new_geometry()
    return model, geo

def vis(restart=False, file=None):
    if restart: restart_GMSH()
    if file: gmsh.open(file)
    gmsh.fltk.run()
    return

def eval_on_surf(model, surfs, f, x0=0.25, y0=0.25):
    res = []
    vals = [model.getNormal, model.getValue, model.getCurvature]
    fDict = abbreviated_keys_dict(vals, 'eval_surf')
    if isinstance(surfs, int): surfs = [(2, surfs)]
    if isinstance(surfs, tuple): surfs = [surfs]
    surfs = list(map(tuple, np.abs(surfs)))
    for surf in np.abs(surfs):
        bounds = model.getParametrizationBounds(*surf)
        x = bounds[1][0] * x0 + (1 - x0) * bounds[0][0]
        y = bounds[1][1] * y0 + (1 - y0) * bounds[0][1]
        pars = list(surf) + [[x, y]]
        pars = pars[1:] if f in ['nv', 'normal'] else pars
        res.append(fDict[f](*pars))
    return np.squeeze(res)

def entities_in_vol(model, vol):
    bounds = model.getBoundary
    evalM = model.getValue
    if isinstance(vol, int): vol = (3, vol)
    if isinstance(vol, tuple): vol = [vol]
    surfs = bounds(vol)
    lines = np.array(bounds(surfs, False))
    lines = np.unique(np.abs(lines[:, 1]))
    lines = [(1, i) for i in lines]
    points = np.array(bounds(lines, False))[:, 1]
    points = [(0, i) for i in np.unique(points)]
    xyz = np.array([model.getValue(*i, []) for i in points])
    return surfs, lines, points, xyz

def unify_surfs(model, vol):
    geo = check_CAD(model)
    geo.synchronize()
    surfs = np.array(entities_in_vol(model, vol)[0])
    model.addPhysicalGroup(2, surfs[:, 1], -1, 'Surface')
    return

def meshing(model, ls, vol=-1):
    writing, mesh = False, model.mesh
    if isinstance(vol, list): vol = vol[0]
    if isinstance(vol, tuple): vol = vol[1]
    if vol > 0: mesh.setOutwardOrientation(vol)
    if isinstance(ls, list):
        ls, name, writing = *ls, True
        if name[-4:] != '.msh': name += '.msh'
    gmsh.option.setNumber("Mesh.MeshSizeMax", ls)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2)
    mesh.generate(2)
    mesh.setOrder(2)
    if writing: gmsh.write(name)
    if writing: gmsh.open(name)
    elms = mesh.get_elements(2)[1][0]
    ps = mesh.get_nodes()[1].reshape(-1, 3)
    msg = f'Mesh: {len(ps)} points in {len(ps)} elements.'
    stat = ['mean),','SD),','worst).']
    for k in ['minSICN','gamma']:
        es = mesh.get_element_qualities(elms, k)
        qs = np.round([es.mean(), es.std(), es.min()], 2)
        dat = ' '.join([f'{j} ({i}'  for i,j in zip(stat, qs)])
        print(f'Element quality ({k}): ' + dat)
        plt.figure()
        plt.hist(es, 25)
        plt.xlabel(f'Element quality ({k})')
        plt.ylabel(f'Count (total: {len(es)})')
    if writing:
        ps, elms = basic_read(name)
    else:
        elms = mesh.get_elements(2)[2][0].reshape(-1, 6) - 1
    flatTris = [[0, 3, 5], [1, 4, 3], [2, 5, 4], [3, 4, 5]]
    flatTris = elms[:, flatTris].reshape(-1, 3)
    trim = trimesh.Trimesh(ps, flatTris)
    chi = trim.euler_number
    flag = ' ' if trim.is_watertight else ' not '
    print(f'Mesh is{flag}watertight!')
    print(f'Genus (holes): {1 - chi // 2}')
    if not trim.is_watertight:
        trimesh.repair.broken_faces(trim, color=[255, 0, 0, 255])
        a = trimesh.scene.scene.Scene(trim)
        a.show('gl', smooth=True, resolution=(750, 750))
    return

## Docstrings for functions in this module ###
funcs = [abbreviated_keys_dict, restart_GMSH, new_geometry]
funcs.extend([check_CAD, is_part_of, vis, eval_on_surf])
funcs.extend([entities_in_vol, meshing])
path = os.path.join(PATH[0], 'docs', PATH[1][:-3], '')
for i in funcs:
    with open(path + i.__name__ + '.txt', 'r') as f:
        i.__doc__ = f.read()