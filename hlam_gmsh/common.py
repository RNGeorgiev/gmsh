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

def geo_centers(obj, densities=[]):
    com = gmsh.model.occ.get_center_of_mass
    mass = gmsh.model.occ.get_mass
    bounds = gmsh.model.get_boundary
    c_mass, massT = [], []
    if obj[0][0] == 2: surfs = obj
    if obj[0][0] == 3:
        if not len(densities): densities = len(obj) * [1]
        densities = np.array(densities)
        centers = np.array([com(*i) for i in obj])
        weights = densities * [mass(*i) for i in obj]
        massT = weights.sum()
        c_mass = (weights * centers.T).sum(1) / massT
        surfs = list(map(tuple, np.abs(bounds(obj))))
    centers = np.array([com(*i) for i in surfs])
    weights = [mass(*i) for i in surfs]
    areaT = sum(weights)
    c_surf = (weights * centers.T).sum(1) / areaT
    return c_mass, massT, c_surf, areaT

def mesh_quality(mesh):
    if type(mesh) is str:
        model, _, _ = new_geometry()
        gmsh.open(mesh)
        mesh = model.mesh
    el_qual = 'Element quality ('
    es = mesh.get_elements(2)[1][0]
    nPs = len(mesh.get_nodes()[1])//3
    stat = ['mean),', 'SD),', 'worst).']
    qs = mesh.get_element_qualities
    msg = f'Mesh: {nPs} points in {len(es)} elements.'
    q0s = np.array([qs(es, i) for i in ['minSICN', 'gamma']])
    qs = np.transpose([q0s.mean(1), q0s.std(1), q0s.min(1)])
    for i,j in zip(['minSICN','gamma'], qs):
        dat = ' '.join([f'{j:.2f} ({i}'  for i,j in zip(stat, j)])
        print(f'{el_qual}{i}): {dat}')
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
    plt.subplots_adjust(wspace=0.05)
    xlabels = [f'{el_qual}{k})' for k in ['minSICN', 'gamma']]
    [i.set_xlabel(j) for i,j in zip(axs, xlabels)]
    [i.hist(j, 25) for i,j in zip(axs, q0s)]
    axs[0].set_ylabel(f'Count (total: {len(es)})')
    return

def is_watertight(mesh, viz=False):
    if type(mesh) is str:
        ps, es = basic_read(mesh)
    else:
        ps = mesh.get_nodes()[1].reshape(-1, 3)
        es = mesh.get_elements(2)[2][0].reshape(-1, 6) - 1
    flatTris = [[0, 3, 5], [1, 4, 3], [2, 5, 4], [3, 4, 5]]
    flatTris = es[:, flatTris].reshape(-1, 3)
    trim = trimesh.Trimesh(ps, flatTris)
    chi = trim.euler_number
    flag = ' ' if trim.is_watertight else ' not '
    if not trim.is_watertight: viz = True
    print(f'Mesh is{flag}watertight!')
    print(f'Genus (holes): {1 - chi // 2}')
    if viz:
        trimesh.repair.broken_faces(trim, color=[255, 0, 0, 255])
        a = trimesh.scene.scene.Scene(trim)
        a.show('gl', smooth=True, resolution=(900, 900))
    return

def meshing(model, ls, vol=-1):
    writing, mesh = False, model.mesh
    if isinstance(vol, list): vol = vol[0]
    if isinstance(vol, tuple): vol = vol[1]
    if vol > 0: mesh.setOutwardOrientation(vol)
    if isinstance(ls, list):
        ls, name, writing = *ls, True
        if name[-4] != '.': name += '.msh'
    gmsh.option.setNumber("Mesh.MeshSizeMax", ls)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2)
    mesh.generate(2)
    mesh.setOrder(2)
    if writing: gmsh.write(name)
    mesh_quality(mesh)
    is_watertight(mesh)
    return

## Docstrings for functions in this module ###
funcs = [abbreviated_keys_dict, restart_GMSH, new_geometry]
funcs.extend([check_CAD, is_part_of, vis, eval_on_surf])
funcs.extend([entities_in_vol, unify_surfs, geo_centers])
funcs.extend([mesh_quality, is_watertight, meshing])
path = os.path.join(PATH[0], 'docs', PATH[1][:-3], '')
for i in funcs:
    with open(path + i.__name__ + '.txt', 'r') as f:
        i.__doc__ = f.read()