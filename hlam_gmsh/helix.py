"""High-level functions to generate an active helix
starting and ending with spherical caps.

"""

import os
import numpy as np
from numpy.linalg import norm
from math import pi, cos, sin, atan
from scipy.spatial.transform import Rotation
from .common import new_geometry, vis, meshing,
from .common import check_CAD, eval_on_surf

PATH = os.path.split(__file__)

def align_vectors(v1, v2):
    v1, v2 = [np.array(i) for i in [v1, v2]]
    v1, v2 = [i / norm(i) for i in [v1, v2]]
    v, c = [i(v1, v2) for i in [np.cross, np.dot]]
    k = -np.cross(np.eye(3), v)
    rotM = np.eye(3) + k + k@k / (1 + c)
    rotM = Rotation.from_matrix(rotM)
    rotM = rotM.as_euler('XYZ')
    rotM = np.vstack((np.eye(3), rotM)).T
    return rotM

def backbone(turns, R, l):
    model, geo, opts = new_geometry()
    res = sum(turns) if isinstance(turns, list) else turns
    opts.setNumber("Geometry.NumSubEdges", int(200 * res))
    Ns = [0] + turns if isinstance(turns, list) else [0, turns]
    ts = np.arange(0, sum(Ns) + 5e-3, 1e-2)
    xs = R * np.cos(ts * 2 * pi)
    ys = R * np.sin(ts * 2 * pi)
    zs = ts * l
    xyz = np.transpose([xs, ys, zs])
    ps = [geo.addPoint(*i) for i in xyz]
    ranges = np.array(np.ceil(np.cumsum(Ns) * 1e2), int)
    ranges = list(zip(ranges[:-1], ranges[1:] + 1))
    geo.synchronize()
    return model, xyz, ps, ranges

def start_cap(model, xyz, r, xi, q):
    spring = []
    geo = check_CAD(model)
    bounds = model.getBoundary
    f = 2 * xi * pi if 0 < xi < 1 else pi
    f0 = atan(1e2 * xyz[1, 2] / 2 / pi / xyz[0, 0])
    pars = [[*xyz[0], r, -1, 0, pi / 2, i] for i in [f, 2 * pi - f]]
    capA, capP = [[(3, geo.addSphere(*i))] for i in pars]
    geo.rotate(capA, *xyz[0], 0, 0, 1, q - f / 2)
    geo.rotate(capP, *xyz[0], 0, 0, 1, q + f / 2)
    geo.rotate(capA + capP, *xyz[0], 1, 0, 0, pi / 2 + f0)
    geo.synchronize()
    for cap in [capA, capP]:
        surfs = list(map(tuple, np.abs(bounds(cap))))
        nvs = eval_on_surf(model, surfs, 'nv')[:, 2]
        nvs = np.isclose(nvs, sin(f0))
        curves = eval_on_surf(model, surfs, '1/r') > 0
        spring.append([i for i,j in zip(surfs, curves) if j])
        spring.append([i for i,j in zip(surfs, nvs) if j])
    return spring

def chunk(model, ps, r, rng, spring):
    pipes = []
    geo = check_CAD(model)
    bounds = model.getBoundary
    bb = geo.addWire([geo.addSpline(ps[slice(*rng)])])
    for k in spring[1::2]:
        pipe = geo.addPipe(k[-1:], bb, 'Frenet')
        geo.synchronize()
        surfs = bounds(pipe)
        curves = eval_on_surf(model, surfs, '1/r')
        curves = np.argmin(np.abs(curves - 1 / r))
        pipes = pipes + [surfs[i] for i in [curves, -1]]
    pipe = enumerate(map(tuple, np.abs(pipes)))
    [spring[i].append(j) for i,j in pipe]
    return spring

def end_cap(model, turns, xyz, r, xi, q, spring):
    geo = check_CAD(model)
    bounds = model.getBoundary
    f = 2 * xi * pi if 0 < xi < 1 else pi
    f0 = sum(turns) % 1 * 2 * pi - q - f / 2
    nv = eval_on_surf(model, spring[-1][-1],'nv')
    rots = align_vectors(nv, [0, 0, 1])[::-1]
    pars = [[*xyz[-1], r, -1, 0, pi / 2, i] for i in [f, 2 * pi - f]]
    capA, capP = [[(3, geo.addSphere(*i))] for i in pars]
    [geo.rotate(capA + capP, *xyz[-1], *i) for i in rots]
    geo.rotate(capA, *xyz[-1], *nv, f0)
    geo.rotate(capP, *xyz[-1], *nv, f0 + f)
    geo.synchronize()
    for cap in [capA, capP]:
        offset = 0 if cap == capA else 2
        surfs = list(map(tuple, np.abs(bounds(cap))))
        curves = eval_on_surf(model, surfs, '1/r') > 0
        curves = [i for i,j in zip(surfs, curves) if j]
        spring[offset] = spring[offset] + curves
    spring = [spring[i] for i in [0, 2]]
    return spring


def assemble_helix(model,spring,xi):
    geo = check_CAD(model)
    sA, sP = [geo.copy(i) for i in spring]
    sA, sP = [geo.fuse(i[:1], i[1:])[0] for i in [sA, sP]]
    geo.remove(model.get_entities(3), True)
    helix = geo.fragment(sA, sP)[0]
    geo.synchronize()
    n = len(sA)
    sA, sP = np.array([sA, sP])
    same = [0] + list(range(1, n - 2 + n % 2, 2))
    if n % 2: same = same + [n - 1]
    other = list(set(range(n)) - set(same))
    sA, sP = [np.array(i) for i in [sA, sP]]
    if 0 < xi < 1:
        sAnew = np.vstack((sA[same], sP[other]))[:, 1]
        sPnew = np.vstack((sP[same], sA[other]))[:, 1]
    else:
        new = np.hstack((sA, sP))[:, 1::2]
        sAnew = (new[same] if xi else new[other]).ravel()
        sPnew = (new[other] if xi else new[same]).ravel()
    gs = enumerate([sAnew, sPnew])
    ns = ['Active', 'Passive']
    gs = [model.addPhysicalGroup(2, j, -1, ns[i]) for i,j in gs]
    return helix

def helix(turns, xi, R, r, l=1, q=0, ls=0, viz=True):
    msg = 'Warning! Tube radius larger than '
    msg = msg + 'maximum mesh size.'
    if (ls and r < ls): print(msg)
    q = np.radians(q)
    model, xyz, ps, ranges = backbone(turns, R, l)
    spring = start_cap(model, xyz, r, xi, q)
    for rng in ranges:
        spring = chunk(model, ps, r, rng, spring)
    spring = end_cap(model, turns, xyz, r, xi, q, spring)
    helix = assemble_helix(model, spring, xi)
    if ls: meshing(model, ls)
    if viz: vis()
    return helix

### Docstrings for functions in this package ###
funcs = [align_vectors, backbone, start_cap, chunk, end_cap]
funcs.extend([assemble_helix, helix])
path = os.path.join(PATH[0], 'docs', PATH[1][:-3], '')
for i in funcs:
    with open(path + i.__name__ + '.txt', 'r') as f:
        i.__doc__ = f.read()