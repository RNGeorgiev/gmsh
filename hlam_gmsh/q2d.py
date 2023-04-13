"""High-level functions to generate quasi-2D dimers and trimers
comprising disks and rods with rounded edges.

"""

from math import sin, pi, tan
import numpy as np
from .common import is_part_of, meshing, vis, new_geometry
from .common import eval_on_surf, check_CAD

PATH = os.path.split(__file__)

def get_walls(model, vol):
    vol = [(3, vol)] if isinstance(vol, int) else vol
    vol = [vol] if isinstance(vol, tuple) else vol
    surfs = np.abs(model.getBoundary(vol))
    nvs = eval_on_surf(model, surfs, 'nv')[:, 2]
    tops, bots, sides = [surfs[nvs == i] for i in [1, -1, 0]]
    edges = np.vstack((tops, bots, sides))
    edges = np.setdiff1d(surfs[:, 1], edges[:, 1])
    if len(edges):
        edges = np.array([(2, i) for i in edges])
        nvs = eval_on_surf(model, edges, 'nv')[:, 2]
        top_edges, bot_edges = [edges[i] for i in [nvs > 0, nvs < 0]]
    else:
        top_edges, bot_edges = [(2, 0)], [(2, 0)]
    surfs = [tops, bots, sides, top_edges, bot_edges]
    surfs = [list(map(tuple, i)) for i in surfs]
    return surfs

def grouping(model, vol):
    walls = get_walls(model, vol)
    gs = [model.addPhysicalGroup(2, np.array(i)[:, 1]) for i in walls]
    ns = ['Top_base', 'Bottom_base', 'Sides']
    ns = enumerate(ns + ['Top_edge', 'Bottom_edge'])
    foo = [model.setPhysicalName(2, gs[i], j) for i,j in ns]
    return

def fillet_Z(model, vol, rZ):
    geo = check_CAD(model)
    bases = sum(get_walls(model, vol)[:2], [])
    zEdges = np.abs(model.getBoundary(bases, False))[:, 1]
    filleted = geo.fillet([vol[0][1]], zEdges, [rZ])
    geo.synchronize()
    return filleted

def fillet_XY(model, vol, rXY):
    geo = check_CAD(model)
    bounds = model.getBoundary
    sides = get_walls(model, vol)[2]
    xyEdges = np.abs(bounds(sides, False))[:, 1]
    xyEdges = np.setdiff1d(xyEdges, np.abs(bounds(sides))[:, 1])
    filleted = geo.fillet([vol[0][1]], xyEdges, [rXY])
    geo.synchronize()
    return filleted

def fillets(model,vol,rZ,rXY):
    geo = check_CAD(model)
    geo.synchronize()
    if rXY: vol = fillet_XY(model, vol, rXY)
    if rZ: vol = fillet_Z(model,vol,rZ)
    return vol

def disk(r, H, rZ=0.1, partOf=None,
         gs=False, ls=0, viz=True):
    model,geo = is_part_of(partOf)
    cyl = [(3, geo.addCylinder(0, 0, -H/2, 0, 0, H, r))]
    cyl = fillets(model, cyl, rZ, 0)
    if gs : grouping(model, cyl)
    if ls : meshing(model, ls, cyl)
    if viz : vis()
    return cyl

def ring(R, r, H, rZ=0.1, partOf=None,
         gs=False, ls=0, viz=True):
    model, geo = is_part_of(partOf)
    if rZ >= ((R - r) / 2) or rZ >= H / 2:
        ring = [(3, geo.addTorus(0, 0, 0, (R + r) / 2, (R - r) / 2))]
    else:
        cO = [(3, geo.addCylinder(0, 0, -H / 2, 0, 0, H, R))]
        cI = [(3, geo.addCylinder(0, 0, -H / 2, 0, 0, H, r))]
        ring = geo.cut(cO, cI)[0]
        ring = fillets(model, ring, rZ, 0)
        if gs: grouping(model, ring)
    geo.synchronize()
    if ls: meshing(model, ls, ring)
    if viz: vis()
    return ring

def rod(l, w, H, rZ=0.1, rXY=0.1 ,partOf=None,
        gs=False, ls=0,viz=True):
    model, geo = is_part_of(partOf)
    box = [(3, geo.addBox(-l/2, -w/2, -H/2, l, w, H))]
    box = fillets(model, box, rZ, rXY)
    if gs: grouping(model, box)
    if ls: meshing(model, ls, box)
    if viz: vis()
    return box

def prism(R, a, H, rZ=0, rXY=0, partOf=None,
          gs=False, ls=0, viz=True):
    model, geo = is_part_of(partOf)
    p1X = R + rXY * (1 / sin(a / 360 * pi) - 1)
    dx = (R - rXY) * sin(pi * (1 / 2 - a / 180)) + rXY
    dy = tan(a / 360 * pi) * (p1X + dx)
    p1 = geo.addPoint(p1X, 0, -H / 2)
    p2 = geo.addPoint(-dx, dy, -H / 2)
    p3 = geo.addPoint(-dx, -dy, -H / 2)
    ps = [p1, p3, p2, p1]
    lines = [geo.addLine(ps[i], ps[i + 1]) for i in range(3)]
    bottom = geo.addPlaneSurface([geo.addCurveLoop(lines)])
    prism = geo.extrude([(2, bottom)], 0, 0, H)[1:2]
    prism = fillets(model, prism, rZ, rXY)
    if gs: grouping(model, prism)
    if ls: meshing(model, ls, prism)
    if viz: vis()
    return prism

def dimer(k, H, rZ=0.1, rS=0.56, rd=[1.85, 0.41],
          gs=True, ls=0, viz=True):
    model, geo, _ = new_geometry()
    dXYZs = [[rd[0] * i, 0, 0] for i in [1, 1/2]]
    dL, dS = [disk(i * rS, H, rZ, model, viz=False) for i in [k, 1]]
    rd = rod(*rd, H, rZ, 0, model, viz=False)
    objs = enumerate([dS, rd])
    [geo.translate(j, *dXYZs[i]) for i,j in objs]
    dimer = geo.fuse(dL, dS + rd)[0]
    geo.synchronize()
    if gs: grouping(model, dimer)
    if ls: meshing(model, ls, dimer)
    if viz: vis()
    return dimer

def trimer(k, f, H, l=1, rZ=0.1, rS=0.56, rodUL=2 * [[1.85, 0.41]],
           gs=True, ls=0, viz=True):
    model, geo, _ = new_geometry()
    dXYZs = sum([[i[0], i[0]/2] for i in rodUL], [])
    dXYZs = np.transpose([dXYZs] + 2 * [4 * [0]])
    dM = disk(k*rS, H, rZ, model, viz=False)
    dU = disk(rS, H, rZ, model, viz=False)
    dL = disk(l*rS, H, rZ, model, viz=False)
    rU, rL = [rod(*i, H, rZ, 0, model, viz=False) for i in rodUL]
    objs = enumerate(dU + rU + dL + rL)
    [geo.translate([j],*dXYZs[i]) for i,j in objs]
    geo.rotate(dU + rU, 0, 0, 0, 0, 0, 1, f / 360 * pi)
    geo.rotate(dL + rL, 0, 0, 0, 0, 0, 1, -f / 360 * pi)
    trimer = geo.fuse(dM, dU + dL + rU + rL)[0]
    geo.synchronize()
    if gs: grouping(model, trimer)
    if ls: meshing(model, ls, trimer)
    if viz: vis()
    return trimer

## Docstrings for functions in this module ###
funcs = [get_walls, grouping, fillet_Z, fillet_XY, fillets]
funcs.extend([disk, ring, rod, prism, dimer, trimer])
path = os.path.join(PATH[0], 'docs', PATH[1][:-3], '')
for i in funcs:
    with open(path + i.__name__ + '.txt', 'r') as f:
        i.__doc__ = f.read()