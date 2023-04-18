"""High-level functions to generate an corrugated wall,
whose flat parts have a normal (1, 0, 0).

"""

import os
import numpy as np
from .common import new_geometry, vis, meshing

PATH = os.path.split(__file__)

def flat(w, h, ls, viz=True):
    model, geo, _ = new_geometry()
    xyzs = np.array([[0, 0], [w, 0], [w, h], [0, h]]) - [w / 2, h / 2]
    order = [0, 1, 2, 3, 0]
    ps = np.array([geo.addPoint(0, *i) for i in xyzs])
    lines = [geo.addLine(*ps[order[i:i+2]]) for i in range(4)]
    geo.addPlaneSurface([geo.addCurveLoop(lines)])
    geo.synchronize()
    if ls: meshing(model, ls)
    if viz: vis()
    return

def triangle(A, N, l):
    xs = l * np.arange(-N / 2, N / 2 + 1 / 8, 0.25)
    ys = np.array([0] + N * [A, 0, -A, 0])
    return xs, ys

def sawtooth(A, N, l):
    xs1 = np.arange(-N / 2, N / 2 + 0.5)
    xs2 = 2 * [xs1[i:i+2].mean() for i in range(len(xs1)-1)]
    xs = l * np.sort(np.hstack((xs1, xs2)))
    ys = np.array((len(xs1) - 1) * [0, A/2, -A / 2] + [0])
    return xs, ys

def square(A, N, l):
    xs = np.arange(-N / 2, N / 2 + 0.25, 0.5)
    xs = l * np.sort(np.hstack((xs[0], xs[-1], xs, xs[1:-1])))
    ys = np.array([0] + N * [A, A, -A, -A] + [0])
    return xs, ys

def sine(A, N, l):
    xs = np.linspace(-N / 2, N / 2, N * 100 + 1) * l
    ys = A * np.sin(xs / l * 2 * np.pi)
    return xs, ys

def refinement(model, curves, surfs,  opts, ls, ratio):
    setOpt = opts.setNumber
    field = model.mesh.field
    setOne = field.setNumber
    setMany = field.setNumbers
    if isinstance(ls, list): ls = ls[0]
    keys = ['Distance', 'Threshold','Constant', 'Min']
    f1, f2, f3, f4 = [field.add(i) for i in keys]
    setMany(f1, "CurvesList", curves)
    setOne(f1, "Sampling", 200)
    keys = ['InField', 'SizeMin', 'SizeMax', 'DistMin', 'DistMax']
    vals = [f1, ls / ratio, ls, ls, 5 * ls]
    [setOne(f2, i, j) for i,j in zip(keys, vals)]
    setMany(f3, "SurfacesList", surfs)
    keys, vals = ['VIn', 'VOut'], [ls / ratio, ls]
    [setOne(f3, i, j) for i,j in zip(keys, vals)]
    setMany(f4, "FieldsList", [f2, f3])
    field.setAsBackgroundMesh(f4)
    setOpt("Mesh.MeshSizeExtendFromBoundary", 0)
    setOpt("Mesh.MeshSizeFromPoints", 0)
    setOpt("Mesh.MeshSizeFromCurvature", 0)
    setOpt("Mesh.Algorithm", 5)
    return

def wavy_wall(waveform, w, A, N, l, q, above, below,
              ls=0, ratio=1, viz=True):
    model, geo, opts = new_geometry()
    zs, xs = eval(waveform)(A, N, l)
    ys = np.repeat(-w / 2, len(xs))
    xyz = np.transpose([xs, ys, zs])
    ps = [geo.addPoint(*i) for i in xyz]
    if waveform == 'sine':
        profile = [(1, geo.add_bspline(ps))]
        geo.remove([(0, i) for i in ps[1:-1]])
    else:
        lines = [geo.add_line(*i) for i in zip(ps[:-1], ps[1:])]
        profile = [(1, i) for i in lines]
    dz = w * np.tan(np.radians(q))
    corrugation = geo.extrude(profile, 0, w, dz)
    curves = [i[1] for i in profile + corrugation if i[0] == 1]
    surfs = [i[1] for i in corrugation if i[0] == 2]
    bot, top = [[corrugation[i][1]] for i in [2, -1]]
    for i in [[0, above], [-w, 0], [0, -above - dz]]:
        lastPoint = geo.get_max_tag(0)
        top.append(geo.extrude([(0, lastPoint)], 0, *i)[1][1])
    bot.append(geo.extrude([(0, 1)], 0, 0, -below)[1][1])
    for i in [[w, 0], [0, below + dz]]:
        lastPoint = geo.get_max_tag(0)
        bot.append(geo.extrude([(0, lastPoint)], 0, *i)[1][1])
    geo.add_surface_filling(geo.add_wire(top[::-1]))
    geo.add_surface_filling(geo.add_wire(bot[::-1]))
    geo.synchronize()
    if ls: refinement(model, curves, surfs, opts, ls, ratio)
    if ls: meshing(model, ls)
    if viz: vis()
    return

### Docstrings for functions in this package ###
funcs = [flat, triangle, sawtooth, square, sine]
funcs.extend([refinement, wavy_wall])
path = os.path.join(PATH[0], 'docs', PATH[1][:-3], '')
for i in funcs:
    with open(path + i.__name__ + '.txt', 'r') as f:
        i.__doc__ = f.read()