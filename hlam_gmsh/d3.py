"""High-level functions to generate dimers and trimers
comprising spheres and cylinders.

"""

import os
from math import pi
from .common import new_geometry, meshing, vis, is_part_of, unify_surfs

PATH = os.path.split(__file__)

def ball(r, partOf=None,
         gs=True, ls=0, viz=False):
    model, geo = is_part_of(partOf)
    ball = [(3, geo.addSphere(0, 0, 0, r))]
    geo.synchronize()
    if gs: unify_surfs(model, ball)
    if ls: meshing(model, ls, ball)
    if viz: vis()
    return ball

def spheroid(r, c=1, partOf=None,
             gs=True, ls=0, viz=False):
    model, geo = is_part_of(partOf)
    spheroid = [(3, geo.addSphere(0, 0, 0, r))]
    t = [0, 0, 0, 1, 1, c]
    geo.dilate(spheroid, *t)
    geo.synchronize()
    if gs: unify_surfs(model, spheroid)
    if ls: meshing(model, ls, spheroid)
    if viz: vis()
    return spheroid

def dimer(k, rS=0.56, cyl=[1.85, 0.20],
          gs=True, ls=0, viz=True):
    model, geo, _ = new_geometry()
    bL = geo.addSphere(0, 0, 0, k * rS)
    bS = geo.addSphere(cyl[0], 0, 0, rS)
    cyl = geo.addCylinder(0, 0, 0, cyl[0], 0, 0, cyl[1])
    bL, bS, cyl = list(zip(3 * [3], [bL, bS, cyl]))
    dimer = geo.fuse([bL], [bS, cyl])[0]
    geo.synchronize()
    if gs: unify_surfs(model, dimer)
    if ls: meshing(model, ls, dimer)
    if viz: vis()
    return dimer

def trimer(k, f, l=1, rS=0.56, cylUL=2 * [[1.85, 0.20]],
           gs=True, ls=0, viz=True):
    model, geo, _ = new_geometry()
    bM, bU, bL = [geo.addSphere(0, 0, 0, i * rS) for i in [k, 1, l]]
    cylU = geo.addCylinder(0, 0, 0, cylUL[0][0], 0, 0, cylUL[0][1])
    cylL = geo.addCylinder(0, 0, 0, cylUL[1][0], 0, 0, cylUL[1][1])
    objs = [bM, bU, bL, cylU, cylL]
    bM, bU, bL, cylU, cylL = list(zip(5 * [3], objs))
    objs = enumerate([bU, bL])
    foo = [geo.translate([j], cylUL[i][0], 0, 0) for i,j in objs]
    geo.rotate([bU, cylU], 0, 0, 0, 0, 0, 1, f/360 * pi)
    geo.rotate([bL, cylL], 0, 0, 0, 0, 0, 1, -f/360 * pi)
    trimer = geo.fuse([bM], [bL, bU, cylL, cylU])[0]
    geo.synchronize()
    if gs: unify_surfs(model, trimer)
    if ls: meshing(model, ls, trimer)
    if viz: vis()
    return trimer

## Docstrings for functions in this module ###
funcs = [ball, spheroid, dimer, trimer]
path = os.path.join(PATH[0], 'docs', PATH[1][:-3], '')
for i in funcs:
    with open(path + i.__name__ + '.txt', 'r') as f:
        i.__doc__ = f.read()