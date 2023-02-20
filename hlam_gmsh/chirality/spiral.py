import numpy as np
from numpy.linalg import norm
from math import pi,cos,sin,sqrt
from gmsh_RNG.d3 import spheroid
from gmsh_RNG.common import new_geometry,is_part_of,eval_on_surf
from gmsh_RNG.common import entities_in_vol

def spiral_2d(R,d,W,dL,sign=1,f0=0):
    f = 0
    r_max = 0
    dW = d+W
    spiral = np.ndarray((0,9))
    while r_max <= R:
        r = f*(W+d)/2/pi+W/2
        x,y = r*cos(f)-W/2,r*sin(f)
        dx = dW*cos(f)-(pi*W+f*dW)*sin(f)
        dy = dW*sin(f)+(pi*W+f*dW)*cos(f)
        dx,dy = [dy,-dx]/np.sqrt(dx*dx+dy*dy)*W/2
        iX,iY,oX,oY = x-dx,y-dy,x+dx,y+dy
        r_max = max([r_max,sqrt(iX**2+iY**2)])
        spiral = np.vstack((spiral,[r,r_max,f,x,y,iX,iY,oX,oY]))
        df = dL/sqrt((W+f*dW/pi)**2+(dW/2*pi)**2)
        f = f+df
    spiral[:,4::2] = spiral[:,4::2] if sign>0 else -spiral[:,4::2]
    if f0:
        addZs = np.zeros((spiral.shape[0],1))
        for i in range(3,9,2):
            tmp = np.hstack((spiral[:,i:i+2],addZs))
            rot = Rotation.from_rotvec([0,0,f0])
            spiral[:,i:i+2] = rot.apply(tmp)[:,:-1]
    return spiral.T

def spiral_3d(R,d,W,dL,sign,z,partOf=[],ls=0,viz=False):
    model,geo,mesh,opts = is_part_of(partOf)
    spiral2D = spiral_2d(R,d,W,dL,sign,0)
    oXYZ,iXYZ = spiral2D[5:7][:,::-1],spiral2D[7:9]
    oXYZ = np.vstack((oXYZ,z[0]*np.ones(oXYZ.shape[1])))
    iXYZ = np.vstack((iXYZ,z[0]*np.ones(iXYZ.shape[1])))
    origin = geo.addPoint(0,0,z[0])
    oPs = [geo.addPoint(*i) for i in oXYZ.T]
    iPs = [geo.addPoint(*i) for i in iXYZ.T]
    iL = geo.addBSpline(iPs)
    oL = geo.addBSpline(oPs)
    eL = geo.addLine(iPs[-1],oPs[0])
    sC = [iPs[0],origin,oPs[-1]]
    sC = geo.addCircleArc(*sC[::sign])
    spiral = geo.addCurveLoop([oL,eL,iL,sC])
    spiral = [(2,geo.addPlaneSurface([spiral]))]
    spiral = geo.extrude(spiral,0,0,np.diff(z)[0])[1:2]
    geo.remove([(0,i) for i in oPs+iPs])
    geo.remove([(0,origin)])
    geo.synchronize()
    meshing(mesh,ls,spiral) if ls else None
    vis(0) if viz else None
    return spiral

def cut_spiral(model,geo,spiral,body,semi=False):
    body = [body] if isinstance(body,tuple) else body
    spiral = [spiral] if isinstance(spiral,tuple) else spiral
    cp = geo.copy
    cutBody = geo.cut(cp(body),cp(spiral))[0]
    cutSpiral = geo.intersect(cp(spiral),cp(body))[0]
    if semi:
        geo.synchronize()
        z,x = np.abs(model.getBoundingBox(*body[0])[2:4])
        box = [(3,geo.addBox(-1.2*x,-1.2*x,0,2.4*x,2.4*x,1.2*z))]
        cutBody = geo.cut(cutBody,box)[0]
    geo.synchronize()
    return cutBody,cutSpiral

def spiral_points(model,geo,body,spiral):
    bb = model.getBoundingBox
    zBody,zSpiral = [list(np.round(bb(*i),6)[[2,5]]) for i in [body,spiral]]
    flag = np.argmax(np.abs(zSpiral))
    if flag:
        zSpiral.pop(1) if zSpiral[1]>zBody[1] else None
    else:
        zSpiral.pop(0) if zSpiral[0]<zBody[0] else None
    _,_,_,xyz = entities_in_vol(model,spiral)
    W = norm(xyz[xyz[:,2] == zSpiral[0]],axis=1).min()
    spiralT,bodyT = cut_spiral(model,geo,spiral,body)
    geo.synchronize()
    _,_,_,xyz = entities_in_vol(model,spiralT)
    filters = [[xyz[:,2] == i] for i in zSpiral]
    foo = [i.append(~(norm(xyz[:,:2],axis=1) == W)) for i in filters]
    xyz = xyz[np.any(np.all(filters,1),0)]
    geo.remove(spiralT+bodyT,True)
    geo.synchronize()
    return xyz

def angles(model,geo,bodies,spirals):
    xyz = [[],np.ndarray((0,3))]
    n = len(bodies)-1
    if n:
        xyz[1] = spiral_points(model,geo,bodies[1],spirals[1])
    xyz[0] = spiral_points(model,geo,bodies[0],spirals[0])
    f0,flag = 0,True
    while flag:
        fA = np.arctan2(*np.vstack(xyz)[:,1::-1].T)+f0
        fN = np.arctan2(*xyz[0][:,1::-1].T)+f0
        df = sum(fN[np.isclose(xyz[0][:,2],0)])-f0 if n else 0
        flag = np.any(np.abs(fA)<=0.2)
        f0 = f0+0.01 if flag else f0
    return f0,df

def grouping(model,body,spiral,act=True):
    sA,sP = (spiral,body) if act else ([],spiral+body)
    ns = ['Active','Passive']
    gs = enumerate([list(np.array(i)[:,1]) for i in [sA,sP]])
    gs = [model.addPhysicalGroup(2,j,-1,ns[i]) for i,j in gs]
    return