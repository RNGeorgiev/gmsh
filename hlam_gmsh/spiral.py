"""High-level functions to generate a spheroid with
an spiral(loxodrome) active strip.  

"""

import os
import numpy as np
from math import pi
from numpy.linalg import norm
from scipy.special import ellipe
from ..d3 import spheroid
from ..common import check_CAD,new_geometry,eval_on_surf,vis,meshing

def spiral_2d(R,T,sign=1,f0=0,p=1,df=0.01):
    """Generates points on an Archimedean spiral with T turns
    and maximum radius R. The direction of the spiral is set
    via sign and its starting angle is given by f0. The inverse
    of p (1/p) controls the shape of a the spiral. The angle step
    is set through df.

    Parameters
    ----------
    R : float
        Outer radius of the spiral.
    T : float
        Number of turns.
    sign : int, default=1
        Sets the spiral's direction of rotation:
        counter-clockwise with 1 and clockwise with -1.
    f0 : float, default=0
        Starting angle of the spiral.
    p : float, default=1
        The spiral shape is set to 1/p
        (https://mathworld.wolfram.com/ArchimedeanSpiral.html).
    df: float, default=0.01
        Angle increment.
    
    Returns
    -------
    numpy array,numpy array
        XY-coordinates of the spiral points.
    
    """
    N = int(T*pi/df)+1
    fs = np.linspace(0,2*T*pi*sign,N)
    b = R/(T*2*pi)**(1/p)
    rs = np.sign(fs)*b*np.abs(fs)**(1/p)
    xs = rs*np.cos(fs)
    ys = rs*np.sin(fs)
    if f0:
        xyz = np.vstack((xs,ys,np.zeros(len(xs)))).T
        rot = Rotation.from_rotvec([0,0,f0])
        xs,ys = rot.apply(xyz).T[:2]
    return xs,ys

def ellipse_perimeter(R,c):
    """Finds the perimeter of an ellipse with semiaxes R and c.

    Parameters
    ----------
    R,c : float,float
        Semiaxes of the ellipse.

    Returns
    -------
    float
        Ellipse perimeter
    
    """
    a,b = (R,c) if c>R else (c,R)
    e_sq = 1.0-b**2/a**2
    return 4*a*ellipe(e_sq)

def loxodrome(turns,R,c,sign=1,df=pi/50):
    """Generates points on a loxodrome -- a line, which crosses
    each meridian on a spheroid at a constant angle.The resulting
    spiral has T, an equatorial radius R, and pole-to-pole 
    distance 2*c*R. The rotation direction of the  loxodrome is
    given by sign. The angle step is set through df.

    Parameters
    ----------
    R : float
        Equatorial radius of the loxodrome.
    c : float
        Pole-to-pole distance in units of equatorial diameter.
    T : float
        Number of turns.
    sign : int, default=1
        Sets the loxodrome's direction of rotation:
        counter-clockwise with 1 and clockwise with -1.
    df: float, default=0.01
        Angle increment.
    
    Returns
    -------
    numpy array
        XYZ-coordinates of the loxodrome points.
    
    """
    f_max,df = [sign*i for i in [2*turns*pi,df]]
    fs = np.arange(0,f_max+df/2,df/2)
    xs = R*np.cos(fs)*np.cos((fs/f_max-1/2)*pi)
    ys = R*np.sin(fs)*np.cos((fs/f_max-1/2)*pi)
    zs = -c*np.sin((fs/f_max-1/2)*pi)
    return np.transpose([xs,ys,zs])

def spiral_tube(geo,r,turns,R,c,sign,df):
    """Generates a spiral tube by extruding a circle of radius r
    along a loxodrome. Each end of the tube is capped with a sphere
    of radius 1.1*r. The caps have a larger radius than the tube to
    The parameters turns, R, c, sign, and df 
    have the same meaning as in the loxodrome function. The 

    Parameters
    ----------
    geo : class
        Geometry, in which the spiral tube is created.
    r : float
        Tube radius.
    R : float
        Equatorial radius of the loxodrome.
    c : float
        Pole-to-pole distance in units of equatorial diameter.
    T : float
        Number of turns.
    sign : int, default=1
        Sets the loxodrome's direction of rotation:
        counter-clockwise with 1 and clockwise with -1.
    df: float, default=0.01
        Angle increment
    
    Returns
    -------
    numpy array
        XYZ-coordinates of the loxodrome points.
    
    """
    xyz = loxodrome(turns,R,c,sign,df)
    ps = [geo.addPoint(*i) for i in xyz]
    bb = geo.addWire([geo.addSpline(ps)])
    nv = xyz[1]-xyz[0]
    disk = [(2,geo.addDisk(*xyz[0],r,r,zAxis=nv))]
    pipe = geo.addPipe(disk,bb)
    geo.remove(disk+[(1,bb)],True)
    geo.remove([(0,i) for i in ps])
    caps = [(3,geo.addSphere(*xyz[i],r*1.1)) for i in [0,-1]]
    spiral = geo.fuse(pipe,caps)[0]
    geo.synchronize()
    return spiral

def clipping(geo,spiral,zs,R,c):
    """Removes parts of the spiral which are outside the  
    Parameters
    ----------
    geo : class
        Geometry, in which the spiral tube is created.
    r : float
        Tube radius.
    R : float
        Equatorial radius of the loxodrome.
    c : float
        Pole-to-pole distance in units of equatorial diameter.
    T : float
        Number of turns.
    sign : int, default=1
        Sets the loxodrome's direction of rotation:
        counter-clockwise with 1 and clockwise with -1.
    df: float, default=0.01
        Angle increment
    
    Returns
    -------
    numpy array
        XYZ-coordinates of the loxodrome points.
    
    """
    zs[0] = -1.1 if zs[0]<-0.98 else zs[0]
    zs[1] = 1.1 if zs[1]>0.98 else zs[1]
    boxPars = [[-R,-R,c*i,2*R,2*R,0] for i in zs]
    boxPars = 1.1*np.array(boxPars).T
    boxPars[5] = c*np.array([-1.1,1.1])-boxPars[2]
    boxes = [geo.addBox(*i) for i in boxPars.T]
    spiral = geo.cut(spiral,[(3,i) for i in boxes])[0]
    geo.synchronize()
    return spiral

def get_surf(model,volTag,R,c):
    geo = check_CAD(model)
    surfs = model.getBoundary(volTag)
    vals = eval_on_surf(model,surfs,'val')
    vect = [R**2,R**2,c**2]
    flags = np.isclose(np.sum(vals**2/vect,1),1)
    surfs = [surfs[i] for i,j in enumerate(flags) if j]
    surfs = geo.copy(list(map(tuple,np.abs(surfs))))
    surf = geo.fuse(surfs[:1],surfs[1:])[0] if len(surfs)>1 else surfs
    return surf

def grouping(model,spiral,body):
    walls = [[i[1] for i in j] for j in [spiral,body]]
    gs = [model.addPhysicalGroup(2,i) for i in walls]
    ns = enumerate(['Active','Passive'])
    foo = [model.setPhysicalName(2,gs[i],j) for i,j in ns]
    return

def spheroidal_spiral(turns,act,R,c,zs=[-1,1],
                      sign=1,df=pi/50,ls=0,viz=True):
    model,geo,opts = new_geometry()
    spiral = spiral_tube(geo,act/2,turns,R,c,sign,df)
    spiral = clipping(geo,spiral,zs,R,c)
    body = spheroid(R,c,model)
    geo.rotate(body,0,0,0,0,0,1,-sign*pi)
    cutBody = geo.cut(body,spiral,-1,False,False)[0]
    cutSpiral = geo.intersect(spiral,body)[0]
    geo.synchronize()
    spiral = get_surf(model,cutSpiral,R,c)
    body = get_surf(model,cutBody,R,c)
    geo.remove(cutSpiral+cutBody,True)
    spiral_body = geo.fragment(spiral,body)[0]
    geo.synchronize()
    grouping(model,spiral,body)
    #opts.setNumber('Mesh.Algorithm',1)
    meshing(model, ls) if ls else None
    vis(0) if viz else None
    return spiral_body