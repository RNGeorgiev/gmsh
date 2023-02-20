"""High-level functions to generate dimers and trimers 
comprising spheres and cylinders.

"""

from math import pi
from .common import new_geometry,meshing,vis,is_part_of

def ball(r,partOf=None,ls=0,viz=False):
    """Shorthand to generate a model of a (0,0,0)-centered sphere 
    with radius r. Optionally, meshes the model, writes the mesh,
    and visualizes is. 

    Parameters
    ----------
    r : float
        Radius of the sphere.
    partOf: class, default=None
        Determines if the ball is part of a complex shape.
        For details, see common.is_part_of()
    ls : float or list, default=0
        If ls is a float it sets the maximum mesh size. If ls is
        a list, containing a float and a string, sets the maximum
        element size to the float and exports a GMSH 2.2 *.msh file
        using string as name.
    viz : bool, default=False
        A flag controlling visualisation.

    Returns
    -------
    int
        Volume tag of the ball.
    
    """
    model,geo = is_part_of(partOf)
    ball = [(3,geo.addSphere(0,0,0,r))]
    geo.synchronize()
    meshing(model,ls,ball) if ls else None
    vis() if viz else None
    return ball

def spheroid(r,c=1,partOf=None,ls=0,viz=False):
    """Generates a model of a (0,0,0)-centered spheroid with
    one semiaxis r in the xy-plane and another semi-axis c*r 
    normal to it. Optionally, meshes the model, writes the mesh,
    and visualizes is. 

    Parameters
    ----------
    r : float
        Semi-axis in the xy-plane.
    c : float
        Ratio between the two semi-axes.
    partOf: model, default=None
        Determines if the spheroid is part of a complex shape.
        For details, see common.is_part_of()
    ls : float or list, default=0
        If ls is a float it sets the maximum mesh size. If ls is
        a list, containing a float and a string, sets the maximum
        element size to the float and exports a GMSH 2.2 *.msh file
        using string as name.
    viz : bool, default=False
        A flag controlling visualisation.

    Returns
    -------
    int
        Volume tag of the ball.
    
    """
    model,geo = is_part_of(partOf)
    spheroid = [(3,geo.addSphere(0,0,0,r))]
    t = [0,0,0,1,1,c]
    geo.dilate(spheroid,*t)
    geo.synchronize()
    meshing(model,ls,spheroid) if ls else None
    vis(0) if viz else None
    return spheroid

def dimer(k,rS=0.56,cyl=[1.85,0.20],ls=0,viz=True):
    """Generates a model of a dimer comprising a large sphere of radius
    k*rS and a small sphere of radius rS connected via a cylinder of 
    length cyl[0] and radius cyl[1]. The large sphere is centered at 
    (0,0,0). The cylinder's axis is parallel to the x-axis. 
    
    Optionally, meshes the model, writes the mesh, and visualizes is.

    Parameters
    ----------
    k : float
        Ratio of the two sphere radii.
    rS : float, default=0.56
        Radius of the smaller sphere.
    cyl: list, default=[1.85,0.20]
        [Length, Radius] of the connecting cylinder.
    ls : float or list, default=0
        If ls is a float it sets the maximum mesh size. If ls is
        a list, containing a float and a string, sets the maximum
        element size to the float and exports a GMSH 2.2 *.msh file
        using string as name.
    viz : bool, default=False
        A flag controlling visualisation.

    Returns
    -------
    int
        Volume tag of the dimer.
    
    """
    model,geo,mesh,_ = new_geometry()
    bL = geo.addSphere(0,0,0,k*rS)
    bS = geo.addSphere(cyl[0],0,0,rS)
    cyl = geo.addCylinder(0,0,0,cyl[0],0,0,cyl[1])
    bL,bS,cyl = list(zip(3*[3],[bL,bS,cyl]))
    dimer = geo.fuse([bL],[bS,cyl])[0]
    geo.synchronize()
    meshing(model,ls,dimer) if ls else None
    vis() if viz else None
    return dimer

def trimer(k,f,l=1,rS=0.56,cylUL=2*[[1.85,0.20]],ls=0,viz=True):
    """Generates a model of a trimer (L--M--U) comprising a sphere M
    of radius k*rS, centered at (0,0,0), between two other spheres.
    The angle between the three spheres (LMU) is given by f. Sphere U
    has radius rS and sphere L has radius l*rS. Spheres M and U are
    connected via a cylinder MU with length cylUL[0][0] and radius
    cylUL[0][1]. Spheres M and L are connected via a cylinder ML with
    length cylUL[1][0] and radius cylUL[1][1]
 
    Optionally, meshes the model, writes the mesh, and visualizes is.

    Parameters
    ----------
    k : float
        Ratio of sphere M's radius to sphere U's radius.
    f : float
        Angle between the three spheres (L-M-U) in degrees.
    l : float
        Ratio of sphere L's radius to sphere U's radius.
    rS : float, default=0.56
        Radius of sphere U.
    cylUL : list, default=[[1.85,0.20],[1.85,0.20]]
        [[Length, Radius] of cylinder MU,
        [Length, Radius] of cylinder LU].
    ls : float or list, default=0
        If ls is a float it sets the maximum mesh size. If ls is
        a list, containing a float and a string, sets the maximum
        element size to the float and exports a GMSH 2.2 *.msh file
        using string as name.
    viz : bool, default=False
        A flag controlling visualisation.

    Returns
    -------
    int
        Volume tag of the trimer.
    
    """
    model,geo,mesh,_ = new_geometry()
    bM,bU,bL = [geo.addSphere(0,0,0,i*rS) for i in [k,1,l]]
    cylU,cylL = [geo.addCylinder(0,0,0,i[0],0,0,i[1]) for i in cylUL]
    bM,bU,bL,cylU,cylL = list(zip(5*[3],[bM,bU,bL,cylU,cylL]))
    objs = enumerate([bU,bL])
    foo = [geo.translate([j],cylUL[i][0],0,0) for i,j in objs]
    geo.rotate([bU,cylU],0,0,0,0,0,1,f/360*pi)
    geo.rotate([bL,cylL],0,0,0,0,0,1,-f/360*pi)
    trimer = geo.fuse([bM],[bL,bU,cylL,cylU])[0]
    geo.synchronize()
    meshing(model,ls,trimer) if ls else None
    vis() if viz else None
    return trimer
