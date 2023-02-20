"""High-level functions to generate quasi-2D dimers and trimers
comprising disks and rods with rounded edges.

"""

from math import sin,pi,tan
import numpy as np
from gmsh_RNG.common import is_part_of,meshing,vis,new_geometry
from gmsh_RNG.common import eval_on_surf,check_CAD

def get_walls(model,vol):
    """Computes the normal of each surface and places each in one of
    five groups, depending on the z-component NVz of said normal:
        1) top walls (NVz = 1)
        2) bottom walls (NVz = -1)
        3) side walls (NVz = 0)
        4) top edges (1 > NVz > 0)
        5) bottom edges (-1 < NVz < 0)
        
    Parameters
    ----------
    model : class
        Model containing the body.
    vol : int, tuple, or dimTag
        Volume tag of the body, whose surfaces are to be clasified.

    Returns
    -------
    list
        [list of top walls, list of bottom walls,
         list of side walls,
         list of top edges, list of bottom edges]
    
    """
    vol = [(3,vol)] if isinstance(vol,int) else vol
    vol = [vol] if isinstance(vol,tuple) else vol
    surfs = np.abs(model.getBoundary(vol))
    nvs = eval_on_surf(model,surfs,'nv')
    tops,bots,sides = [surfs[nvs[:,2] == i] for i in [1,-1,0]]
    edges = np.vstack((tops,bots,sides))
    edges = np.setdiff1d(surfs[:,1],edges[:,1])
    if len(edges):
        edges = np.array([(2,i) for i in edges])
        nvs = eval_on_surf(model,edges,'nv')
        top_edges = edges[nvs[:,2]>0]
        bot_edges = edges[nvs[:,2]<0]
    else:
        top_edges,bot_edges = [(2,0)],[(2,0)]
    surfs = [tops,bots,sides,top_edges,bot_edges]
    surfs = [list(map(tuple,i)) for i in surfs]
    return surfs

def grouping(model,vol):
    """Applies a physical group to each surface depending on
    its orientation. For details see get_walls().
        
    Parameters
    ----------
    model : class
        Model containing the body.
    vol : int
        Volume tag of the body, whose surfaces are clasified .

    Returns
    -------
    None
    
    """
    walls = get_walls(model,vol)
    gs = [model.addPhysicalGroup(2,np.array(i)[:,1]) for i in walls]
    ns = ['Top base','Bottom base','Sides']
    ns = enumerate(ns+['Top edge','Bottom edge'])
    foo = [model.setPhysicalName(2,gs[i],j) for i,j in ns]
    return

def fillet_Z(model,vol,rZ):
    """Fillets all edges prallel to the xy-plane.
    Parameters
    ----------
    model : type
        Model containing the body.
    vol : int
        Volume tag of the body, whose edges are filleted.
    rZ : float
        Rounding radius for edges parallel to the xy-plane.

    Returns
    -------
    int
        Volume tag of the filleted body.
    
    """
    geo = check_CAD(model)
    bases = sum(get_walls(model,vol)[:2],[])
    zEdges = np.abs(model.getBoundary(bases,False))[:,1]
    filleted = geo.fillet([vol],zEdges,[rZ])[0][1]
    geo.synchronize()
    return filleted

def fillet_XY(model,vol,rXY):
    """Fillets all edges normal to the xy-plane.
    Parameters
    ----------
    model : class
        Model containing the body.
    vol : int
        Volume tag of the body, whose edges are filleted.
    rXY : float
        Rounding radius for all edges normal to the xy-plane.

    Returns
    -------
    int
        Volume tag of the filleted body.
    
    """
    geo = check_CAD(model)
    bounds = model.getBoundary
    sides = get_walls(model,vol)[2]
    xyEdges = np.abs(bounds(sides,False))[:,1]
    xyEdges = np.setdiff1d(xyEdges,np.abs(bounds(sides))[:,1])
    filleted = geo.fillet([vol],xyEdges,[rXY])[0][1]
    geo.synchronize()
    return filleted

def fillets(model,vol,rZ,rXY):
    """Shorthand to fillet edges of vol in model.
    Parameters
    ----------
    model : type
        Model containing the body.
    vol : int
        Voume tag of the body.
    rZ : float
        Rounding radius for all edges on surfaces normal to the z-axis.
    rXY : float
        Rounding radius for all edges normal to the xy-plane.

    Returns
    -------
    int
        Volume tag of the filleted body.
    
    """
    geo = check_CAD(model)
    geo.synchronize()
    vol = fillet_XY(model,vol,rXY) if rXY else vol
    vol = fillet_Z(model,vol,rZ) if rZ else vol
    return vol

def disk(r,Hp,rZ=0,partOf=None,
         ls=0,gs=False,viz=True):
    """Generates a model of a cylinder of radius r and thickness
    Hp, centered at (0,0,0). The edges of the cylinder can be
    rounded off with a fillet radius of rZ. The disk can be part
    of a complex body by supplying model as partOf.
    
    Can also be used to create a pill-like particle when rZ=r and
    Hp>2r.
    
    Optionally, meshes the disk, writes the mesh, and visualizes is.

    Parameters
    ----------
    r : float
        Cylinder radius.
    Hp : float
        Particle thickness.
    rZ : float, default=0
        Rounding radius for all edges on surfaces normal to the z-axis.
    partOf: class, default=None
        Determines if the cylinder is part of a complex shape.
        For details, see common.is_part_of()
    ls : float or list, default=0
        If ls is a float it sets the maximum mesh size. If ls is
        a list, containing a float and a string, sets the maximum
        element size to the float and exports a GMSH 2.2 *.msh file
        using string as name.
    gs : bool, default=False
        A flag, which assigns physical groups to the:
            1) bases (surfaces normal to the z-axis)
            2) sides (surfaces normal to the xy-plane)
            3) edges (surfaces which are neither bases, nor sides)
    viz : bool, default=False
        A flag controlling visualisation.

    Returns
    -------
    int
        Volume tag of the cylinder.
    
    """
    model,geo = is_part_of(partOf)
    cyl = geo.addCylinder(0,0,-Hp/2,0,0,Hp,r)
    cyl = fillets(model,cyl,rZ,0)
    grouping(model,cyl) if gs else None
    meshing(model,ls,cyl) if ls else None
    vis() if viz else None
    return cyl

def ring(R,r,Hp,rZ=0.1,partOf=None,
         ls=0,gs=False,viz=True):
    """Generates a model of a ring centered at (0,0,0) with an outer
    radius R, inner radius r and thickness Hp. The edges of the ring
    can be rounded off with a fillet radius of rZ. The ring becomes a
    torus if the rounding radius is too large. The ring can be a part
    of a complex body by supplying model as partOf.
    
    Optionally, meshes the ring, writes the mesh, and visualizes is.

    Parameters
    ----------
    R : float
        Outer radius.
    r : float
        Inner radius.
    Hp : float
        Particle thickness.
    rZ : float, default=0
        Rounding radius for all edges on surfaces normal to the z-axis.
    partOf: class, default=None
        Determines if the ring is part of a complex shape.
        For details, see common.is_part_of()
    ls : float or list, default=0
        If ls is a float it sets the maximum mesh size. If ls is
        a list, containing a float and a string, sets the maximum
        element size to the float and exports a GMSH 2.2 *.msh file
        using string as name.
    gs : bool, default=False
        A flag, which assigns physical groups to the:
            1) bases (surfaces normal to the z-axis)
            2) sides (surfaces normal to the xy-plane)
            3) edges (surfaces which are neither bases, nor sides)
    viz : bool, default=False
        A flag controlling visualisation.

    Returns
    -------
    int
        Volume tag of the ring.
    
    """
    model,geo = is_part_of(partOf)
    if rZ>=((R-r)/2) or rZ>=Hp/2:
        ring = geo.addTorus(0,0,0,(R+r)/2,(R-r)/2)
    else:
        cO = geo.addCylinder(0,0,-Hp/2,0,0,Hp,R)
        cI = geo.addCylinder(0,0,-Hp/2,0,0,Hp,r)
        ring = geo.cut([(3,cO)],[(3,cI)])[0][0][1]
        ring = fillets(model,ring,rZ,0)
        grouping(model,ring) if gs else None
    geo.synchronize()
    meshing(model,ls,ring) if ls else None
    vis() if viz else None
    return ring

def rod(l,w,Hp,rZ=0,rXY=0,partOf=None,
        ls=0,gs=False,viz=True):
    """Generates a model of a rod of lenght l (along thex-axis)
    and width w (along the y-axis). The rod is centered at (0,0,0)
    and its thickness is Hp. All edges of the rod can be rounded off.
    Edges normal to the xy-plane are filleted with a radius rXY and
    edges parallel to it are filleted with a radius rZ. The rod can
    be a part of a complex body by supplying model as partOf.
    
    Optionally, meshes the rod, writes the mesh, and visualizes is.

    Parameters
    ----------
    l : float
        Rod length.
    w : float
        Rod width.
    Hp : float
        Particle thickness.
    rZ : float, default=0
        Rounding radius for all edges normal to the the z-axis.
    rXY : float, default=0
        Rounding radius for all edges normal to the xy-plane.
    partOf: class, default=None
        Determines if the rod is part of a complex shape.
        For details, see common.is_part_of()
    ls : float or list, default=0
        If ls is a float it sets the maximum mesh size. If ls is
        a list, containing a float and a string, sets the maximum
        element size to the float and exports a GMSH 2.2 *.msh file
        using string as name.
    gs : bool, default=False
        A flag, which assigns physical groups to the:
            1) bases (surfaces normal to the z-axis)
            2) sides (surfaces normal to the xy-plane)
            3) edges (surfaces which are neither bases, nor sides)
    viz : bool, default=False
        A flag controlling visualisation.

    Returns
    -------
    int
        Volume tag of the rod.
    
    """
    model,geo = is_part_of(partOf)
    box = geo.addBox(-l/2,-w/2,-Hp/2,l,w,Hp)
    box = fillets(model,box,rZ,rXY)
    grouping(model,box) if gs else None
    meshing(model,ls,box) if ls else None
    vis() if viz else None
    return box

def prism(R,a,Hp,rZ=0,rXY=0,partOf=None,
          ls=0,gs=False,viz=True):
    """Generates a model of a prism centered at (0,0,0) and inscribed
    in a circle or radius R. The prism base is an isosceles triangle
    with angle between the equal sides a. All edges of the prism can
    be rounded off. Edges normal to the xy-plane are filleted with
    a radius rXY and edges parallel to it are filleted with a radius
    rZ. The in-plane rounding does not change the radius of the
    escribing circle. The height of the prism is Hp. The prism can be
    a part of a complex body by supplying model as partOf.
    
    Optionally, meshes the prism, writes the mesh, and visualizes is.

    Parameters
    ----------
    R : float
        Radius of the escribing circle.
    a : float
        Angle between the two equal sides in degrees.
    Hp : float
        Particle thickness.
    rZ : float, default=0
        Rounding radius for all edges normal to the z-axis.
    rXY : float, default=0
        Rounding radius for all edges normal to the xy-plane.
    partOf: class, default=None
        Determines if the prism is part of a complex shape.
        For details, see common.is_part_of().
    ls : float or list, default=0
        If ls is a float it sets the maximum mesh size. If ls is
        a list, containing a float and a string, sets the maximum
        element size to the float and exports a GMSH 2.2 *.msh file
        using string as name.
    gs : bool, default=False
        A flag, which assigns physical groups to the:
            1) bases (surfaces normal to the z-axis)
            2) sides (surfaces normal to the xy-plane)
            3) edges (surfaces which are neither bases, nor sides)
    viz : bool, default=False
        A flag controlling visualisation.

    Returns
    -------
    int
        Volume tag of the prism.
    
    """
    model,geo = is_part_of(partOf)
    p1X = R+rXY*(1/sin(a/360*pi)-1)
    dx = (R-rXY)*sin(pi*(1/2-a/180))+rXY
    dy = tan(a/360*pi)*(p1X+dx)
    p1 = geo.addPoint(p1X,0,-Hp/2)
    p2 = geo.addPoint(-dx,dy,-Hp/2)
    p3 = geo.addPoint(-dx,-dy,-Hp/2)
    ps = [p1,p3,p2,p1]
    lines = [geo.addLine(ps[i],ps[i+1]) for i in range(3)]
    bottom = geo.addPlaneSurface([geo.addCurveLoop(lines)])
    prism = geo.extrude([(2,bottom)],0,0,Hp)[1][1]
    prism = fillets(model,prism,rZ,rXY)
    grouping(model,prism) if gs else None
    meshing(model,ls,prism) if ls else None
    vis(0) if viz else None
    return prism

def dimer(k,Hp,rZ=0.1,rS=0.56,rd=[1.85,0.41],
          ls=0,gs=False,viz=True):
    """Generates a q2D dimer comprising a large cylinder of radius
    k*rS and a small cylinder of radius rS connected via a rod of
    length rd[0] and width rd[1]. All three bodies have a thickness
    Hp. The large cylinder is centered at (0,0,0). The long axis of
    the rod is parallel to the x-axis. The edges of the cylinders
    and the rod can be rounded off with a fillet radius of rZ.
    
    Optionally, meshes the dimer, writes the mesh, and visualizes is.

    Parameters
    ----------
    k : float
        Ratio of the two cylinder radii.
    Hp : float
        Particle thickness.
    rZ : float, default=0.1
        Rounding radius for all edges normal to the z-axis.
    rS : float, default=0.56
        Radius of the smaller cylinder.
    rd: list, default=[1.85,0.41]
        [Length, Width] of the connecting rod.
    ls : float or list, default=0
        If ls is a float it sets the maximum mesh size. If ls is
        a list, containing a float and a string, sets the maximum
        element size to the float and exports a GMSH 2.2 *.msh file
        using string as name.
    gs : bool, default=False
        A flag, which assigns physical groups to the:
            1) bases (surfaces normal to the z-axis)
            2) sides (surfaces normal to the xy-plane)
            3) edges (surfaces which are neither bases, nor sides)
    viz : bool, default=False
        A flag controlling visualisation.

    Returns
    -------
    int
        Volume tag of the dimer.
    
    """
    model,geo,_ = new_geometry()
    dXYZs = [[rd[0]*i,0,0] for i in [1,1/2]]
    dL,dS = [disk(i*rS,Hp,rZ,model,viz=False) for i in [k,1]]
    rd = rod(*rd,Hp,rZ,0,model,viz=False)
    objs = enumerate(zip(2*[3],[dS,rd]))
    foo = [geo.translate([j],*dXYZs[i]) for i,j in objs]
    objs = list(zip(2*[3],[dS,rd]))
    dimer = geo.fuse([(3,dL)],objs)[0][0][1]
    geo.synchronize()
    grouping(model,dimer) if gs else None
    meshing(model,ls,dimer) if ls else None
    vis() if viz else None
    return dimer

def trimer(k,f,Hp,l=1,rZ=0.1,rS=0.56,rodUL=2*[[1.85,0.41]],
           ls=0,gs=False,viz=True):
    """Generates a q2D trimer (L--M--U) comprising a cylinder
    M of radius k*rS, centered at (0,0,0), between two other cylinders.
    The angle between the three cylinders (LMU) is given by f. Cylinder
    U has radius rS and cylinder L has radius l*rS. Cylinders M and U
    are connected via a rod MU with length rodUL[0][0] and width
    rod UL[0][1]. Cylinders M and L are connected via a rod ML with
    length rodUL[1][0] and radius rodUL[1][1]. All five bodies have a
    thickness Hp. The edges of the cylinders and the rods can be rounded
    off with a fillet radius of rZ.
 
    Optionally, meshes the model, writes the mesh, and visualizes is.

    Parameters
    ----------
    k : float
        Ratio of cylinder M's radius to cylinder U's radius.
    f : float
        Angle between the three cylinders (L-M-U) in degrees.
    Hp : float
        Particle thickness.
    l : float
        Ratio of cylinder L's radius to cylinder U's radius.
    rZ : float, default=0.1
        Rounding radius for all edges normal to the z-axis.
    rS : float, default=0.56
        Radius of cylinder U.
    rodUL : list, default=[[1.85,0.41],[1.85,0.41]]
        [[Length, Width] of rod MU, [Length, Width] of cylinder ML].
    ls : float or list, default=0
        If ls is a float it sets the maximum mesh size. If ls is
        a list, containing a float and a string, sets the maximum
        element size to the float and exports a GMSH 2.2 *.msh file
        using string as name.
    gs : bool, default=False
        A flag, which assigns physical groups to the:
            1) bases (surfaces normal to the z-axis)
            2) sides (surfaces normal to the xy-plane)
            3) edges (surfaces which are neither bases, nor sides)
    viz : bool, default=False
        A flag controlling visualisation.

    Returns
    -------
    int
        Volume tag of the trimer.
    
    """
    model,geo,_ = new_geometry()
    dXYZs = sum([[i[0],i[0]/2] for i in rodUL],[])
    dXYZs = np.transpose([dXYZs]+2*[4*[0]])
    dM,dU,dL = [disk(i*rS,Hp,rZ,model,viz=False) for i in [k,1,l]]
    rU,rL = [rod(*i,Hp,rZ,0,model,viz=False) for i in rodUL]
    objs = enumerate(zip(4*[3],[dU,rU,dL,rL]))
    foo = [geo.translate([j],*dXYZs[i]) for i,j in objs]
    objs = list(zip(4*[3],[dU,rU,dL,rL]))
    geo.rotate(objs[:2],0,0,0,0,0,1,f/360*pi)
    geo.rotate(objs[2:],0,0,0,0,0,1,-f/360*pi)
    trimer = geo.fuse([(3,dM)],objs)[0][0][1]
    geo.synchronize()
    grouping(model,trimer) if gs else None
    meshing(model,ls,trimer) if ls else None
    vis(0) if viz else None
    return trimer
