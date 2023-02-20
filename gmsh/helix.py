"""High-level functions to generate an active helix
starting and ending with spherical caps.

"""

import numpy as np
from numpy.linalg import norm
from math import pi,cos,sin
from scipy.spatial.transform import Rotation
from gmsh_RNG.common import new_geometry,vis,meshing,check_CAD

def backbone(turns,R,r,l):
    """Initializes the model, geometry, and opts classes.
    Additionally, generates points along a helical path, which
    are later used to construct BSpline curves for extrusion.

    Parameters
    ----------
    turns : list
        Determines how many periods are made
        before the activity switches.
    R : float
        Distance from the helix centerline to the center of the tube.
    r : float
        Helical tube radius.
    l : float
        Helical pitch.  

    Returns
    -------
    class,numpy array,list,list,list
        Model class of the helix, 
        coordinates of the guide points,
        tags of the guide points,
        start and end point for each helical chunk
        rotations about the y- and z-axes applied to the guide points.
    
    """
    model,geo,opts = new_geometry()
    res = sum(turns) if isinstance(turns,list) else turns
    opts.setNumber("Geometry.NumSubEdges", int(31*res))
    Ns = [0]+turns if isinstance(turns,list) else [0,turns]
    ts = np.arange(0,sum(Ns)+0.005,0.01)
    xs = R*np.cos(ts*2*pi)
    ys = R*np.sin(ts*2*pi)
    zs = ts*l
    xyz = np.transpose([xs,ys,zs])
    drs = xyz-xyz[0]
    rotZ = np.arccos(drs[1,0]/norm(drs[1,:2]))
    rotZ = Rotation.from_rotvec([0,0,-rotZ])
    rotY = np.arccos(norm(drs[1,:2])/norm(drs[1]))
    rotY = Rotation.from_rotvec([0,rotY,0])
    xyz = rotY.apply(rotZ.apply(drs))+xyz[0]
    ps = [geo.addPoint(*i) for i in xyz]
    ranges = np.array(np.ceil(np.cumsum(Ns)*100),int)
    ranges = list(zip(ranges[:-1],ranges[1:]+1))
    geo.synchronize()
    return model,xyz,ps,ranges,[rotY,rotZ]

def start_cap(model,ps,xyz,r,xi,q):
    """Generates the starting cap of the helix.

    Parameters
    ----------
    model : class
        Necessary for geometrical operations.
    ps : list
        tags of the guide points.
    xyz : numpy array
        Coordinates of the guide points
    r : float
        Helical tube/cap radius.
    xi : float
        Fraction of the helix/cap which is active.
    q : float
        Orientation of the active strip in degrees.        

    Returns
    -------
    list
        Contains lists of dimTags of the:
        1) volume covered by the active surface of the cap,
        2) active surface to be extruded along the next BSpline chunk,
        3) active surface of the cap,
        4) volume covered by the passive surface of the cap,
        5) passive surface to be extruded along the next BSpline chunk,
        6) passive surface of the cap.
    """
    geo = check_CAD(model)
    f = xi*pi if 0<xi<1 else pi/2
    f0 = -f-q/180*pi if 0<xi<1 else -f
    pB = ps[0]
    pT = geo.addPoint(*(xyz[0]+[-r,0,0]))
    pE = geo.addPoint(*(xyz[0]+[0,-cos(f0)*r,-sin(f0)*r]))
    lT,lE = [geo.addLine(i,pB) for i in [pT,pE]]
    cC = geo.addCircleArc(pE,pB,pT)
    sC = geo.addPlaneSurface([geo.addCurveLoop([lT,lE,cC])])
    cA = geo.revolve([(2,sC)],0,0,0,-1,0,0,-2*f)
    cP = geo.revolve(cA[:1],0,0,0,-1,0,0,-2*(pi-f))
    spring = [[j] for j in sum([i[1:] for i in [cA,cP]],[])]
    geo.synchronize()
    return spring

def chunk(model,ps,rng,spring):
    """Generates a chunk of the helix. Assumes the activity
    does not switch across different chunks, i.e. an active
    surface is extruded to produce an active volume. This is
    corrected (if necessary) once all chunks and the end cap
    are generated.

    Parameters
    ----------
    model : class
        Helix model.
    ps : list
        tags of the guide points.
    rng : tuple
        A range of point tags along which the active 
        and passive surfaces should be extruded.
    spring : list
        Contains lists of dimTags of the active and
        passive parts of the helix, as output by either
        start_cap or chunk.

    Returns
    -------
    list
        Contains updated lists of tags of the:
        1) volumes covered by the active surfaces,
        2) active surface to be extruded along the next BSpline chunk,
        3) active surfaces,
        4) volumes covered by the passive surfaces,
        5) passive surface to be extruded along the next BSpline chunk,
        6) passive surfaces.
    
    """
    geo = check_CAD(model)
    neigh = model.getAdjacencies
    bb = geo.addWire([geo.addBSpline(ps[slice(*rng)])])
    pipeA = geo.addPipe([(2,spring[1][-1])],bb,'Frenet')[0][1]
    pipeP = geo.addPipe([(2,spring[4][-1])],bb,'Frenet')[0][1]
    geo.synchronize()
    pipeA = np.hstack(([pipeA],neigh(3,pipeA)[1][[-1,1]]))
    pipeP = np.hstack(([pipeP],neigh(3,pipeP)[1][[-1,1]]))
    pipe = np.hstack([pipeA,pipeP])
    foo = [spring[i].append(j) for i,j in enumerate(pipe)]
    return spring

def end_cap(model,geo,r,xyz,xi,spring):
    """Generates the ending cap of the helix which adopts the
    activity of the preceding chunk of the helix.
    Parameters
    ----------
    model : type
        Helix model.
    geo : type
        Necessary for geometrical operations.
    r : float
        Helical tube/cap radius.
    xyz : numpy array
        Coordinates of the guide points.
    xi : float
        Fraction of the helix/cap which is active.
    spring : list
        Contains lists of GMSH tags of the active and
        passive parts of the helix, as output by chunk.

    Returns
    -------
    list
        Contains updated lists of GMSH tags of the:
        1) volumes covered by the active surfaces,
        2) active surfaces to be extruded along the guide points,
        3) active surfaces,
        4) volumes covered by the passive surfaces,
        5) passive surfaces to be extruded along the guide points,
        6) passive surfaces.
    
    """
    geo = check_CAD(model)
    bounds = model.getBoundary
    f = xi*pi if 0<xi<1 else pi/2
    lbs = [i[1] for i in bounds([(2,spring[1][-1])])]
    lss = [i[1] for i in bounds([(2,spring[2][-1])])]
    curve = np.intersect1d(np.abs(lbs),np.abs(lss))[0]
    curve = np.where(np.abs(lbs) == curve)[0][0]
    lE = lbs[(np.sign(lbs[curve])+curve)%3]
    nv = model.getNormal(spring[1][-1],(0.5,0.5))
    pT = geo.addPoint(*(xyz[-1]+r*nv))
    points = np.array(bounds([(1,lE)]))
    drs = [model.getValue(*i,[])-xyz[-1] for i in points]
    pE = points[norm(drs,axis=1)>r/2][0,1]
    pB = points[points[:,1] != pE][0,1]
    lT = geo.addLine(pB,pT)
    cC = geo.addCircleArc(pT,pB,pE)
    sC = geo.addPlaneSurface([geo.addCurveLoop([lT,lE,cC])])
    cA = geo.revolve([(2,sC)],*xyz[-1],*nv,-2*f)
    cP = geo.revolve(cA[:1],*xyz[-1],*nv,-2*(pi-f))
    geo.synchronize()
    end = np.hstack([np.array(i).T[1,[1,3,2]] for i in [cA,cP]])
    end = list(map(list,end.reshape(-1,1)))
    spring = [spring[i]+j for i,j in enumerate(end)]
    return spring


def assemble_helix(model,geo,spring,rots):
    """Assembles all chunks and caps and stitches all faces together.
    Additionally, rotates the helix to align its centerline with the
    z-axis.
    
    Parameters
    ----------
    model : type
        Helix model.
    geo : type
        Necessary for geometrical operations.
    spring : list
        Contains lists of GMSH tags of the active and
        passive parts of the helix, as output by end_cap.
    rots : list
        Rotations about the y- and z-axes applied
        to the guide points.

    Returns
    -------
    list,list
        Active and passive surfaces of the resulting volumes after
        the general fuse operation and rotation.
    
    """
    geo = check_CAD(model)
    nv = model.getNormal
    vA,_,sA0,vP,_,sP0 = spring
    sA_nv,sP_nv = [[nv(i,[0.5,0.5]) for i in j] for j in [sA0,sP0]]
    vols = geo.fragment([(3,vA[0])],[(3,i) for i in vA[1:]+vP])[0]
    geo.synchronize()
    surfs = np.array(model.getBoundary(vols,oriented=False))
    all_nv = [nv(i[1],[0.5,0.5]) for i in surfs]
    sA = [surfs[np.where(np.all(i == all_nv,1))][0,1] for i in sA_nv]
    sP = [surfs[np.where(np.all(i == all_nv,1))][0,1] for i in sP_nv]
    p0 = model.getValue(0,1,[0,0])
    surfN = geo.get_max_tag(2)
    rotY = [*p0,0,1,0,-rots[0].as_rotvec()[1]]
    rotZ = [*p0,0,0,1,-rots[1].as_rotvec()[2]]
    geo.rotate([(3,i) for i in spring[0]+spring[3]],*rotY)
    geo.rotate([(3,i) for i in spring[0]+spring[3]],*rotZ)
    sA,sP = [[i+2*surfN for i in j] for j in [sA,sP]]
    geo.synchronize()
    return sA,sP

def grouping(model,sA,sP,xi):
    """Re-maps the active and passive surfaces of
    the helix depending on the coverage.

    Parameters
    ----------
    model : type
        Helix model.
    sA : list
        List of active surfaces, as output by assemble_helix.
    sP : float
        List of passive surfaces, as output by assemble_helix.
    xi : float
        Surface coverage.        

    Returns
    -------
    None
    
    """
    n = len(sA)
    same = [0]+list(range(1,n-2+n%2,2))
    same = same+[n-1] if n%2 else same
    other = list(set(range(n))-set(same))
    sA,sP = [np.array(i) for i in [sA,sP]]
    if 0<xi<1:
        sAnew = np.hstack((sA[same],sP[other]))
        sPnew = np.hstack((sP[same],sA[other]))
    else:
        new = np.vstack((sA,sP))
        sAnew = new[:,same].ravel() if xi else new[:,other].ravel()
        sPnew = new[:,other].ravel() if xi else new[:,same].ravel()
    gs = enumerate([sAnew,sPnew])
    ns = ['Active','Passive']
    gs = [model.addPhysicalGroup(2,j,-1,ns[i]) for i,j in gs]
    return

def helix(turns,act,R,r,l=1,q=0,ls=0,viz=1):
    """Generates a model of a helix of radius R and pitch l having
    an active strip. The helix has a tube radius of r and comprises
    sum(turns) windings. It begins and ends with spherical caps. The
    starting cap is centered at (R,0,0). The size of the strip is
    controlled via act, which takes values from 0 to 1, where 0
    denotes an entirely passive starting cap and 1 corresponds to
    an entirely active starting cap. Fractional activities make the
    cap azimuth [-pi*act,pi/act] active, where 0 is parallel to the
    x-axis. The orientation of the strip can be controlled via q.
    If more than one value is passed to turns the activity switches
    after turns[0] periods. The first turns[0] periods the helix adopts
    the same activity as the starting cap. The end cap adopts the
    activity of the last portion of the helix.
 
    Optionally, meshes the model, writes the mesh, and visualizes is.

    Parameters
    ----------
    turns : list
        Determines how many periods are made before the activity switches. 
    act : float
        Fraction of the helix which is active.
    R : float
        Distance from the helix centerline to the center of the tube.
    r : float
        Helical tube radius.
    l : float, default=1
        Helical pitch.
    q : float, default=0
        Orientation of the active strip.
    ls : float or list, default=0
        If ls is a float it sets the maximum mesh size. If ls is a list,
        containing a float and a string, sets the maximum element size
        to the float and exports a file named after the string.
    viz : bool, default=False
        A flag controlling visualisation.

    Returns
    -------
    None
    
    """
    model,xyz,ps,ranges,rots = backbone(turns,R,r,l)
    spring = start_cap(model,ps,xyz,r,act,q)
    for rng in ranges:
        spring = chunk(model,ps,rng,spring)
    spring = end_cap(model,r,xyz,act,spring)
    sA,sP = assemble_helix(model,spring,rots)
    grouping(model,sA,sP,act)
    meshing(model,ls) if ls else None
    vis(0) if viz else None
    return helix