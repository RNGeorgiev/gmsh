"""High-level functions to generate an active helix
starting and ending with spherical caps.

"""

import numpy as np
from numpy.linalg import norm
from math import pi,cos,sin,atan
from scipy.spatial.transform import Rotation
from ..common import new_geometry,vis,meshing,check_CAD,eval_on_surf

def align_vectors(v1,v2):
    """Computes the rotation angles about the x-, y-, and z- axis
    needed to align v1 to v2. An identity matrix of size 3 is 
    concatinated horizontally with the resulting column vector.

    Parameters
    ----------
    v1 : list or numpy array
        Vector to be rotated.
    v2 : list or numpy array
        Vector, with which v1 is to align.

    Returns
    -------
    numpy array
        Resulting 4-by-3 matrix.
    
    """
    v1,v2 = [np.array(i) for i in [v1,v2]]
    v1,v2 = [i/norm(i) for i in [v1,v2]]
    v,c = [i(v1,v2) for i in [np.cross,np.dot]]
    k = -np.cross(np.eye(3),v)
    rotM = np.eye(3)+k+k@k/(1+c)
    rotM = Rotation.from_matrix(rotM)
    rotM = rotM.as_euler('XYZ')
    rotM = np.vstack((np.eye(3),rotM)).T
    return rotM

def backbone(turns,R,l):
    """Initializes the model, geometry, and opts classes.
    Additionally, generates points along a helical path, which
    are later used to construct Spline curves for extrusion.

    Parameters
    ----------
    turns : list or int
        Determines how many periods are made
        before the activity switches.
    R : float
        Distance from the helix centerline to the center of the tube.
    l : float
        Helical pitch.  

    Returns
    -------
    class,numpy array,list,list
        Model class of the helix, 
        coordinates of the guide points,
        tags of the guide points,
        start and end point for each helical chunk.
    
    """
    model,geo,opts = new_geometry()
    res = sum(turns) if isinstance(turns,list) else turns
    opts.setNumber("Geometry.NumSubEdges", int(200*res))
    Ns = [0]+turns if isinstance(turns,list) else [0,turns]
    ts = np.arange(0,sum(Ns)+5e-3,1e-2)
    xs = R*np.cos(ts*2*pi)
    ys = R*np.sin(ts*2*pi)
    zs = ts*l
    xyz = np.transpose([xs,ys,zs])
    ps = [geo.addPoint(*i) for i in xyz]
    ranges = np.array(np.ceil(np.cumsum(Ns)*1e2),int)
    ranges = list(zip(ranges[:-1],ranges[1:]+1))
    geo.synchronize()
    return model,xyz,ps,ranges

def start_cap(model,xyz,r,xi,q):
    """Generates the starting cap of the helix.

    Parameters
    ----------
    model : class
        Necessary for geometrical operations.
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
        1) active surface to be extruded along the next Spline chunk,
        2) active surface of the cap,
        4) passive surface to be extruded along the next Spline chunk,
        5) passive surface of the cap.
    """
    spring = []
    geo = check_CAD(model)
    bounds = model.getBoundary
    f = 2*xi*pi if 0 < xi < 1 else pi
    f0 = atan(1e2*xyz[1,2]/2/pi/xyz[0,0])
    pars = [[*xyz[0],r,-1,0,pi/2,i] for i in [f,2*pi-f]]
    capA,capP = [[(3,geo.addSphere(*i))] for i in pars]
    geo.rotate(capA,*xyz[0],0,0,1,q-f/2)
    geo.rotate(capP,*xyz[0],0,0,1,q+f/2)
    geo.rotate(capA+capP,*xyz[0],1,0,0,pi/2+f0)
    geo.synchronize()
    for cap in [capA,capP]:
        surfs = list(map(tuple,np.abs(bounds(cap))))
        nvs = eval_on_surf(model,surfs,'nv')[:,2]
        nvs = np.isclose(nvs,sin(f0))
        curves = eval_on_surf(model,surfs,'1/r') > 0
        spring.append([i for i,j in zip(surfs,curves) if j])
        spring.append([i for i,j in zip(surfs,nvs) if j])
    return spring

def chunk(model,ps,r,rng,spring):
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
        Tags of the guide points.
    r : float
        Helical tube radius.
    rng : tuple
        A range of point tags along which the active 
        and passive surfaces should be extruded.
    spring : list
        Contains lists of dimTags of the active and passive
        parts of the helix, as output by either start_cap or chunk.

    Returns
    -------
    list
        Contains updated lists of dimTags of the:
        1) active surface to be extruded along the next Spline chunk,
        2) active surfaces,
        3) passive surface to be extruded along the next Spline chunk,
        4) passive surfaces.
    
    """
    pipes = []
    geo = check_CAD(model)
    bounds = model.getBoundary
    bb = geo.addWire([geo.addSpline(ps[slice(*rng)])])
    for k in spring[1::2]:
        pipe = geo.addPipe(k[-1:],bb,'Frenet')
        geo.synchronize()
        surfs = bounds(pipe)
        curves = eval_on_surf(model,surfs,'1/r')
        curves = np.argmin(np.abs(curves-1/r))
        pipes = pipes+[surfs[i] for i in [curves,-1]]
    pipe = enumerate(map(tuple,np.abs(pipes)))
    [spring[i].append(j) for i,j in pipe]
    return spring

def end_cap(model,turns,xyz,r,xi,q,spring):
    """Generates the ending cap of the helix which adopts the
    activity of the preceding chunk of the helix.
    Parameters
    ----------
    model : class
        Helix model.
    turns : list or int
        Determines how many periods are made
        before the activity switches.
    xyz : numpy array
        Coordinates of the guide points.
   r : float
        Helical tube/cap radius.
    xi : float
        Fraction of the helix/cap which is active.
    q : float
        Orientation of the active strip in degrees.     
    spring : list
        Contains lists of dimTags of the active and passive
        parts of the helix, as output by either chunk.

    Returns
    -------
    list
        Contains updated lists of dimTags of the:
        1) active surfaces,
        2) passive surfaces.
    
    """
    geo = check_CAD(model)
    bounds = model.getBoundary
    f = 2*xi*pi if 0 < xi < 1 else pi
    f0 = sum(turns)%1*2*pi-q-f/2
    nv = eval_on_surf(model,spring[-1][-1],'nv')
    rots = align_vectors(nv,[0,0,1])[::-1]
    pars = [[*xyz[-1],r,-1,0,pi/2,i] for i in [f,2*pi-f]]
    capA,capP = [[(3,geo.addSphere(*i))] for i in pars]
    [geo.rotate(capA+capP,*xyz[-1],*i) for i in rots]
    geo.rotate(capA,*xyz[-1],*nv,f0)
    geo.rotate(capP,*xyz[-1],*nv,f0+f)
    geo.synchronize()
    for cap in [capA,capP]:
        offset = 0 if cap == capA else 2
        surfs = list(map(tuple,np.abs(bounds(cap))))
        curves = eval_on_surf(model,surfs,'1/r') > 0
        curves = [i for i,j in zip(surfs,curves) if j]
        spring[offset] = spring[offset]+curves
    spring = [spring[i] for i in [0,2]]
    return spring


def assemble_helix(model,spring,xi):
    """Assembles all chunks and caps and fuses all faces together.
    Additionally, re-maps the active and passive surfaces of the
    helix depending on the coverage.
    
    Parameters
    ----------
    model : class
        Helix model.
    spring : list
        Contains lists of dimTags of the active and passive
        parts of the helix, as output by end_cap.
    xi : float
        Surface coverage.

    Returns
    -------
    list
        Contains lists of dimTags of the active and passive
        surfaces, comprising the helix.
    
    """
    geo = check_CAD(model)
    sA,sP = [geo.copy(i) for i in spring]
    sA,sP = [geo.fuse(i[:1],i[1:])[0] for i in [sA,sP]]
    geo.remove(model.get_entities(3),True)
    helix = geo.fragment(sA,sP)[0]
    geo.synchronize()
    n = len(sA)
    sA,sP = np.array([sA,sP])
    same = [0]+list(range(1,n-2+n%2,2))
    same = same+[n-1] if n%2 else same
    other = list(set(range(n))-set(same))
    sA,sP = [np.array(i) for i in [sA,sP]]
    if 0<xi<1:
        sAnew = np.vstack((sA[same],sP[other]))[:,1]
        sPnew = np.vstack((sP[same],sA[other]))[:,1]
    else:
        new = np.hstack((sA,sP))[:,1::2]
        sAnew = (new[same] if xi else new[other]).ravel()
        sPnew = (new[other] if xi else new[same]).ravel()
    gs = enumerate([sAnew,sPnew])
    ns = ['Active','Passive']
    gs = [model.addPhysicalGroup(2,j,-1,ns[i]) for i,j in gs]
    return helix

def helix(turns,xi,R,r,l=1,q=0,ls=0,viz=True):
    """Generates a model of a helix of radius R and pitch l having
    an active strip. The helix has a tube radius of r and comprises
    sum(turns) windings. It begins and ends with spherical caps. The
    starting cap is centered at (R,0,0). The size of the strip is
    controlled via xi, which takes values from 0 to 1, where 0
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
        Number of many periods made before the activity switches.
    xi : float
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
        If ls is a float it sets the maximum mesh size. If ls is
        a list, containing a float and a string, sets the maximum
        element size to the float and exports a GMSH 2.2 *.msh file
        using string as name.
    viz : bool, default=False
        A flag controlling visualisation.

    Returns
    -------
    None
    
    """
    msg = 'Warning! Tube radius larger than '
    msg = msg+'maximum mesh size.'
    print(msg) if (ls and r<ls) else None
    q = np.radians(q)
    model,xyz,ps,ranges = backbone(turns,R,l)
    spring = start_cap(model,xyz,r,xi,q)
    for rng in ranges:
        spring = chunk(model,ps,r,rng,spring)
    spring = end_cap(model,turns,xyz,r,xi,q,spring)
    helix = assemble_helix(model,spring,xi)
    meshing(model,ls) if ls else None
    vis(0) if viz else None
    return helix