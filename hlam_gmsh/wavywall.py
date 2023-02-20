"""High-level functions to generate an corrugated wall,
whose flat parts have a normal (1,0,0).

"""

import numpy as np
from gmsh_RNG.common import new_geometry,vis,meshing

def flat(w,h,ls,viz=True):
    """Generates a model of a flat wall of width w (along the y-axis)
    and height h (along the z-axis) centered at (0,0,0).
    
    Optionally, meshes the model, writes the mesh, and visualizes is.

    Parameters
    ----------
    waveform : str
        One of four available shapes:
        sine, sawtooth, triangle, and square.
    w : float
        Plate width.
    h : float
        Plate height.
    ls : float or list, default=0
        If ls is a float it sets the maximum mesh size in the flat region.
        If ls is a list, containing a float and a string, sets the maximum
        element size in the flat region to the float and exports a file
        named after the string.
    viz : bool, default=False
        A flag controlling visualisation.

    Returns
    -------
    None
    
    """
    model,geo,mesh = new_geometry()
    xyzs = [[-w/2,-h/2],[w/2,-h/2],[w/2,h/2],[-w/2,h/2]]
    order = [0,1,2,3,0]
    ps = np.array([geo.addPoint(0,*i) for i in xyzs])
    lines = [geo.addLine(*ps[order[i:i+2]]) for i in range(4)]
    geo.addPlaneSurface([geo.addCurveLoop(lines)])
    geo.synchronize()
    meshing(mesh,ls) if ls else None
    vis(0) if viz else None
    return

def triangle(A,N,l):
    """Generates a triangular wave profile.

    Parameters
    ----------
    A : float
        Wave amplitude.
    N : int
        Periods.
    l : float
        Wavelength.
    
    Returns
    -------
    list,numpy array
        Point coordinates defining the wave.
    
    """
    xs = l*np.arange(-N/2,N/2+0.5*0.25,0.25)
    ys = [0]+N*[A,0,-A,0]
    return xs,ys

def sawtooth(A,N,l):
    """Generates a sawtooth wave profile.

    Parameters
    ----------
    A : float
        Wave amplitude.
    N : int
        Periods.
    l : float
        Wavelength.
    
    Returns
    -------
    list,numpy array
        Point coordinates defining the wave.
    
    """
    xs1 = np.arange(-N/2,N/2+0.5)
    xs2 = 2*[xs1[i:i+2].mean() for i in range(len(xs1)-1)]
    xs = l*np.sort(np.hstack((xs1,xs2)))
    ys = (len(xs1)-1)*[0,A/2,-A/2]+[0]
    return xs,ys

def square(A,N,l):
    """Generates a square wave profile.

    Parameters
    ----------
    A : float
        Wave amplitude.
    N : int
        Periods.
    l : float
        Wavelength.
    
    Returns
    -------
    list,numpy array
        Point coordinates defining the wave.
    
    """
    xs = np.arange(-N/2,N/2+0.25,0.5)
    xs = l*np.sort(np.hstack((xs[0],xs[-1],xs,xs[1:-1])))
    ys = [0]+N*[A,A,-A,-A]+[0]
    return xs,ys

def sine(A,N,l):
    """Generates a sine wave profile.

    Parameters
    ----------
    A : float
        Wave amplitude.
    N : int
        Periods.
    l : float
        Wavelength.
    
    Returns
    -------
    list,numpy array
        Point coordinates defining the wave.
    
    """
    xs = np.linspace(-N/2,N/2,N*100+1)*l
    ys = A*np.sin(xs/l*2*np.pi)
    return xs,ys

def refinement(curves,surfs,mesh,opts,ls,ratio):
    """Refines the mesh on the wall in the corrugated
    region and ensures a smooth transion to the coarser
    mesh in the flat part of the wall.

    Parameters
    ----------
    curves : list
        List of tags of corrugation curves 
    mesh : type
        Mesh to be refined.
    opts : type
        Necessary to set mesh options.
    ls : float
        Mesh size in the coarse region.
    ratio: float
        Refinement factor. The mesh size in the
        refined region is ls/ratio.
    
    Returns
    -------
    None
    
    """
    ls = ls[0] if isinstance(ls,list) else ls
    field = mesh.field
    setOne = field.setNumber
    setMany = field.setNumbers
    setOpt = opts.setNumber
    f1 = field.add("Distance")
    setMany(f1, "CurvesList", curves)
    setOne(f1, "Sampling", 200)
    f2 = field.add("Threshold")
    setOne(2, "InField", 1)
    setOne(2, "SizeMin", ls/ratio)
    setOne(2, "SizeMax", ls)
    setOne(2, "DistMin", ls)
    setOne(2, "DistMax", 5*ls)
    f3 = field.add("Constant")
    setMany(f3, "SurfacesList", surfs)
    setOne(f3,"VIn",ls/ratio)
    setOne(f3,"VOut",ls)
    f4 = field.add("Min")
    setMany(f4, "FieldsList", [2, 3])
    field.setAsBackgroundMesh(4)
    setOpt("Mesh.MeshSizeExtendFromBoundary", 0)
    setOpt("Mesh.MeshSizeFromPoints", 0)
    setOpt("Mesh.MeshSizeFromCurvature", 0)
    setOpt("Mesh.Algorithm", 5)
    return
    

def wavy_wall(waveform,w,A,N,l,q,above,below,ls=0,ratio=1,viz=1):
    """Generates a model of a wall of width w (along the y-axis)
    with corrugations centered at (0,0,0). The corrugation shape
    is set through waveform, its amplitude is A, the number of
    periods is N and its wavelength is l. The corrugation tilt,
    relative to y-axis, is q degrees. The heights (along the z-axis)
    of the flat regions above and below the corrugations are set by
    the variables above and below. The maximum mesh size in the flat
    regions is given by ls (or ls[0] if the mesh is to be exported).
    The maximum mesh size in the corrugated region is ratio times
    smaller than in the flat part.
    
    Optionally, meshes the model, writes the mesh, and visualizes is.

    Parameters
    ----------
    waveform : str
        One of four available shapes:
        sine, sawtooth, triangle, and square.
    w : float
        Plate width.
    A : float
        Corrugation amplitude.
    N : int
        Number of periods.
    l : float
        Wavelength.
    q : float
        Tilt angle of the corrugation in degrees.
    ls : float or list, default=0
        If ls is a float it sets the maximum mesh size in the flat region.
        If ls is a list, containing a float and a string, sets the maximum
        element size in the flat region to the float and exports a file
        named after the string.
    ratio : float
        Refinement factor for the corrugated region.
    viz : bool, default=False
        A flag controlling visualisation.

    Returns
    -------
    None
    
    """
    model,geo,mesh,opts = new_geometry(options=1)
    zs,xs = eval(waveform)(A,N,l)
    ys = np.repeat(-w/2,len(xs))
    xyz = np.transpose([xs,ys,zs])
    ps = [geo.addPoint(*i) for i in xyz]
    if waveform == 'sine':
        profile = [(1,geo.add_bspline(ps))]
        geo.remove([(0,i) for i in ps[1:-1]])
    else:
        lines = [geo.add_line(*i) for i in zip(ps[:-1],ps[1:])]
        profile = [(1,i) for i in lines]
    dz = w*np.tan(q/180*np.pi)
    corrugation = geo.extrude(profile,0,w,dz)
    curves = [i[1] for i in profile+corrugation if i[0]==1]
    surfs = [i[1] for i in corrugation if i[0]==2]
    bot,top = [[corrugation[i][1]] for i in [2,-1]]
    for i in [[0,above],[-w,0],[0,-above-dz]]:
        lastPoint = geo.get_max_tag(0)
        top.append(geo.extrude([(0,lastPoint)],0,*i)[1][1])
    bot.append(geo.extrude([(0,1)],0,0,-below)[1][1])
    for i in [[w,0],[0,below+dz]]:
        lastPoint = geo.get_max_tag(0)
        bot.append(geo.extrude([(0,lastPoint)],0,*i)[1][1])
    geo.add_surface_filling(geo.add_wire(top[::-1]))
    geo.add_surface_filling(geo.add_wire(bot[::-1]))
    geo.synchronize()
    refinement(curves,surfs,mesh,opts,ls,ratio) if ls else None
    meshing(mesh,ls) if ls else None
    vis(0) if viz else None
    return
