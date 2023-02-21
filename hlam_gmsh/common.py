"""Common functions used by the other modules.

"""
import os
import gmsh
import trimesh
import numpy as np
import multiprocessing as mp
from matplotlib import pyplot as plt
from .io import read as gread

def restart_GMSH():
    """Shorthand to close a running instance of GMSH (if 
    there is one), and starts a new one.
    
    Parameters
    ----------
    None

    Returns
    -------
    None
    
    """
    gmsh.finalize() if gmsh.is_initialized() else None
    gmsh.initialize()
    return

def new_geometry(occ=True):
    """
    Shorthand to restart GMSH (see restart_GMSH)
    and generate a list of commonly used classes, namely:
    1) model
    2) geometry (refered to as geo in the package)
    4) options (refered to as opts in the package)
    The function sets the number of GMSH threads to the
    maximum available and enables parallelized geometry
    computations if the OpenCASCADE kernel is used.
    

    Parameters
    ----------
    occ : bool, default=True
        A flag denoting the CAD kernel to be used: OpenCASCADE if True,
        otherwise the built-in GMSH kernel.

    Returns
    -------
    class,class,class
        Model, geometry, and options classes.
    
    """
    restart_GMSH()
    model = gmsh.model
    geo = gmsh.model.occ if occ else gmsh.model.geo 
    mesh = gmsh.model.mesh
    opts = gmsh.option
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count())
    opts.setNumber("General.NumThreads",mp.cpu_count())
    opts.setNumber("Geometry.OCCParallel",1) if occ else None
    return model,geo,opts

def check_CAD(model):
    """Determines the CAD kernel (OpenCASCADE or built-in) being used
    via the number of entities in both. As only one or the other is
    usually used, only one of the two contains entities.

    Parameters
    ----------
    model : class
        Model, in which the CAD kernel is determined.

    Returns
    -------
    class
        The active CAD kernel.
    
    """
    occs = [model.occ.getMaxTag(i) for i in range(4)]
    geos = [model.geo.getMaxTag(i) for i in range(4)]
    flag = np.argwhere(np.any([occs,geos],1))[0,0]
    kernel = model.geo if flag else model.occ
    return kernel

def is_part_of(model):
    """Placed in geometric primitive functions to determine whether the
    primitive (rod, ball etc.) is a part of a complex object or not. 

    Parameters
    ----------
    model : class
        Contains the model, geo, and opts classes.

    Returns
    -------
    class,class
        Model and geometry classes.
    
    """
    if model:
        geo = check_CAD(model)
    else:
        model,geo,_ = new_geometry()
    return model,geo

def vis(restart=False,file=None):
    """Shorthand to launch the GMSH GUI.

    Parameters
    ----------
    restart : bool, default=False
        A flag determining whether to executes restart_GMSH or not.
    file : str, default=None
        Import and visualize a .msh file.

    Returns
    -------
    None
    
    """
    restart_GMSH() if restart else None
    gmsh.open(file) if file else None
    gmsh.fltk.run()
    return

def eval_on_surf(model,surfs,f,x0=0.25,y0=0.25):
    """Shorthand to evaluate the function f on surfaces surfs,
    using scaled coordinates [x0,y0], each spanning [0,1]. One
    can evaluate the normal at [x0,y0] with the key words
    'normal' and 'nv' or the XYZ coordinates of a point with
    the key words 'value' and 'val'.

    Parameters
    ----------
    model : class
        Model containing the surface to be evaluated.
    surf : int, tuple, or dimTags
        Surfaces to be evaluated.
    f : str
        Function to be evaluated.
    x0 : float, default=0.25
        Scaled x-coordinate of the parametric point at which
        f is evaluated.
    y0 : float, default=0.25
        Scaled y-coordinate of the parametric point at which
        f is evaluated.

    Returns
    -------
    numpy array
    
    """
    res = []
    surfs = [(2,surfs)] if isinstance(surfs,int) else surfs
    surfs = [surfs] if isinstance(surfs,tuple) else surfs
    fDict = {'nv':model.getNormal,'normal':model.getNormal,
             'val':model.getValue,'value':model.getValue}
    for surf in np.abs(surfs):
        bounds = model.getParametrizationBounds(*surf)
        x = bounds[1][0]*x0+(1-x0)*bounds[0][0]
        y = bounds[1][1]*y0+(1-y0)*bounds[0][1]
        pars = list(surf)+[[x,y]]
        pars = pars[1:] if f in ['nv','normal'] else pars
        res.append(fDict[f](*pars))
    return np.array(res)

def entities_in_vol(model,vol):
    """Generates the dimensional tags for all surfaces,
    all lines and all points inside a body vol, part of model.
    Additionally, evaluates the XYZ coordinates of all points. 

    Parameters
    ----------
    model : class
        Model containing the volume to be evaluated.
    vol : int, tuple, or dimTag
        Volume to be evaluated.

    Returns
    -------
    surface dimTags, line dimTags, point dimTags, numpy array
    
    """
    bounds = model.getBoundary
    evalM = model.getValue
    vol = (3,vol) if isinstance(vol, int) else vol
    vol = [vol] if isinstance(vol,tuple) else vol
    surfs = bounds(vol)
    lines = np.array(bounds(surfs,False))
    lines = np.unique(np.abs(lines[:,1]))
    lines = [(1,i) for i in lines]
    points = np.array(bounds(lines,False))[:,1]
    points = [(0,i) for i in np.unique(points)]
    xyz = np.array([model.getValue(*i,[]) for i in points])
    return surfs,lines,points,xyz

def meshing(model,ls,vol=-1):
    """Meshes the current model, checks if the mesh is
    watertight, outputs a 3D model with bad faces (if any),
    and prints quality of element statistics such as mean,
    std, worst and a histogram.

    Parameters
    ----------
    model : class
        Model to be meshed.
    ls : float or list
        If ls is a float it sets the maximum mesh size.
        If ls is a list, containing a float and a string, 
        sets the maximum element size to the float and
        exports a GMSH 2.2 *.msh file using string as name.
    vol : int, tuple, or dimTag, default=-1
        Body for which the normals of each element are set to
        point outward.

    Returns
    -------
    None
    
    """
    mesh = model.mesh
    writing = False
    vol = vol[0] if isinstance(vol,list) else vol
    vol = vol[1] if isinstance(vol,tuple) else vol
    mesh.setOutwardOrientation(vol) if vol>0 else None
    if isinstance(ls,list):
        ls,name = ls
        name = name+'.msh' if name[-4:] != '.msh' else name
        writing = True
    gmsh.option.setNumber("Mesh.MeshSizeMax", ls)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2)
    mesh.generate(2)
    mesh.setOrder(2)
    gmsh.write(name) if writing else None
    gmsh.open(name) if writing else None
    elms = mesh.get_elements(2)[1][0]
    ps = mesh.get_nodes()[1].reshape(-1,3)
    msg = 'Mesh: '+str(len(ps))+' points in '
    print(msg+str(len(elms))+' elements.')
    stat = [' (mean),',' (SD),',' (worst).']
    for k in ['minSICN','gamma']:
        qs = mesh.get_element_qualities(elms,k)
        enum = enumerate([qs.mean(),qs.std(),qs.min()])
        dat = ' '.join([str(round(j,2))+stat[i] for i,j in enum])
        print('Element quality ('+k+'): '+dat)
        plt.figure()
        plt.hist(qs,25)
        plt.xlabel('Element quality ('+k+')')
        plt.ylabel('Count (total: '+str(len(qs))+')')
    if writing:
        ps,elms,_ = gread(name,0)[:3]
    else:
        elms = mesh.get_elements(2)[2][0].reshape(-1,6)-1
    flatTris = [[0,3,5],[1,4,3],[2,5,4],[3,4,5]]
    flatTris = elms[:,flatTris].reshape(-1,3)
    trim = trimesh.Trimesh(ps,flatTris)
    chi = trim.euler_number
    msgN = 'Mesh is not watertight!'
    msgY = 'Mesh is watertight.'
    print(msgY) if trim.is_watertight else print(msgN)
    print('Genus (holes): '+str(int(1-chi/2)))
    if not trim.is_watertight:
        trimesh.repair.broken_faces(trim, color=[255, 0, 0, 255])
        a = trimesh.scene.scene.Scene(trim)
        a.show('gl',smooth=True,resolution=(750,750))
    return
