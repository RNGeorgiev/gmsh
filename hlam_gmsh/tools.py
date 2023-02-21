import numpy as np
import pandas as pd
from numpy_indexed import indices as fastIdx

def fast2Dwhere(A):
    """Finds the positions of all non-negative integers
    in a two-dimensional array.

    Parameters
    ----------
    A : 2D numpy array
        Array in which the unique entries are mapped.

    Returns
    -------
    array,list,list
        The array contains the number of times each integer
        appears in A. At position 0 are the occurances of 0,
        at position 1 are the occurances of 1, etc. The first
        list comprises arrays of the rows in which each entry
        of A appears. The second list comprises arrays of the
        columns in which each entry of A appears.
    """
    w = A.shape[1]
    A = A.ravel()
    idxs = np.argsort(A, kind='mergesort')
    lens = np.bincount(A)
    rows = np.split(idxs//w, np.cumsum(lens[:-1]))
    cols = np.split(idxs%w, np.cumsum(lens[:-1]))
    return lens,rows,cols

def nodeInElmAtPos(elms):
    """Given an N-by-6 array of elements elms (row Idx comprises
    the indices of the nodes in element Idx) returns an array
    containing occurances of each node ndInAt. ndInAt[:,0] lists
    the number of elements each nodes is part of. ndInAt[i,1::2]
    lists the element indices node i is part of. ndInAt[i,2::2]
    lists the position of node i within the aformentioned element.
    Each row is padded to a total length of (2*max(ndInAt[:,0])+1)
    with -1.
    
    Parameters
    ----------
    elms : N-by-6 numpy array
        Element array (N = total number of elements)
    
    Returns
    -------
    array
        Array with total occurances, element indices and
        element positions.
    
    """
    ls,rows,cols = fast2Dwhere(elms)
    ndInAt = -np.ones((len(ls),ls.max()*2+1),int)
    ndInAt[:,0] = ls
    for i,j in enumerate(ls):
        ndInAt[i,1:2*j+1:2] = rows[i]
        ndInAt[i,2:2*j+1:2] = cols[i]
    return ndInAt

def node_reorder(ps,elms,ndInAt):
    """Reorders the nodes in ps and elements in elm by
    proximity starting from (one of) the most connected
    node(s).
    
    Parameters
    ----------
    ps : N-by-3 numpy array
        Node array (N = total number of nodes)
    elms : N-by-6 numpy array
        Element array (N = total number of elements)
    ndInAt : N-by-M numpy array
        Node-in-element map
            N = total number of nodes
            M = 2*(maximum connectivity of nodes)+1
    
    Returns
    -------
    array,array,array,array,array
        1) reordered node array newPs
        2) reordered element array newElms
        3) updated node-in-element map newNdInAt
        4) node map between ps and newPs
        5) element map between elms and newElms
    
    """
    seeds=[np.argmax(ndInAt[:,0])]
    nSeeds=seeds
    while len(seeds)<len(ndInAt):
        newElms=pd.unique(ndInAt[nSeeds][:,1::2].ravel())
        if -1 in newElms:
            newElms=np.delete(newElms,np.where(newElms==-1)[0])
        tmp=pd.unique(elms[newElms].ravel())
        nSeeds=np.setdiff1d(tmp,seeds)
        nSeeds=nSeeds[np.hstack([np.where(nSeeds==i)[0] for i in tmp])]
        seeds=seeds+list(nSeeds)
    newPs=ps[seeds]
    newElms=pd.unique(ndInAt[seeds,1::2].ravel())
    newElms=elms[np.delete(newElms,np.where(newElms==-1)[0])]
    ls,rows,cols=fast2Dwhere(newElms)
    for i,j in enumerate(seeds):
        newElms[(rows[j],cols[j])]=i
    newNdInAt=nodeInElmAtPos(newElms)
    mapPs = fastIdx(ps, newPs).astype(int)
    mapElms = fastIdx(np.sort(elms,1),np.sort(mapPs[newElms],1))
    return newPs,newElms,newNdInAt,mapPs,mapElms.astype(int)

def group_elements(msh,tags):
    """Used when reading the data msh in a GMSH 2.2 *.msh
    file. Divides the mesh elements into physical groups.
    
    Parameters
    ----------
    msh : list
        Lines of a *.msh file.
    tags : numpy array
        Tags representing a physical group for each element.
        
    Returns
    -------
    dict,array
        1) Element dictionary:
            {physical group:array of element indices}.
        2) Activity per element.
    
    """
    physN = '$PhysicalNames'
    startGs = [i+2 for i,j in enumerate(msh) if j==physN][0]
    nGs = int(msh[startGs-1])
    gs = [msh[i].split(' ') for i in range(startGs,startGs+nGs)]
    gsInv = {i[1]:i[2].replace('"','') for i in gs}
    gs = {i[2].replace('"',''):int(i[1]) for i in gs}
    gElms = {i:np.where(tags==gs[i])[0] for i in gs.keys()}
    gPs = np.array([gsInv[str(i)] for i in tags])
    return gElms,gPs
    
def act_via_dict(msh,acts,ps,elms):
    """Used when reading the data msh in a GMSH 2.2 *.msh
    file. Sets the activity of each node according to the
    element rules given in the dictionary acts.
    
    Parameters
    ----------
    msh : list
        Lines of a *.msh file.
    acts : dictionary
        Tags representing a physical group for each element
    ps : N-by-3 numpy array
        Node array (N = total number of nodes)
    elms : N-by-6 numpy array
        Element array (N = total number of elements)
        
    Returns
    -------
    dict,array
        1) Element dictionary:
            {physical group:array of element indices}.
        2) Activity per node.
    
    """
    tags = np.array(elms,dtype=int)[:,3]
    elms = np.array(elms,dtype=int)[:,-6:]-1
    ndIn = [i[1:i[0]*2+1:2] for i in nodeInElmAtPos(elms)]
    if '$PhysicalNames' not in msh:
        raise Exception('Physical groups missing from mesh file')
    gElms,gPs = group_elements(msh,tags)
    aPs = [sum([acts[i] for i in gPs[j]])/len(j) for j in ndIn]
    return gElms,np.array(aPs)

def act_via_num(msh,acts,ps,elms):
    """Used when reading the data msh in a GMSH 2.2 *.msh
    file. Sets the activity of each node to the value acts.
    
    Parameters
    ----------
    msh : list
        Lines of a *.msh file.
    acts : int or float
        Constant node activity
    ps : N-by-3 numpy array
        Node array (N = total number of nodes)
    elms : N-by-6 numpy array
        Element array (N = total number of elements)
        
    Returns
    -------
    dict,array
        1) Element dictionary:
            {physical group:array of element indices}.
        2) Activity per node.
    
    """
    gElms = None
    if '$PhysicalNames' in msh:
        tags = np.array(elms,dtype=int)[:,3]
        gElms,_ = group_elements(msh,tags)
    aPs = np.repeat(acts,ps.shape[0])
    return gElms,np.array(aPs)

def act_via_str(msh,acts,ps,elms):
    """Used when reading the data msh in a GMSH 2.2 *.msh
    file. Sets the activity of each node according to the
    node rule given in the string acts.
    
    Parameters
    ----------
    msh : list
        Lines of a *.msh file.
    acts : string
        Represents a function to be evaluated for all nodes
        using their coordinates xs,ys, and zs.
    ps : N-by-3 numpy array
        Node array (N = total number of nodes)
    elms : N-by-6 numpy array
        Element array (N = total number of elements)
        
    Returns
    -------
    dict,array
        1) Element dictionary:
            {physical group:array of element indices}.
        2) Activity per node.
    
    """
    gElms = None
    xs,ys,zs = ps.T
    if '$PhysicalNames' in msh:
        tags = np.array(elms,dtype=int)[:,3]
        gElms,_ = group_elements(msh,tags)
    aPs = eval(acts)
    return gElms,np.array(aPs)

def read(mshFile,acts):
    """Reads in a GMSH 2.2 *.msh file (mshFile) and prepares
    a node array (ps), an element array (elms). Also returns
    the number of nodes nPs and elements nElms in the mesh.
    Additionally, constructs a dictionary, where the value
    of each key is a list of element numbers belonging to
    the corresponding physical group. Finally, computes the
    activity of each node according to the Python type of acts:
        1) string - an expression which can make use of
        the coordinates of the nodes (xs,ys,zs), such as:
        [x**2 if x>0 else 0 for x in xs].
    
        2) float or int - all node activities are set to the
        value of acts.
    
        3) dictionary - the activity of each node is the mean
        of the activities of the elements it is part of. If
        type(acts)!=dict (float, int or string), the mshFile
        need not list physical groups. Otherwise, mshFile must
        contain physical groups and their names must match the
        dictionary keys supplied.
    
    Parameters
    ----------
    mshFile : string
        File name of GMSH 2.2 *.msh file
    acts : 
        Sets the activity of the nodes. See above for details.
    
    Returns
    -------
    array,array,int,int,dict,array
        1) node array
        2) element array
        3) number of nodes in the mesh
        4) number of elements in the mesh
        5) element-physical group map
        6) node acitvities
    
    """
    funcSwitch = {'dict':act_via_dict,'float':act_via_num,
                  'int':act_via_num,'str':act_via_str}
    with open(mshFile) as f:
        msh = f.read().splitlines()
    startPs = [i+2 for i,j in enumerate(msh) if j=='$Nodes'][0]
    startElms = [i+2 for i,j in enumerate(msh) if j=='$Elements'][0]
    nPs,nElms = [int(msh[i-1]) for i in [startPs,startElms]]
    ps = ' '.join(msh[startPs:startPs+nPs])
    ps = np.fromstring(ps,dtype=float,sep=' ').reshape(-1,4)[:,1:]
    elmStrs = [i.split(' ') for i in msh[startElms:startElms+nElms]]
    elms = np.array([i for i in elmStrs if i[1]=='9'])
    gElms,aPs = funcSwitch[type(acts).__name__](msh,acts,ps,elms)
    elms = np.array(elms,dtype=int)[:,-6:]-1
    ndInAt = nodeInElmAtPos(elms)
    if not np.all(ndInAt[:,0]):
        errMsg = 'Point(s) in '+mshFile+' not part of any element'
        raise Exception(errMsg)
    ps,elms,ndInAt,mapPs,mapElms = node_reorder(ps,elms,ndInAt)
    aPs,gElmsN = aPs[mapPs],{}
    if gElms:
        for i in gElms.keys():
            gElmsN.update({i:mapElms.argsort()[gElms[i]]})
    gElms = gElmsN if len(gElmsN.keys()) else None
    return ps,elms,ndInAt,nPs,nElms,gElms,aPs

def write(fileName,ps,elms,groups=None):
    """Writes in a GMSH 2.2 *.msh file (mshFile), prepares
    a node array (ps), an element array (elms). Also returns
    the number of nodes nPs and elements nElms in the mesh.
    Additionally, constructs a dictionary, where the value
    of each key is a list of element numbers belonging to
    the corresponding physical group. Finally, computes the
    activity of each node according to the Python type of acts:
        1) string - an expression which can make use of
        the coordinates of the nodes (xs,ys,zs), such as:
        [x**2 if x>0 else 0 for x in xs].
    
        2) float or int - all node activities are set to the
        value of acts.
    
        3) dictionary - the activity of each node is the mean
        of the activities of the elements it is part of. If
        type(acts)!=dict (float, int or string), the mshFile
        need not list physical groups. Otherwise, mshFile must
        contain physical groups and their names must match the
        dictionary keys supplied.
    
    Parameters
    ----------
    mshFile : string
        File name of GMSH 2.2 *.msh file
    acts : 
        Sets the activity of the nodes. See above for details.
    
    Returns
    -------
    array,array,int,int,dict,array
        1) node array
        2) element array
        3) number of nodes in the mesh
        4) number of elements in the mesh
        5) element-physical group map
        6) node acitvities
    
    """
    groupsStr = '$PhysicalNames\n\n$EndPhysicalNames\n'
    nElms = len(elms)
    iElm = range(1,nElms+1)
    tElm,dElm,eElm = np.tile([9,2,1],(nElms,1)).T
    gElm = np.zeros(nElms,dtype=int)
    if groups:
        gs = ['"'+i+'"' for i in list(groups.keys())]
        gStr = str(len(gs))
        gStr = [gStr]+['2 '+str(i+1)+' '+j for i,j in enumerate(gs)]
        for i,j in enumerate(groups.keys()):
            gElm[groups[j]] = i+1 
    fElms = np.vstack((iElm,tElm,dElm,gElm,gElm,elms.T+1)).T
    with open(fileName,'w') as f:
        f.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n')
        if groups:
            f.write(groupsStr[:15]+'\n'.join(gStr)+groupsStr[15:])
        f.write('$Nodes\n')
        f.write(str(len(ps))+'\n')
        for i,j in enumerate(ps):
            f.write(str(i+1)+' '+' '.join(map(str,j))+'\n')
        f.write('$EndNodes\n$Elements\n')
        f.write(str(nElms)+'\n')
        f.write('\n'.join([' '.join(map(str,i)) for i in fElms]))
        f.write('\n$EndElements')
    return