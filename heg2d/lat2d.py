import numpy as np

def axes3d(axes):
  axes1 = np.zeros([3, 3])
  for iax, avec in enumerate(axes):
    axes1[iax, :2] = avec
  axes1[2, 2] = 1
  return axes1

def pos3d(pos):
  natom, ndim = pos.shape
  pos1 = np.zeros([natom, ndim+1])
  pos1[:, :ndim] = pos
  return pos1

def tile_points(axes0, pos0, tmat):
  """ tile primitive cell using given supercell matrix """
  # make slab in 3D, tile using ase
  from ase import Atoms
  from ase.build import make_supercell
  cell0 = Atoms(cell=axes3d(axes0), positions=pos3d(pos0), pbc=[1, 1, 0])
  cell1 = make_supercell(cell0, axes3d(tmat))
  # get back 2D cell and coordinate
  axes1 = cell1.get_cell()[:2, :2]
  pos1 = cell1.get_positions()[:, :2]
  return axes1, pos1

def alat2rs(alat, nptcl):
  area = np.sqrt(3)/2*alat**2
  rs = (area/nptcl/np.pi)**0.5
  return rs

def rs2alat(rs, nptcl):
  area = nptcl*np.pi*rs**2
  alat = (2./np.sqrt(3)*area)**0.5
  return alat

def axes0_pos0(name, rs):
  """ get primitive cell for "name" lattice at density rs

  Parameters
  ----------
  name : str
    lattice name, must be one of ['tri', 'hex', 'kag'], which stand for
     ['triangular', 'hexagonal', 'kagome'], respectively.
  rs : float
    Wigner-Seitz density parameter

  Returns
  -------
  axes0 : array_like
    lattice vectors, shape (2, 2)
  pos0 : array_like
    particle positions, shape (nptcl, 2)
  """
  upos_map = {
    'tri': np.array([[0.0, 0.0]]),
    'hex': np.array([[0.0, 0.0], [1./3, 1./3]]),
    'kag': np.array([[0.0, 0.0], [0.5, 0], [0, 0.5]])
  }
  if name not in upos_map:
    raise RuntimeError('no %s in %s' % (name, upos_map.keys()))
  upos0 = upos_map[name]
  # determine lattice parameter
  nptcl = len(upos0)
  alat = rs2alat(rs, nptcl)
  # primitive lattice vectors
  axes0 = alat*np.array([
    [1.0, 0.0],
    [1./2, np.sqrt(3)/2]
  ])
  pos0 = np.dot(upos0, axes0)
  return axes0, pos0
