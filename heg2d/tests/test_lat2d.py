import numpy as np
from heg2d import lat2d

def test_rs2lat():
  rs = 1.
  alat = lat2d.rs2alat(rs, 1)
  alat_expect = 1.9046256137279147
  assert np.isclose(alat, alat_expect)
  alat3 = lat2d.rs2alat(rs, 3)
  assert np.isclose(alat3, alat_expect*3**0.5)

def test_lat2rs():
  alat = 1.
  rs1 = lat2d.alat2rs(alat, 1)
  rs_expect = 0.525037567904332
  assert np.isclose(rs1, rs_expect)
  rs3 = lat2d.alat2rs(alat, 3)
  assert np.isclose(rs3, rs_expect/3**0.5)
