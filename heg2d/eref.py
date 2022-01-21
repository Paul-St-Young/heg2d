import numpy as np

def ehf(rs, pol=0):
  kinetic = (1+pol**2)/(2*rs**2)
  p32 = (1+pol)**(3./2) + (1-pol)**(3./2)
  potential = -4./(3*np.pi*2**0.5*rs)*p32
  return kinetic + potential

def ec_tanatar_ceperley_1989(rs):
  # correlation energy
  # !!!! hard-code unpolarized case
  a0 = -0.1784  # ha/e
  a1 = 1.13; a2=0.905; a3=0.4165
  x = rs**0.5
  nume = 1+a1*x
  ec = a0*nume/(nume+a2*x**2+a3*x**3)
  return ec

ectc=ec_tanatar_ceperley_1989
