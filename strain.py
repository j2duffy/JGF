from GF import *
from math import exp as expRe
from scipy.integrate import dblquad

Seps = 1.0	# The epsilon for strain
Ssigma = 0.165		# Poisson's ration in graphene
Salpha = 3.37		# A strain constant, taken from the literature

def strainZ(Seps,Ssigma):
  """Gets the correct bond length ratios for zigzag strain"""
  ratio1 = 1.0 + 3.0*Seps/4.0 - Seps*Ssigma/4.0	# R1/R0
  ratio2 = 1.0-Seps*Ssigma	# R2/R0
  return ratio1, ratio2


def strainA(Seps,Ssigma):
  """Gets the correct bond length ratios for armchair strain"""
  ratio1 = 1.0 + Seps/4.0 - 3.0/4.0*Seps*Ssigma	# R1/R0
  ratio2 = 1.0-Seps*Ssigma	# R2/R0
  return ratio1, ratio2


def SHopping(ratio1,ratio2):
  t1 = t*expRe(-Salpha*(ratio1-1.0))
  t2 = t*expRe(-Salpha*(ratio2-1.0))
  return t1, t2


def gtest(m,n,s,E):
  """Supposed to be the graphene GF under strain, but having numerical difficulties... or maybe not"""
  def gterm(kA,kZ):  
    if s == 0:
      N = E
    elif s == 1:
      N = t2 + 2*t1*exp(1j*kA)*cos(kZ)
    else:
      N = t2 + 2*t1*exp(-1j*kA)*cos(kZ)
    Beps2 = t2**2+4*t1*t2*cos(kA)*cos(kZ)+4*t1**2 *cos(kZ)**2
    
    g = N*exp(1j*kA*(m+n)+1j*kZ*(m-n))/(E**2-Beps2)
    return g.real
  t1,t2 = strainZ(Seps,Ssigma)
  return dblquad(gterm, -pi/2.0, pi/2.0, lambda kZ: -pi, lambda kZ: pi)
  

if __name__ == "__main__":
  m,n = 0,0
  #kA = 1.1
  #kZ = 0.9
  Elist = np.linspace(-3.0+1j*eta,3.0+1j*eta,101)
  glist = [gtest(m,n,0,E) for E in Elist]
  pl.plot(Elist,glist)
  pl.show()