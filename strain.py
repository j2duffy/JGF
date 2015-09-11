from GF import *
from Recursive import *
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


def HStrainArmStrip(N,p=1,SubsList=[],TopList=[],CenterList=[],t1=t,t2=t):
  """Creates the Hamiltonian for an armchair strip N atoms across and p "unit cells" in width.
  Populates it with whatever impurities you desire.
  Matrix order is Prisine->Top->Center."""
  ntop = len(TopList)	# Number of top adsorbed impurities
  ncenter = len(CenterList)	# Number of center adsorbed impurities
  H = np.zeros((2*N*p+ntop+ncenter,2*N*p+ntop+ncenter))		# Make sure our hamiltonian has space for sites+center+top
  # nn elements
  for j in range(0,2*p*N-N+1,N):
    for i in range(j,j+N-1):
      H[i,i+1] = H[i+1,i] = t1
  # Other elements
  for j in range(0,2*p*N-2*N+1,2*N):
    for i in range(j,j+N,2):
      H[i,i+N] = H[i+N,i] = t2
  for j in range(N,2*p*N-3*N+1,2*N):
    for i in range(j+1,j+N,2):
      H[i,i+N] = H[i+N,i] = t2
      
  # Any substitutional impurities
  for i in SubsList:
    H[i,i] = eps_imp
  # Any top adsorbed impurities
  for i,k in enumerate(TopList):
    H[2*N*p+i,k] = H[k,2*N*p+i] = tau
    H[2*N*p+i,2*N*p+i] = eps_imp
  # Any center adsorbed impurities
  for i,k in enumerate(CenterList):
    H[2*N*p+ntop+i,2*N*p+ntop+i] = eps_imp
    for j in range(3) + range(N,N+3):
      H[2*N*p+ntop+i,k+j] = H[k+j,2*N*p+ntop+i] = tau
  return H


def VStrainArmStrip(N,t2=t):
  """Calculates the LR and RL connection matrices for the armchair strip."""
  VLR, VRL = np.zeros([2*N,2*N]),np.zeros([2*N,2*N])
  for i in range(1,N,2):
    VLR[N+i,i] = VRL[i,N+i] = t2
  return VLR, VRL


def VStrainArmStripBigLSmallR(N,p,t2=t):
  """Connection matrices for the RIGHT SIDE of the Big Strip to the LEFT SIDE of the regular strip."""
  VLR = np.zeros((2*N*p,2*N))
  VRL = np.zeros((2*N,2*N*p))
  for i in range(1,N,2):
    VLR[2*p*N-N+i,i] = VRL[i,2*p*N-N+i] = t2
  return VLR, VRL
   
   
def VStrainArmStripSmallLBigR(N,p,t2=t):
  """Connection matrices for the LEFT SIDE of the Big Strip to the RIGHT SIDE of the regular strip."""
  VLR = np.zeros((2*N,2*N*p))
  VRL = np.zeros((2*N*p,2*N))
  for i in range(1,N,2):
    VLR[N+i,i] = VRL[i,N+i] = t2
  return VLR, VRL


def LeadsStrain(N,E):
  """Gets the semi-infinte leads for an armchair nanoribbon of width N.
  Also returns the connection matrices, because we always seem to need them."""
  HC = HStrainArmStrip(N,p=1,t1=t,t2=t)
  VLR, VRL = VArmStrip(N)	
  gC = gGen(E,HC)
  gL = RubioSancho(gC,VRL,VLR)
  gR = RubioSancho(gC,VLR,VRL)
  return gL,gR,VLR,VRL


def KuboStrainSubs(N,p,E,ImpList,t1,t2):
  """Calculates the conductance of a GNR with substitutional impurities (given in ImpList) using the Kubo Formula."""
  gL,gR,VLR,VRL = LeadsStrain(N,E-1j*eta)
  # Scattering region and connection matrices 
  HM = HStrainArmStrip(N,p,SubsList=ImpList,t1=t1,t2=t2)
  gM = gGen(E-1j*eta,HM)
  VbLsR, VsRbL = VArmStripBigLSmallR(N,p)		# Notation VbLsR means a big strip on the left connects to a small strip on the right
  VsLbR, VbRsL = VArmStripSmallLBigR(N,p)

  # Calculate the advanced GFs
  GR = RecAdd(gR,gM,VsRbL,VbLsR)[:2*N,:2*N]	# The new rightmost cell
  
  return Kubo(gL,GR,VLR,VRL)
  

if __name__ == "__main__":
  N = 5
  p = 1
  ImpList = [0]
  t1,t2 = strainA(Seps,Ssigma)
  Elist = np.linspace(-3.0,3.0,201)
  Klist = [KuboStrainSubs(N,p,E,ImpList,t1,t2) for E in Elist]
  pl.plot(Elist,Klist)
  pl.savefig("plot.png")
  pl.show()