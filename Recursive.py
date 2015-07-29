# Contains a set of routines for building recursive structures.
# Mostly focused on armchair nanoribbons
# The N numbering convention is used everywhere unless noted.
import numpy as np
from numpy import dot
from GF import *
import collections
import functools
from operator import mul
import random

rtol = 1.0e-4		# Default tolerance for recursive methods.


def HArmStrip(N):
  """Creates the Hamiltonian of an armchair strip (building block of a nanoribbon)."""
  H = np.zeros([2*N,2*N])
  # Adject elements
  for i in range(N-1):
    H[i,i+1], H[i+1,i] = 2*(t,)
  for i in range(N,2*N-1):
    H[i,i+1], H[i+1,i] = 2*(t,)
  # Other elements
  for i in range(0,N,2):
    H[i,N+i], H[N+i,i] = 2*(t,)
  return H


def HBigArmStrip(N,p):
  """Creates the Hamiltonian for an armchair strip N atoms across and p "unit cells" in width."""
  if N%2 == 0:
    return HBigArmStripEven(N,p)
  else:
    return HBigArmStripOdd(N,p)


def HBigArmStripOdd(N,p):
  """Creates the Hamiltonian for an odd armchair strip N atoms across and p "unit cells" in width."""
  H = np.zeros((2*N*p,2*N*p))
  # nn elements
  for j in range(0,2*p*N-N+1,N):
    for i in range(j,j+N-1):
      H[i,i+1], H[i+1,i] = 2*(t,)
  # off diagonal elements
  for i in range(0,2*N*p-N,2):
    H[i,i+N], H[i+N,i] = 2*(t,)
  return H


def HBigArmStripEven(N,p):
  """Creates the Hamiltonian for an even armchair strip N atoms in width and p unit cells in length."""
  H = np.zeros((2*N*p,2*N*p))
  for j in range(0,2*p*N-N+1,N):
    for i in range(j,j+N-1):
      H[i,i+1], H[i+1,i] = 2*(t,)
  for j in range(0,2*p*N-2*N+1,2*N):
    for i in range(j,j+N-1,2):
      H[i,i+N], H[i+N,i] = 2*(t,)
  for j in range(N,2*p*N-3*N+1,2*N):
    for i in range(j+1,j+N,2):
      H[i,i+N], H[i+N,i] = 2*(t,)
  return H


def HBigArmStripSubs(N,p,Imp_List):
  """Creates the Hamiltonian for an armchair strip N atoms across and p "unit cells" in width.
  Adds an impurity with energy eps_imp at ever site in Imp_List."""
  if N%2 == 0:
    return HBigArmStripSubsEven(N,p,Imp_List)
  else:
    return HBigArmStripSubsOdd(N,p,Imp_List)
  return H


def HBigArmStripSubsOdd(N,p,Imp_List):
  """Creates the Hamiltonian for the large odd armchair strip with impurities"""
  H = np.zeros((2*N*p,2*N*p))
  for j in range(1,2*p+1):
    for i in range((j-1)*N,j*N-1):
      H[i,i+1], H[i+1,i] = 2*(t,)
  for i in range(0,2*N*p-N,2):
    H[i,i+N], H[i+N,i] = 2*(t,)
  for i in Imp_List:
    H[i,i] = eps_imp
  return H


def HBigArmStripSubsEven(N,p,Imp_List):
  """Creates the Hamiltonian for the large even armchair strip with impurities"""
  H = np.zeros((2*N*p,2*N*p))
  for j in range(0,2*p*N-N+1,N):
    for i in range(j,j+N-1):
      H[i,i+1], H[i+1,i] = 2*(t,)
  for j in range(0,2*p*N-2*N+1,2*N):
    for i in range(j,j+N-1,2):
      H[i,i+N], H[i+N,i] = 2*(t,)
  for j in range(N,2*p*N-3*N+1,2*N):
    for i in range(j+1,j+N,2):
      H[i,i+N], H[i+N,i] = 2*(t,)
  for i in Imp_List:
    H[i,i] = eps_imp
  return H


def HBigArmStripTop(N,p,Imp_List):
  """Creates the Hamiltonian for an odd armchair strip N atoms across and p "unit cells" in width.
  Populates the strip with a number of top-adsorbed impurities given in Imp_list. These are added in the higher elements of the mx.
  It uses the N numbering convention."""
  if N%2 == 0:
    return HBigArmStripTopEven(N,p,Imp_List)
  else:
    return HBigArmStripTopOdd(N,p,Imp_List)
  return H


def HBigArmStripTopOdd(N,p,Imp_List):
  """Creates the Hamiltonian for an odd armchair strip N atoms across and p "unit cells" in width.
  Populates the strip with a number of top-adsorbed impurities given in Imp_list. These are added in the higher elements of the mx.
  It uses the N numbering convention."""
  nimp = len(Imp_List)
  H = np.zeros((2*N*p+nimp,2*N*p+nimp))
  for j in range(1,2*p+1):
    for i in range((j-1)*N,j*N-1):
      H[i,i+1], H[i+1,i] = 2*(t,)
  for i in range(0,2*N*p-N,2):
    H[i,i+N], H[i+N,i] = 2*(t,)
  
  for i,k in enumerate(Imp_List):
    H[2*N*p+i,k] = tau
    H[k,2*N*p+i] = tau
    H[2*N*p+i,2*N*p+i] = eps_imp
  return H


def HBigArmStripTopEven(N,p,Imp_List):
  """Creates the Hamiltonian for an even armchair strip N atoms across and p "unit cells" in width.
  Populates the strip with a number of top-adsorbed impurities given in Imp_list. These are added in the higher elements of the mx.
  It uses the N numbering convention."""
  nimp = len(Imp_List)
  H = np.zeros((2*N*p+nimp,2*N*p+nimp))
  for j in range(0,2*p*N-N+1,N):
    for i in range(j,j+N-1):
      H[i,i+1], H[i+1,i] = 2*(t,)
  for j in range(0,2*p*N-2*N+1,2*N):
    for i in range(j,j+N-1,2):
      H[i,i+N], H[i+N,i] = 2*(t,)
  for j in range(N,2*p*N-3*N+1,2*N):
    for i in range(j+1,j+N,2):
      H[i,i+N], H[i+N,i] = 2*(t,)
  
  for i,k in enumerate(Imp_List):
    H[2*N*p+i,k] = tau
    H[k,2*N*p+i] = tau
    H[2*N*p+i,2*N*p+i] = eps_imp
  return H


def HBigArmStripCenter(N,p,Imp_List):
  """Creates the Hamiltonian for an odd armchair strip N atoms across and p "unit cells" in width.
  Populates the strip with a number of center-adsorbed impurities given in Imp_list. These are added in the higher elements of the mx."""
  if N%2 == 0:
    return HBigArmStripCenterEven(N,p,Imp_List)
  else:
    return HBigArmStripCenterOdd(N,p,Imp_List)
  return H


def HBigArmStripCenterOdd(N,p,Imp_List):
  """Creates the Hamiltonian for an odd armchair strip N atoms across and p "unit cells" in width.
  Populates the strip with a number of center-adsorbed impurities given in Imp_list. These are added in the higher elements of the mx."""
  nimp = len(Imp_List)
  H = np.zeros((2*N*p+nimp,2*N*p+nimp))
  for j in range(1,2*p+1):
    for i in range((j-1)*N,j*N-1):
      H[i,i+1], H[i+1,i] = 2*(t,)
  for i in range(0,2*N*p-N,2):
    H[i,i+N], H[i+N,i] = 2*(t,)
  
  for i,k in enumerate(Imp_List):
    for j in range(3) + range(N,N+3):
      H[2*N*p+i,k+j] = tau
      H[k+j,2*N*p+i] = tau
    H[2*N*p+i,2*N*p+i] = eps_imp
  return H


def HBigArmStripCenterEven(N,p,Imp_List):
  """Creates the Hamiltonian for an even armchair strip N atoms across and p "unit cells" in width.
  Populates the strip with a number of center-adsorbed impurities given in Imp_list. These are added in the higher elements of the mx."""
  nimp = len(Imp_List)
  H = np.zeros((2*N*p+nimp,2*N*p+nimp))
  for j in range(0,2*p*N-N+1,N):
    for i in range(j,j+N-1):
      H[i,i+1], H[i+1,i] = 2*(t,)
  for j in range(0,2*p*N-2*N+1,2*N):
    for i in range(j,j+N-1,2):
      H[i,i+N], H[i+N,i] = 2*(t,)
  for j in range(N,2*p*N-3*N+1,2*N):
    for i in range(j+1,j+N,2):
      H[i,i+N], H[i+N,i] = 2*(t,)
  
  for i,k in enumerate(Imp_List):
    for j in range(3) + range(N,N+3):
      H[2*N*p+i,k+j] = tau
      H[k+j,2*N*p+i] = tau
    H[2*N*p+i,2*N*p+i] = eps_imp
  return H



def gArmStrip(E,N):
  """Calculates the GF of an armchair strip.
  Might cost a bit to calculate the hamiltonian every time"""
  return inv(E*np.eye(2*N) - HArmStrip(N))


def gBigArmStrip(E,N,p):
  """Calculates the GF for the large armchair strip"""
  return inv(E*np.eye(2*N*p) - HBigArmStrip(N,p))


def gBigArmStripSubs(E,N,p,Imp_List):
  """Calculates the GF for a large armchair strip with substituational impurities at the sites specified in Imp_List."""
  return inv(E*np.eye(2*N*p) - HBigArmStripSubs(N,p,Imp_List))


def gBigArmStripTop(E,N,p,Imp_List):
  """Calculates the GF for a large armchair strip with top adsorbed impurities at the sites specified in Imp_List."""
  nimp = len(Imp_List)
  return inv(E*np.eye(2*N*p+nimp) - HBigArmStripTop(N,p,Imp_List))


def gBigArmStripCenter(E,N,p,Imp_List):
  nimp = len(Imp_List)
  """Calculates the GF for a large armchair strip with center adsorbed impurities at the sites specified in Imp_List."""
  return inv(E*np.eye(2*N*p+nimp) - HBigArmStripCenter(N,p,Imp_List))



def VArmStrip(N):
  """Calculates the LR and RL connection matrices for the armchair strip."""
  VLR, VRL = np.zeros([2*N,2*N]),np.zeros([2*N,2*N])
  for i in range(1,N,2):
    VLR[N+i,i], VRL[i,N+i] = 2*(t,)
  return VLR, VRL


def VArmStripBigSmall(N,p):
  """Connection matrices for the RIGHT SIDE of the Big Strip to the LEFT SIDE of the regular strip."""
  VLR = np.zeros((2*N*p,2*N))
  VRL = np.zeros((2*N,2*N*p))
  for i in range(1,N,2):
    VLR[2*p*N-N+i,i] = t
    VRL[i,2*p*N-N+i] = t
  return VLR, VRL
   
   
def VArmStripSmallBig(N,p):
  """Connection matrices for the LEFT SIDE of the Big Strip to the RIGHT SIDE of the regular strip."""
  VLR = np.zeros((2*N,2*N*p))
  VRL = np.zeros((2*N*p,2*N))
  for i in range(1,N,2):
    VLR[N+i,i], VRL[i,N+i] = 2*(t,)
  return VLR, VRL



def RecAdd(g00,g11,V01,V10):
  """Add a cell g11 to g00 recursively using the Dyson Formula, get a cell G11"""
  return dot(inv(np.eye(g11.shape[0])-dot(dot(g11,V10), dot(g00,V01))),g11)


def gEdge(gC,V01,V10,tol=rtol):
  """Obtain a semi-infinite lead by simple recusive addition (in the 1 direction)"""
  g11 = gC.copy()
  gtemp = np.zeros(2*N)
  while np.linalg.norm(g11 - gtemp)/np.linalg.norm(g11) > tol:
    gtemp = g11.copy()
    g11 = RecAdd(g11,gC,V01,V10)
  return g11


def gOffDiagonal(g22,g11,g10,g01,g00,V12,V21):
  """Connects a cell 2 to a lead ending in cell 1 also containing a cell 0.
  Returns the off diagonal elements G22, G20, G02, G00 """
  G22 = dot(inv(np.eye(g22.shape[0])  - dot(dot(g22,V21),dot(g11,V12)) ),g22)
  G20 = dot(G22,dot(V21,g10))
  G00 = g00 + dot(g01,dot(V12,G20))
  G02 = dot(g01,dot(V12,G22))
  return G22, G20, G02, G00


def RubioSancho(g00,V01,V10,tol=rtol):
  """Uses the Rubio Sancho method to get the GF at the end of a semi-infinite lead."""
  smx = dot(g00,V01)
  tmx = dot(g00,V10)
  identity = np.eye(g00.shape[0]) 
  sprod = identity

  Transf = tmx.copy()
  Transf_Old = np.zeros(g00.shape[0])	# Broadcasts to correct shape
  while np.linalg.norm(Transf - Transf_Old)/np.linalg.norm(Transf) > tol:	# this condition is well dodge, surely you want, like, the relative error?
    # Update t
    temp = inv( identity - dot(tmx,smx) - dot(smx,tmx))
    tmx = dot( temp, dot(tmx,tmx) )
    
    # Update sprod
    sprod = dot(sprod,smx)
  
    # Update Transfer mx
    Transf_Old = Transf.copy()
    Transf = Transf + dot(sprod,tmx)

    # Update s
    smx = dot( temp,  dot(smx,smx) )
  return dot( inv( identity - dot( dot(g00,V01) , Transf)) ,  g00 ) 


def gRibArmRecursive(N,E):
  """Calculates the GF (in a single unit cell) recursively. 
  Probably should be thrown out once you have better routines."""
  VLR, VRL = VArmStrip(N)	# Takes up a little bit of space since this has to be done every E, but really shouldn't have much effect on performance
  gC = gArmStrip(E,N)
  gR = RubioSancho(gC,VRL,VLR)	# The RIGHTMOST cell, not the cell on the right when we add
  gL = RubioSancho(gC,VLR,VRL)
  return RecAdd(gR,gL,VLR,VRL)


def Kubo(N,p,E):
  """Calculates the conductance of a pristine GNR using the Kubo Formula
  This is calculated by connecting a small strip to a big strip to a small strip, which is utterly pointless for the pristine case, and here mainly as an exercise.
  Probably should at some point update this so that it just takes regular strips"""
  def KuboMxs(E): 
    """Gets the appropriate matrices for the Kubo formula.
    Cell 0 on left and cell 1 on the right.
    We need only take the first 2Nx2N matrix in BigStrip to calculate the conductance"""
    gC = gArmStrip(E,N)
    gL = RubioSancho(gC,VsRsL,VsLsR)
    gR = RubioSancho(gC,VsLsR,VsRsL)
    gM = gBigArmStrip(E,N,p)
    GM = RecAdd(gR,gM,VsRbL,VbLsR)
    G11, G10, G01, G00 = gOffDiagonal(GM,gL,gL,gL,gL,VsLbR,VbRsL)
    return G11, G10, G01, G00

  def Gtilde(E):
    """Calculates Gtilde, the difference between advanced and retarded GFs, mulitplied by some stupid complex constant"""
    G11A, G10A, G01A, G00A = KuboMxs(E+1j*eta)
    G11R, G10R, G01R, G00R = KuboMxs(E-1j*eta)
    
    G11T = -1j/2.0*(G11A-G11R)
    G10T = -1j/2.0*(G10A-G10R)
    G01T = -1j/2.0*(G01A-G01R)
    G00T = -1j/2.0*(G00A-G00R)
    
    return G11T[:2*N,:2*N], G10T[:2*N,:2*N], G01T[:2*N,:2*N], G00T[:2*N,:2*N]
  
  VsLsR, VsRsL = VArmStrip(N)		# Notation VsLsR means that a small strip on the left connects to a small strip on the right
  VbLsR, VsRbL = VArmStripBigSmall(N,p)
  VsLbR, VbRsL = VArmStripSmallBig(N,p)
  
  G11T, G10T, G01T, G00T = Gtilde(E)
  V01, V10 = VArmStrip(N)
  
  return np.trace( dot(dot(-G10T,V01),dot(G10T,V01)) + dot(dot(G00T,V01),dot(G11T,V10)) + dot(dot(G11T,V10),dot(G00T,V01)) - dot(dot(G01T,V10),dot(G01T,V10)) )


def KuboSubs(N,p,Imp_List,E):
  """Calculates the Kubo formula for a GNR.
  The GNR is built by connecting a strip to a big strip to another strip.
  This is somewhat pointless for the pristine case, and this is here mainly as a useful exercise."""
  def KuboMxs(E): 
    """Gets the appropriate matrices for the Kubo formula.
    Cell 0 is the leftmost cell (regular strip size) and cell 1 is an adjacent cell on the right (BigStrip size)"""
    gC = gArmStrip(E,N)
    gL = RubioSancho(gC,VsRsL,VsLsR)
    gR = RubioSancho(gC,VsLsR,VsRsL)
    gM = gBigArmStripSubs(E,N,p,Imp_List)
    GM = RecAdd(gR,gM,VsRbL,VbLsR)
    G11, G10, G01, G00 = gOffDiagonal(GM,gL,gL,gL,gL,VsLbR,VbRsL)
    return G11, G10, G01, G00

  def Gtilde(E):
    """Calculates Gtilde, the difference between advanced and retarded GFs"""
    G11A, G10A, G01A, G00A = KuboMxs(E+1j*eta)
    G11R, G10R, G01R, G00R = KuboMxs(E-1j*eta)
    
    G11T = 1.0/(2.0*1j)*(G11A-G11R)
    G10T = 1.0/(2.0*1j)*(G10A-G10R)
    G01T = 1.0/(2.0*1j)*(G01A-G01R)
    G00T = 1.0/(2.0*1j)*(G00A-G00R)
    
    return G11T, G10T, G01T, G00T
  
  VsLsR, VsRsL = VArmStrip(N)
  VbLsR, VsRbL = VArmStripBigSmall(N,p)
  VsLbR, VbRsL = VArmStripSmallBig(N,p)
  
  G11T, G10T, G01T, G00T = Gtilde(E)
  G11 = G11T[:2*N,:2*N]
  G10 = G10T[:2*N,:2*N]
  G01 = G01T[:2*N,:2*N]
  G00 = G00T[:2*N,:2*N]
  V01, V10 = VArmStrip(N)
  
  return np.trace( dot(dot(-G10,V01),dot(G10,V01)) + dot(dot(G00,V01),dot(G11,V10)) + dot(dot(G11,V10),dot(G00,V01)) - dot(dot(G01,V10),dot(G01,V10)) )


def KuboTop(N,p,Imp_List,E):
  def KuboMxs(E): 
    """Gets the appropriate matrices for the Kubo formula.
    Cell 0 is the leftmost cell (regular strip size) and cell 1 is an adjacent cell on the right (BigStrip size)"""
    gC = gArmStrip(E,N)
    gL = RubioSancho(gC,VRLs,VLRs)
    gR = RubioSancho(gC,VLRs,VRLs)
    gM = gBigArmStripTop(E,N,p,Imp_List)
    GM = RecAdd(gR,gM,VRLbs,VLRbs)
    G11, G10, G01, G00 = gOffDiagonal(GM,gL,gL,gL,gL,VLRsb,VRLsb)
    return G11, G10, G01, G00

  def Gtilde(E):
    """Calculates Gtilde, the difference between advanced and retarded GFs"""
    G11A, G10A, G01A, G00A = KuboMxs(E+1j*eta)
    G11R, G10R, G01R, G00R = KuboMxs(E-1j*eta)
    
    G11T = 1.0/(2.0*1j)*(G11A-G11R)
    G10T = 1.0/(2.0*1j)*(G10A-G10R)
    G01T = 1.0/(2.0*1j)*(G01A-G01R)
    G00T = 1.0/(2.0*1j)*(G00A-G00R)
    
    return G11T, G10T, G01T, G00T
  
  VLRs, VRLs = VArmStrip(N)
  VLRbs, VRLbs = VArmStripBigSmall(N,p)
  VLRsb, VRLsb = VArmStripSmallBig(N,p)
  
  G11T, G10T, G01T, G00T = Gtilde(E)
  G11 = G11T[:2*N,:2*N]
  G10 = G10T[:2*N,:2*N]
  G01 = G01T[:2*N,:2*N]
  G00 = G00T[:2*N,:2*N]
  V01, V10 = VArmStrip(N)
  
  return np.trace( dot(dot(-G10,V01),dot(G10,V01)) + dot(dot(G00,V01),dot(G11,V10)) + dot(dot(G11,V10),dot(G00,V01)) - dot(dot(G01,V10),dot(G01,V10)) )


if __name__ == "__main__":
  #N = 12
  p = 4
  for N in [6,7,8,9]:
    El = np.linspace(-3.0,3.0,201)
    Kl = [Kubo(N,p,E) for E in El]
    pl.plot(El,Kl)
    pl.show()




      
