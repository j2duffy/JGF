# Contains a set of routines for building recursive structures.
# Mostly focused on armchair nanoribbons
# The N numbering convention is used everywhere unless noted.
from numpy import dot
from GF import *
from operator import mul
import random

rtol = 1.0e-8		# Default tolerance for recursive methods.
# This tolerance is chosen to eliminate the zero, and is picked purely by observation. 


def HArmStrip(N):
  """Creates the Hamiltonian of an armchair strip (building block of a nanoribbon)."""
  H = np.zeros([2*N,2*N])
  # Adjacent elements
  for i in range(N-1) + range(N,2*N-1):
    H[i,i+1] = H[i+1,i] = t
  # Other elements
  for i in range(0,N,2):
    H[i,N+i] = H[N+i,i] = t
  return H


def HBigArmStrip(N,p):
  """Creates the Hamiltonian for an armchair strip N atoms across and p "unit cells" in width."""
  H = np.zeros((2*N*p,2*N*p))
  # nn elements
  for j in range(0,2*p*N-N+1,N):
    for i in range(j,j+N-1):
      H[i,i+1] = H[i+1,i] = t
  if N%2 == 0:
    # Other elements even
    for j in range(0,2*p*N-2*N+1,2*N):
      for i in range(j,j+N-1,2):
	H[i,i+N] = H[i+N,i] = t
    for j in range(N,2*p*N-3*N+1,2*N):
      for i in range(j+1,j+N,2):
	H[i,i+N] = H[i+N,i] = t
  else:
    # Other elements odd
    for i in range(0,2*N*p-N,2):
      H[i,i+N] = H[i+N,i] = t
      
  return H


def HBigArmStripSubs(N,p,Imp_List):
  """Creates the Hamiltonian for an armchair strip N atoms across and p "unit cells" in width.
  Adds an impurity with energy eps_imp at every site in Imp_List."""
  H = HBigArmStrip(N,p)
  for i in Imp_List:
    H[i,i] = eps_imp
  return H


def HBigArmStripTop(N,p,Imp_List):
  """Creates the Hamiltonian for an armchair strip N atoms across and p "unit cells" in width.
  Populates the strip with a number of top-adsorbed impurities given in Imp_list. 
  These are added in the higher elements of the mx."""
  nimp = len(Imp_List)
  H = np.zeros((2*N*p+nimp,2*N*p+nimp))
  H[:2*N*p,:2*N*p] = HBigArmStrip(N,p)
  
  for i,k in enumerate(Imp_List):
    H[2*N*p+i,k] = H[k,2*N*p+i] = tau
    H[2*N*p+i,2*N*p+i] = eps_imp
  return H


def HBigArmStripCenter(N,p,Imp_List):
  """Creates the Hamiltonian for an armchair strip N atoms across and p "unit cells" in width.
  Populates the strip with a number of center-adsorbed impurities given in Imp_list. 
  These are added in the higher elements of the mx."""
  nimp = len(Imp_List)
  H = np.zeros((2*N*p+nimp,2*N*p+nimp))
  H[:2*N*p,:2*N*p] = HBigArmStrip(N,p)
  
  for i,k in enumerate(Imp_List):
    for j in range(3) + range(N,N+3):
      H[2*N*p+i,k+j] = H[k+j,2*N*p+i] = tau
    H[2*N*p+i,2*N*p+i] = eps_imp
  return H



def gGen(E,H):
  """Calculates the GF given a Hamiltonian"""
  return inv(E*np.eye(H.shape[0]) - H)



def VArmStrip(N):
  """Calculates the LR and RL connection matrices for the armchair strip."""
  VLR, VRL = np.zeros([2*N,2*N]),np.zeros([2*N,2*N])
  for i in range(1,N,2):
    VLR[N+i,i] = VRL[i,N+i] = t
  return VLR, VRL


def VArmStripBigSmall(N,p):
  """Connection matrices for the RIGHT SIDE of the Big Strip to the LEFT SIDE of the regular strip."""
  VLR = np.zeros((2*N*p,2*N))
  VRL = np.zeros((2*N,2*N*p))
  for i in range(1,N,2):
    VLR[2*p*N-N+i,i] = VRL[i,2*p*N-N+i] = t
  return VLR, VRL
   
   
def VArmStripSmallBig(N,p):
  """Connection matrices for the LEFT SIDE of the Big Strip to the RIGHT SIDE of the regular strip."""
  VLR = np.zeros((2*N,2*N*p))
  VRL = np.zeros((2*N*p,2*N))
  for i in range(1,N,2):
    VLR[N+i,i] = VRL[i,N+i] = t
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
  while np.linalg.norm(Transf - Transf_Old)/np.linalg.norm(Transf) > tol:
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
  HC = HArmStrip(N)
  gC = gGen(E,HC)
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
    gC = gGen(E,HC)
    gL = RubioSancho(gC,VsRsL,VsLsR)
    gR = RubioSancho(gC,VsLsR,VsRsL)
    gM = gGen(E,HM)
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
  
  HC = HArmStrip(N)
  HM = HBigArmStrip(N,p)
  VsLsR, VsRsL = VArmStrip(N)		# Notation VsLsR means that a small strip on the left connects to a small strip on the right
  VbLsR, VsRbL = VArmStripBigSmall(N,p)
  VsLbR, VbRsL = VArmStripSmallBig(N,p)
  
  G11T, G10T, G01T, G00T = Gtilde(E)
  V01, V10 = VArmStrip(N)
  
  return np.trace( dot(dot(-G10T,V01),dot(G10T,V01)) + dot(dot(G00T,V01),dot(G11T,V10)) + dot(dot(G11T,V10),dot(G00T,V01)) - dot(dot(G01T,V10),dot(G01T,V10)) )


def KuboSubs(N,p,E,Imp_List):
  """Calculates the conductance of a GNR with substitutional impurities using the Kubo Formula.
  This is calculated by connecting a small strip to a big strip to a small strip, which is utterly pointless for the pristine case, and here mainly as an exercise.
  Probably should at some point update this so that it just takes regular strips"""
  def KuboMxs(E): 
    """Gets the appropriate matrices for the Kubo formula.
    Cell 0 on left and cell 1 on the right.
    We need only take the first 2Nx2N matrix in BigStrip to calculate the conductance"""
    gC = gGen(E,HC)
    gL = RubioSancho(gC,VsRsL,VsLsR)
    gR = RubioSancho(gC,VsLsR,VsRsL)
    gM = gGen(E,HM)
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

  HC = HArmStrip(N)
  HM = HBigArmStripSubs(N,p,Imp_List)
  VsLsR, VsRsL = VArmStrip(N)		# Notation VsLsR means that a small strip on the left connects to a small strip on the right
  VbLsR, VsRbL = VArmStripBigSmall(N,p)
  VsLbR, VbRsL = VArmStripSmallBig(N,p)
  
  G11T, G10T, G01T, G00T = Gtilde(E)
  V01, V10 = VArmStrip(N)
  
  return np.trace( dot(dot(-G10T,V01),dot(G10T,V01)) + dot(dot(G00T,V01),dot(G11T,V10)) + dot(dot(G11T,V10),dot(G00T,V01)) - dot(dot(G01T,V10),dot(G01T,V10)) ).real


def KuboTop(N,p,E,Imp_List):
  """Calculates the conductance of a GNR with substitutional impurities using the Kubo Formula.
  This is calculated by connecting a small strip to a big strip to a small strip, which is utterly pointless for the pristine case, and here mainly as an exercise.
  Probably should at some point update this so that it just takes regular strips"""
  def KuboMxs(E): 
    """Gets the appropriate matrices for the Kubo formula.
    Cell 0 on left and cell 1 on the right.
    We need only take the first 2Nx2N matrix in BigStrip to calculate the conductance"""
    gC = gGen(E,HC)
    gL = RubioSancho(gC,VsRsL,VsLsR)
    gR = RubioSancho(gC,VsLsR,VsRsL)
    gM = gGen(E,HM)
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

  nimp = len(Imp_List)

  HC = HArmStrip(N)
  HM = HBigArmStripTop(N,p,Imp_List)
  VsLsR, VsRsL = VArmStrip(N)		# Notation VsLsR means that a small strip on the left connects to a small strip on the right
  
  VbLsR, VsRbL = np.zeros((2*N*p+nimp,2*N)), np.zeros((2*N,2*N*p+nimp))
  VsLbR, VbRsL = np.zeros((2*N,2*N*p+nimp)), np.zeros((2*N*p+nimp,2*N))
  VbLsR[:2*N*p,:2*N], VsRbL[:2*N,:2*N*p] = VArmStripBigSmall(N,p)
  VsLbR[:2*N,:2*N*p], VbRsL[:2*N*p,:2*N] = VArmStripSmallBig(N,p)
  
  G11T, G10T, G01T, G00T = Gtilde(E)
  V01, V10 = VArmStrip(N)
  
  return np.trace( dot(dot(-G10T,V01),dot(G10T,V01)) + dot(dot(G00T,V01),dot(G11T,V10)) + dot(dot(G11T,V10),dot(G00T,V01)) - dot(dot(G01T,V10),dot(G01T,V10)) ).real


def KuboCenter(N,p,E,Imp_List):
  """Calculates the conductance of a GNR with substitutional impurities using the Kubo Formula.
  This is calculated by connecting a small strip to a big strip to a small strip, which is utterly pointless for the pristine case, and here mainly as an exercise.
  Probably should at some point update this so that it just takes regular strips"""
  def KuboMxs(E): 
    """Gets the appropriate matrices for the Kubo formula.
    Cell 0 on left and cell 1 on the right.
    We need only take the first 2Nx2N matrix in BigStrip to calculate the conductance"""
    gC = gGen(E,HC)
    gL = RubioSancho(gC,VsRsL,VsLsR)
    gR = RubioSancho(gC,VsLsR,VsRsL)
    gM = gGen(E,HM)
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

  nimp = len(Imp_List)

  HC = HArmStrip(N)
  HM = HBigArmStripCenter(N,p,Imp_List)
  VsLsR, VsRsL = VArmStrip(N)		# Notation VsLsR means that a small strip on the left connects to a small strip on the right
  
  VbLsR, VsRbL = np.zeros((2*N*p+nimp,2*N)), np.zeros((2*N,2*N*p+nimp))
  VsLbR, VbRsL = np.zeros((2*N,2*N*p+nimp)), np.zeros((2*N*p+nimp,2*N))
  VbLsR[:2*N*p,:2*N], VsRbL[:2*N,:2*N*p] = VArmStripBigSmall(N,p)
  VsLbR[:2*N,:2*N*p], VbRsL[:2*N*p,:2*N] = VArmStripSmallBig(N,p)
  
  G11T, G10T, G01T, G00T = Gtilde(E)
  V01, V10 = VArmStrip(N)
  
  return np.trace( dot(dot(-G10T,V01),dot(G10T,V01)) + dot(dot(G00T,V01),dot(G11T,V10)) + dot(dot(G11T,V10),dot(G00T,V01)) - dot(dot(G01T,V10),dot(G01T,V10)) ).real


if __name__ == "__main__":
  N = 8
  p = 3
  #nimp = 6
  
  #def KGen(E,niter):
    #KG = 0
    #for i in range(niter):
      #Imp_List = random.sample(range(2*N*p),nimp)
      #KG += KuboTop(N,p,E,Imp_List).real
    #return KG/niter
    
  #El = np.linspace(-3.0,3.0,201)
  #Kl = [KGen(E,1000) for E in El]
  #pl.plot(El,Kl)
  #pl.savefig('MonteCarlo.jpg')
  #pl.clf()
    
    
  Imp_List = [0,10,11,32]
  El = np.linspace(-3.0,3.0,201)
  KlC = [KuboCenter(N,p,E,Imp_List) for E in El]
  Imp_List = [1,15,17,29]
  KlT = [KuboTop(N,p,E,Imp_List) for E in El]
  KlS = [KuboSubs(N,p,E,Imp_List) for E in El]
  pl.plot(El,KlC,label='Center')
  pl.plot(El,KlS,label='Subs')
  pl.plot(El,KlT,label='Top')
  pl.legend()
  #pl.savefig('comp.jpg')
  pl.show()



      
