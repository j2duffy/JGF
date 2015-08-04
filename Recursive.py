# Contains a set of routines for building recursive structures.
# Mostly focused on armchair nanoribbons
# The N numbering convention is used everywhere unless noted.
from numpy import dot
from GF import *
from operator import mul
import random
from itertools import combinations

rtol = 1.0e-8		# Default tolerance for recursive methods.
# This tolerance is chosen to eliminate the zero, and is picked purely by observation. 


def choose(n, k):
    """A fast way to calculate binomial coefficients by Andrew Dalke (contrib)."""
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in xrange(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


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



def Kubo(N,E):
  """Calculates the conductance of a pristine GNR using the Kubo Formula"""
  # Leads 
  HC = HArmStrip(N)
  VLR, VRL = VArmStrip(N)	
  gC = gGen(E-1j*eta,HC)	# The advanced GF
  gL = RubioSancho(gC,VRL,VLR)
  gR = RubioSancho(gC,VLR,VRL)

  GRRa, GRLa, GLRa, GLLa = gOffDiagonal(gR,gL,gL,gL,gL,VLR,VRL)
  
  # Calculates Gtilde, the imaginary part of the advanced GF
  GRRt, GRLt, GLRt, GLLt = GRRa.imag, GRLa.imag, GLRa.imag, GLLa.imag
  
  return np.trace( dot(dot(-GRLt,VLR),dot(GRLt,VLR)) + dot(dot(GLLt,VLR),dot(GRRt,VRL)) + dot(dot(GRRt,VRL),dot(GLLt,VLR)) - dot(dot(GLRt,VRL),dot(GLRt,VRL)) )


def KuboSubs(N,p,E,Imp_List):
  """Calculates the conductance of a GNR with substitutional impurities using the Kubo Formula.
  This is calculated by connecting a small strip to a big strip to a small strip, which is utterly pointless for the pristine case, and here mainly as an exercise.
  Probably should at some point update this so that it just takes regular strips"""
  
  # Leads 
  HC = HArmStrip(N)
  VLR, VRL = VArmStrip(N)	
  gC = gGen(E-1j*eta,HC)	# The advanced GF
  gL = RubioSancho(gC,VRL,VLR)
  gR = RubioSancho(gC,VLR,VRL)

  # Scattering region and connection matrices 
  HM = HBigArmStripSubs(N,p,Imp_List)
  gM = gGen(E-1j*eta,HM)
  VbLsR, VsRbL = VArmStripBigSmall(N,p)		# Notation VbLsR means a big strip on the left connects to a small strip on the right
  VsLbR, VbRsL = VArmStripSmallBig(N,p)

  # Calculate the advanced GFs
  GR = RecAdd(gR,gM,VsRbL,VbLsR)[:2*N,:2*N]	# The new rightmost cell
  GRRa, GRLa, GLRa, GLLa = gOffDiagonal(GR,gL,gL,gL,gL,VLR,VRL)
  
  # Calculates Gtilde, the imaginary part of the advanced GF
  GRRt, GRLt, GLRt, GLLt = GRRa.imag, GRLa.imag, GLRa.imag, GLLa.imag
  
  return np.trace( dot(dot(-GRLt,VLR),dot(GRLt,VLR)) + dot(dot(GLLt,VLR),dot(GRRt,VRL)) + dot(dot(GRRt,VRL),dot(GLLt,VLR)) - dot(dot(GLRt,VRL),dot(GLRt,VRL)) )


def KuboTop(N,p,E,Imp_List):
  """Calculates the conductance of a GNR with top-adsorbed impurities using the Kubo Formula."""
  nimp = len(Imp_List)
  # Leads 
  HC = HArmStrip(N)
  VLR, VRL = VArmStrip(N)	
  gC = gGen(E-1j*eta,HC)	# The advanced GF
  gL = RubioSancho(gC,VRL,VLR)
  gR = RubioSancho(gC,VLR,VRL)

  # Scattering region and connection matrices 
  HM = HBigArmStripTop(N,p,Imp_List)
  gM = gGen(E-1j*eta,HM)
  VbLsR, VsRbL = np.zeros((2*N*p+nimp,2*N)), np.zeros((2*N,2*N*p+nimp))
  VsLbR, VbRsL = np.zeros((2*N,2*N*p+nimp)), np.zeros((2*N*p+nimp,2*N))
  VbLsR[:2*N*p,:2*N], VsRbL[:2*N,:2*N*p] = VArmStripBigSmall(N,p)
  VsLbR[:2*N,:2*N*p], VbRsL[:2*N*p,:2*N] = VArmStripSmallBig(N,p)

  # Calculate the advanced GFs
  GR = RecAdd(gR,gM,VsRbL,VbLsR)[:2*N,:2*N]	# The new rightmost cell
  GRRa, GRLa, GLRa, GLLa = gOffDiagonal(GR,gL,gL,gL,gL,VLR,VRL)
  
  # Calculates Gtilde, the imaginary part of the advanced GF
  GRRt, GRLt, GLRt, GLLt = GRRa.imag, GRLa.imag, GLRa.imag, GLLa.imag
  
  return np.trace( dot(dot(-GRLt,VLR),dot(GRLt,VLR)) + dot(dot(GLLt,VLR),dot(GRRt,VRL)) + dot(dot(GRRt,VRL),dot(GLLt,VLR)) - dot(dot(GLRt,VRL),dot(GLRt,VRL)) )


def KuboCenter(N,p,E,Imp_List):
  """Calculates the conductance of a GNR with top-adsorbed impurities using the Kubo Formula."""
  nimp = len(Imp_List)
  # Leads 
  HC = HArmStrip(N)
  VLR, VRL = VArmStrip(N)	
  gC = gGen(E-1j*eta,HC)	# The advanced GF
  gL = RubioSancho(gC,VRL,VLR)
  gR = RubioSancho(gC,VLR,VRL)

  # Scattering region and connection matrices 
  HM = HBigArmStripCenter(N,p,Imp_List)
  gM = gGen(E-1j*eta,HM)
  VbLsR, VsRbL = np.zeros((2*N*p+nimp,2*N)), np.zeros((2*N,2*N*p+nimp))
  VsLbR, VbRsL = np.zeros((2*N,2*N*p+nimp)), np.zeros((2*N*p+nimp,2*N))
  VbLsR[:2*N*p,:2*N], VsRbL[:2*N,:2*N*p] = VArmStripBigSmall(N,p)
  VsLbR[:2*N,:2*N*p], VbRsL[:2*N*p,:2*N] = VArmStripSmallBig(N,p)

  # Calculate the advanced GFs
  GR = RecAdd(gR,gM,VsRbL,VbLsR)[:2*N,:2*N]	# The new rightmost cell
  GRRa, GRLa, GLRa, GLLa = gOffDiagonal(GR,gL,gL,gL,gL,VLR,VRL)
  
  # Calculates Gtilde, the imaginary part of the advanced GF
  GRRt, GRLt, GLRt, GLLt = GRRa.imag, GRLa.imag, GLRa.imag, GLLa.imag
  
  return np.trace( dot(dot(-GRLt,VLR),dot(GRLt,VLR)) + dot(dot(GLLt,VLR),dot(GRRt,VRL)) + dot(dot(GRRt,VRL),dot(GLLt,VLR)) - dot(dot(GLRt,VRL),dot(GLRt,VRL)) )


def ConfigAvSubsTotal(N,p,nimp,E):
  """Calculates the Kubo Formula for every possible case of nimp substitutional impurities in a ribbon of (N,p).
  Averages all cases.
  Way faster due to leaving the calculation of the leads out of the loop body"""
  # Leads 
  HC = HArmStrip(N)
  VLR, VRL = VArmStrip(N)	
  gC = gGen(E-1j*eta,HC)	# The advanced GF
  gL = RubioSancho(gC,VRL,VLR)
  gR = RubioSancho(gC,VLR,VRL)
  
  KT = 0
  for Imp_List in combinations(range(2*N*p),nimp):	# For every possible combination of positions
    # Scattering region and connection matrices 
    HM = HBigArmStripSubs(N,p,Imp_List)
    gM = gGen(E-1j*eta,HM)
    VbLsR, VsRbL = VArmStripBigSmall(N,p)		# Notation VbLsR means a big strip on the left connects to a small strip on the right
    VsLbR, VbRsL = VArmStripSmallBig(N,p)

    # Calculate the advanced GFs
    GR = RecAdd(gR,gM,VsRbL,VbLsR)[:2*N,:2*N]	# The new rightmost cell
    GRRa, GRLa, GLRa, GLLa = gOffDiagonal(GR,gL,gL,gL,gL,VLR,VRL)
    
    # Calculates Gtilde, the imaginary part of the advanced GF
    GRRt, GRLt, GLRt, GLLt = GRRa.imag, GRLa.imag, GLRa.imag, GLLa.imag
  
    K =  np.trace( dot(dot(-GRLt,VLR),dot(GRLt,VLR)) + dot(dot(GLLt,VLR),dot(GRRt,VRL)) + dot(dot(GRRt,VRL),dot(GLLt,VLR)) - dot(dot(GLRt,VRL),dot(GLRt,VRL)) )
    KT += K
  return  KT/choose(2*N*p,nimp)		# Choose should give the size of our list of combinations


#def ConfigAvTopTotal(N,p,nimp,E):
  #"""Calculates the Kubo Formula for every possible case of nimp top adsorbed impurities in a ribbon of (N,p).
  #Averages all cases."""
  #KT = 0
  #for Imp_List in combinations(range(2*N*p),nimp):	# For every possible combination of positions
    #Kl = KuboTop(N,p,E,Imp_List)
    #KT += Kl
  #return  KT/choose(2*N*p,nimp)		# Choose should give the size of our list of combinations
  

def ConfigAvTopTotal(N,p,nimp,E):
  """Calculates the Kubo Formula for every possible case of nimp substitutional impurities in a ribbon of (N,p).
  Averages all cases.
  Way faster due to leaving the calculation of the leads out of the loop body"""
  # Leads 
  HC = HArmStrip(N)
  VLR, VRL = VArmStrip(N)	
  gC = gGen(E-1j*eta,HC)	# The advanced GF
  gL = RubioSancho(gC,VRL,VLR)
  gR = RubioSancho(gC,VLR,VRL)
  
  KT = 0
  for Imp_List in combinations(range(2*N*p),nimp):	# For every possible combination of positions
    # Scattering region and connection matrices 
    HM = HBigArmStripTop(N,p,Imp_List)
    gM = gGen(E-1j*eta,HM)
    VbLsR, VsRbL = np.zeros((2*N*p+nimp,2*N)), np.zeros((2*N,2*N*p+nimp))
    VsLbR, VbRsL = np.zeros((2*N,2*N*p+nimp)), np.zeros((2*N*p+nimp,2*N))
    VbLsR[:2*N*p,:2*N], VsRbL[:2*N,:2*N*p] = VArmStripBigSmall(N,p)
    VsLbR[:2*N,:2*N*p], VbRsL[:2*N*p,:2*N] = VArmStripSmallBig(N,p)

    # Calculate the advanced GFs
    GR = RecAdd(gR,gM,VsRbL,VbLsR)[:2*N,:2*N]	# The new rightmost cell
    GRRa, GRLa, GLRa, GLLa = gOffDiagonal(GR,gL,gL,gL,gL,VLR,VRL)
    
    # Calculates Gtilde, the imaginary part of the advanced GF
    GRRt, GRLt, GLRt, GLLt = GRRa.imag, GRLa.imag, GLRa.imag, GLLa.imag
  
    K = np.trace( dot(dot(-GRLt,VLR),dot(GRLt,VLR)) + dot(dot(GLLt,VLR),dot(GRRt,VRL)) + dot(dot(GRRt,VRL),dot(GLLt,VLR)) - dot(dot(GLRt,VRL),dot(GLRt,VRL)) )

    KT += K
  return  KT/choose(2*N*p,nimp)		# Choose should give the size of our list of combinations


def CenterPositions(N,p):
  """Returns all valid positions for center adsorbed impurities in a BigArmStrip(N,p)
  Positions are NOT in logical order"""
  l = []
  # Short strip first 
  for j in range(0,2*N*p-2*N+1,2*N):
    for i in range(j,j+N-2,2):
      l.append(i)
  # Long strip
  for j in range(N,2*N*p-3*N+1,2*N):
    for i in range(j+1,j+N-2,2):
      l.append(i)
  return l


def CenterPositionsNew(N,p):
  """Returns all valid positions for center adsorbed impurities in a BigArmStrip(N,p)
  Positions are NOT in logical order"""
  l = []
  # Short strip first 
  ss = [i for j in range(0,2*N*p-2*N+1,2*N) for i in range(j,j+N-2,2) ]
  # Long strip
  ls = [i for j in range(N,2*N*p-3*N+1,2*N) for i in range(j+1,j+N-2,2) ]
  return ss + ls


if __name__ == "__main__":
  N = 7
  p = 2

  print CenterPositions(N,p)
  print CenterPositionsNew(N,p)
  
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
       
  #Imp_List = [0,2,4,9,20]
  #El = np.linspace(-3.0,3.0,201)
  #KlC = [KuboCenter(N,p,E,Imp_List) for E in El]
  ##Imp_List = [1]
  #KlT = [KuboTop(N,p,E,Imp_List) for E in El]
  #KlS = [KuboSubs(N,p,E,Imp_List) for E in El]
  #pl.plot(El,KlC,label='Center')
  #pl.plot(El,KlS,label='Subs')
  #pl.plot(El,KlT,label='Top')
  #pl.legend()
  #pl.show()



      
