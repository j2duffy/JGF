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


def random_combination(iterable, r):
  "Random selection from itertools.combinations(iterable, r)"
  pool = tuple(iterable)
  n = len(pool)
  indices = sorted(random.sample(xrange(n), r))
  return tuple(pool[i] for i in indices)


def cumav(l):
  """Gets the cumulative average of a list (or a 1d array)"""
  return np.cumsum(l)/np.arange(1,len(l)+1)


def CenterPositions(N,p):
  """Returns all valid positions for center adsorbed impurities in a BigArmStrip(N,p)
  Positions are NOT in logical order.
  This is not valid in your current code."""
  l = []
  # Short strip first 
  ss = [i for j in range(0,2*N*p-2*N+1,2*N) for i in range(j,j+N-2,2) ]
  # Long strip
  ls = [i for j in range(N,2*N*p-3*N+1,2*N) for i in range(j+1,j+N-2,2) ]
  return ss + ls


def AllPositions(N,p):
  """Gives all of the possible positions in a nanoribbon of width N length p."""
  return [[i,j] for i in range(p) for j in range(2*N)]


def ImpConvert(p,ImpListMess):
  """Converts a list of impurities from (lead,pos) notation to a [[imps in lead 1],[imps in lead 2]...] notation."""
  ImpListOrdered = [[] for i in range(p)]
  for i,j in ImpListMess:
    ImpListOrdered[i].append(j)
  return ImpListOrdered





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


def HArmStripSubs(N,ImpList):
  """Creates the Hamiltonian of an armchair strip with substitutional impurities."""
  H = HArmStrip(N)
  for i in ImpList:
    H[i,i] = eps_imp
  return H


def HArmStripTop(N,ImpList):
  """Creates the Hamiltonian of an armchair strip with top adsorbed impurities."""
  nimp = len(ImpList)
  H = np.zeros((2*N+nimp,2*N+nimp))
  H[:2*N,:2*N] = HArmStrip(N)
  for i,k in enumerate(ImpList):
    H[2*N+i,k] = H[k,2*N+i] = tau
    H[2*N+i,2*N+i] = eps_imp
  return H


def HArmStripCenter(N,ImpList):
  """Creates the Hamiltonian for an armchair strip populated with center adsorbed impurities"""
  nimp = len(ImpList)
  H = np.zeros((2*N+nimp,2*N+nimp))
  H[:2*N,:2*N] = HArmStrip(N)
  
  for i,k in enumerate(ImpList):
    for j in range(3) + range(N,N+3):
      H[2*N+i,k+j] = H[k+j,2*N+i] = tau
    H[2*N+i,2*N+i] = eps_imp
  return H


def HBigArmStrip(N,p):
  """Creates the Hamiltonian for an armchair strip N atoms across and p "unit cells" in width."""
  H = np.zeros((2*N*p,2*N*p))
  # nn elements
  for j in range(0,2*p*N-N+1,N):
    for i in range(j,j+N-1):
      H[i,i+1] = H[i+1,i] = t
  # Other elements
  for j in range(0,2*p*N-2*N+1,2*N):
    for i in range(j,j+N,2):
      H[i,i+N] = H[i+N,i] = t
  for j in range(N,2*p*N-3*N+1,2*N):
    for i in range(j+1,j+N,2):
      H[i,i+N] = H[i+N,i] = t
  return H


def HBigArmStripSubs(N,p,ImpList):
  """Creates the Hamiltonian for an armchair strip N atoms across and p "unit cells" in width.
  Adds an impurity with energy eps_imp at every site in ImpList."""
  H = HBigArmStrip(N,p)
  for i in ImpList:
    H[i,i] = eps_imp
  return H


def HBigArmStripTop(N,p,ImpList):
  """Creates the Hamiltonian for an armchair strip N atoms across and p "unit cells" in width.
  Populates the strip with a number of top-adsorbed impurities given in Imp_list. 
  These are added in the higher elements of the mx."""
  nimp = len(ImpList)
  H = np.zeros((2*N*p+nimp,2*N*p+nimp))
  H[:2*N*p,:2*N*p] = HBigArmStrip(N,p)
  
  for i,k in enumerate(ImpList):
    H[2*N*p+i,k] = H[k,2*N*p+i] = tau
    H[2*N*p+i,2*N*p+i] = eps_imp
  return H


def HBigArmStripCenter(N,p,ImpList):
  """Creates the Hamiltonian for an armchair strip N atoms across and p "unit cells" in width.
  Populates the strip with a number of center-adsorbed impurities given in Imp_list. 
  These are added in the higher elements of the mx."""
  nimp = len(ImpList)
  H = np.zeros((2*N*p+nimp,2*N*p+nimp))
  H[:2*N*p,:2*N*p] = HBigArmStrip(N,p)
  
  for i,k in enumerate(ImpList):
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


def VArmStripBigLSmallR(N,p):
  """Connection matrices for the RIGHT SIDE of the Big Strip to the LEFT SIDE of the regular strip."""
  VLR = np.zeros((2*N*p,2*N))
  VRL = np.zeros((2*N,2*N*p))
  for i in range(1,N,2):
    VLR[2*p*N-N+i,i] = VRL[i,2*p*N-N+i] = t
  return VLR, VRL
   
   
def VArmStripSmallLBigR(N,p):
  """Connection matrices for the LEFT SIDE of the Big Strip to the RIGHT SIDE of the regular strip."""
  VLR = np.zeros((2*N,2*N*p))
  VRL = np.zeros((2*N*p,2*N))
  for i in range(1,N,2):
    VLR[N+i,i] = VRL[i,N+i] = t
  return VLR, VRL



def RecAdd(g00,g11,V01,V10):
  """Add a cell g11 to g00 using the Dyson Formula, get a cell G11"""
  return dot(inv(np.eye(g11.shape[0])-dot(dot(g11,V10), dot(g00,V01))),g11)


def gEdge(gC,V01,V10,tol=rtol):
  """Obtain a semi-infinite lead by simple recursive addition (in the 1 direction)"""
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


def Leads(N,E):
  """Gets the semi-infinte leads for an armchair nanoribbon of width N.
  Also returns the connection matrices, because we always seem to need them."""
  HC = HArmStrip(N)
  VLR, VRL = VArmStrip(N)	
  gC = gGen(E-1j*eta,HC)	# The advanced GF
  gL = RubioSancho(gC,VRL,VLR)
  gR = RubioSancho(gC,VLR,VRL)
  return gL,gR,VLR,VRL



def Kubo(gL,gR,VLR,VRL):
  """Given left and right GF leads (advanced) calculates the Kubo Formula"""
  
  # Gets the off diagonal elements
  GRRa, GRLa, GLRa, GLLa = gOffDiagonal(gR,gL,gL,gL,gL,VLR,VRL)
  # Calculates Gtilde, the imaginary part of the advanced GF
  GRRt, GRLt, GLRt, GLLt = GRRa.imag, GRLa.imag, GLRa.imag, GLLa.imag
  return 2*np.trace( dot(dot(-GRLt,VLR),dot(GRLt,VLR)) + dot(dot(GLLt,VLR),dot(GRRt,VRL)) + dot(dot(GRRt,VRL),dot(GLLt,VLR)) - dot(dot(GLRt,VRL),dot(GLRt,VRL)) )


def KuboPristine(N,E):
  """Calculates the conductance of a pristine GNR using the Kubo Formula"""
  gL,gR,VLR,VRL = Leads(N,E)
  return Kubo(gL,gR,VLR,VRL)


def KuboSubs(N,E,BigImpList):
  """Calculates the conductance of a GNR with substitutional impurities using the Kubo Formula."""
  # Get Leads
  gL,gR,VLR,VRL = Leads(N,E)
  # Build scattering region strip by strip
  for ImpList in BigImpList:
    H = HArmStripSubs(N,ImpList)
    g = gGen(E-1j*eta,H)
    gL = RecAdd(gL,g,VLR,VRL)
  return Kubo(gL,gR,VLR,VRL)


def KuboTop(N,E,BigImpList):
  """Calculates the conductance of a GNR with top-adsorbed impurities using the Kubo Formula."""
  # Get Leads
  gL,gR,VLR,VRL = Leads(N,E)
  # Build scattering region strip by strip
  for ImpList in BigImpList:
    H = HArmStripTop(N,ImpList)
    g = gGen(E-1j*eta,H)
    VTopLR = np.zeros((gL.shape[0],g.shape[0]))
    VTopRL = np.zeros((g.shape[0],gL.shape[0]))
    VTopLR[:2*N,:2*N] = VLR
    VTopRL[:2*N,:2*N] = VRL
    gL = RecAdd(gL,g,VTopLR,VTopRL)
  gL = gL[:2*N,:2*N]		# Only need first 2N elements, V kills the rest
  return Kubo(gL,gR,VLR,VRL)


def KuboCenter(N,p,E,ImpList):
  """Calculates the conductance of a GNR with center adsorbed impurities using the Kubo Formula.
  The impurities are given in ImpList and labelled with the bottom left connecting site."""
  nimp = len(ImpList)
  # Leads 
  gL,gR,VLR,VRL = Leads(N,E)

  # Scattering region and connection matrices 
  HM = HBigArmStripCenter(N,p,ImpList)
  gM = gGen(E-1j*eta,HM)
  VbLsR, VsRbL = np.zeros((2*N*p+nimp,2*N)), np.zeros((2*N,2*N*p+nimp))
  VsLbR, VbRsL = np.zeros((2*N,2*N*p+nimp)), np.zeros((2*N*p+nimp,2*N))
  VbLsR[:2*N*p,:2*N], VsRbL[:2*N,:2*N*p] = VArmStripBigLSmallR(N,p)
  VsLbR[:2*N,:2*N*p], VbRsL[:2*N*p,:2*N] = VArmStripSmallLBigR(N,p)

  # Calculate the advanced GFs
  GR = RecAdd(gR,gM,VsRbL,VbLsR)[:2*N,:2*N]	# The new rightmost cell
  
  return Kubo(gL,GR,VLR,VRL)



def ConfigAvSubsTotal(N,p,nimp,E):
  """Calculates the Kubo Formula for every possible case of nimp substitutional impurities in a ribbon of (N,p). Averages all cases."""
  KT = 0	# Total of all conductance measurements
  imp_pos = AllPositions(N,p)	# Generates all possible positions in [cell,site] notation
  gL,gR,VLR,VRL = Leads(N,E)
  for i in combinations(imp_pos,nimp):	# Gets every possible combination of positions
    BigImpList = ImpConvert(p,i)	# Conver the list to [[sites][sites]...] notation
    GL = gL				# Otherwise this gets overwritten every time
    for ImpList in BigImpList:
      H = HArmStripSubs(N,ImpList)
      g = gGen(E-1j*eta,H)
      GL = RecAdd(GL,g,VLR,VRL)
    KT += Kubo(GL,gR,VLR,VRL)
  return KT/choose(2*N*p,nimp)		# Choose should give the size of our list of combinations


def ConfigAvTopTotal(N,p,nimp,E):
  """Calculates the Kubo Formula for every possible case of nimp substitutional impurities in a ribbon of (N,p). Averages all cases."""
  KT = 0	# Total of all conductance measurements
  imp_pos = AllPositions(N,p)	# Generates all possible positions in [cell,site] notation
  gL,gR,VLR,VRL = Leads(N,E)
  for i in combinations(imp_pos,nimp):	# Gets every possible combination of positions
    BigImpList = ImpConvert(p,i)	# Conver the list to [[sites][sites]...] notation
    GL = gL				# Otherwise this gets overwritten every time
    for ImpList in BigImpList:
      H = HArmStripTop(N,ImpList)
      g = gGen(E-1j*eta,H)
      VTopLR = np.zeros((GL.shape[0],g.shape[0]))
      VTopRL = np.zeros((g.shape[0],GL.shape[0]))
      VTopLR[:2*N,:2*N] = VLR
      VTopRL[:2*N,:2*N] = VRL
      GL = RecAdd(GL,g,VTopLR,VTopRL)
    GL = GL[:2*N,:2*N]
    KT += Kubo(GL,gR,VLR,VRL)
  return KT/choose(2*N*p,nimp)		# Choose should give the size of our list of combinations


def ConfigAvCenterTotal(N,p,nimp,E):
  """Calculates the Kubo Formula for every possible case of nimp substitutional impurities in a ribbon of (N,p). Averages all cases."""
  gL,gR,VLR,VRL = Leads(N,E)
  
  # Should have this escape clause in any case, should probably raise an exception
  if nimp > len(CenterPositions(N,p)): 
    print "Too many impurities!"
    return
  
  KT = 0
  for ImpList in combinations(CenterPositions(N,p),nimp):	# For every possible combination of positions
    # Scattering region and connection matrices 
    HM = HBigArmStripCenter(N,p,ImpList)
    gM = gGen(E-1j*eta,HM)
    VbLsR, VsRbL = np.zeros((2*N*p+nimp,2*N)), np.zeros((2*N,2*N*p+nimp))
    VsLbR, VbRsL = np.zeros((2*N,2*N*p+nimp)), np.zeros((2*N*p+nimp,2*N))
    VbLsR[:2*N*p,:2*N], VsRbL[:2*N,:2*N*p] = VArmStripBigLSmallR(N,p)
    VsLbR[:2*N,:2*N*p], VbRsL[:2*N*p,:2*N] = VArmStripSmallLBigR(N,p)

    # Calculate the advanced GFs
    GR = RecAdd(gR,gM,VsRbL,VbLsR)[:2*N,:2*N]	# The new rightmost cell

    KT += Kubo(gL,GR,VLR,VRL)
  return  KT/choose(len(CenterPositions(N,p)),nimp)		# Choose should give the size of our list of combinations



def CASubsRandom(N,p,nimp,niter,E):
  """Calculates the configurational average for nimp substitutional impurities in an armchair nanoribbons (N,p).
  Randomly chooses niter configurations and returns a list of the results of the Kubo Formula applied in these iterations.
  Samples WITH replacement, which is not ideal"""
  gL,gR,VLR,VRL = Leads(N,E)

  Klist = []
  for i in range(niter):	# For every possible combination of positions
    ImpList = random.sample(range(2*N*p),nimp)		# Get a random sample of 
    # Scattering region and connection matrices 
    HM = HBigArmStripSubs(N,p,ImpList)
    gM = gGen(E-1j*eta,HM)
    VbLsR, VsRbL = VArmStripBigLSmallR(N,p)		# Notation VbLsR means a big strip on the left connects to a small strip on the right
    VsLbR, VbRsL = VArmStripSmallLBigR(N,p)

    # Calculate the advanced GFs
    GR = RecAdd(gR,gM,VsRbL,VbLsR)[:2*N,:2*N]	# The new rightmost cell

    K = Kubo(gL,GR,VLR,VRL)
    Klist.append(K)
  return Klist
  


if __name__ == "__main__":  
  N = 5
  p = 4
  nimp = 1

  Elist = np.linspace(-3.0,3.0,201)
  Klist = [ConfigAvTopTotal(N,p,nimp,E) for E in Elist]
  pl.plot(Elist,Klist)
  pl.show()



	
