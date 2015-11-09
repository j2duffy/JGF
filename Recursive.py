# Contains a set of routines for building recursive structures.
# Mostly focused on armchair nanoribbons
# The N numbering convention is used everywhere unless noted.
from numpy import dot
from GF import *
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


def CumAv(l):
  """Gets the cumulative average of a list (or a 1d array)"""
  return np.cumsum(l)/np.arange(1,len(l)+1)


def PadZeros(M,Msize):
  """Pads a real 2d array with zeros up to the specified size"""
  temp = np.zeros(Msize)
  temp[:M.shape[0],:M.shape[1]] = M
  return temp



def CenterPositions(N,p):
  """Lists all the possible center adsorbed positions in a nanoribbons in [cell,site] notation"""
  l = [[i,j] for i in range(p-1) for j in range(0,N-2,2) + range(N+1,2*N-2,2)]
  return l + [[p-1,j] for j in range(0,N-2,2)]


def AllPositions(N,p):
  """Gives all of the possible sites in a nanoribbon in [cell,site] notation"""
  return [[i,j] for i in range(p) for j in range(2*N)]


def ImpListConvert(p,ImpList):
  """Converts a list of impurities from [cell,site] notation to a [[imps in lead 1],[imps in lead 2]...] notation."""
  ImpListOrdered = [[] for i in range(p)]
  for i,j in ImpList:
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
  """Creates the Hamiltonian for an armchair strip populated with center adsorbed impurities.
  Allows the user to add impurities to the right side of the nanoribbon (which must be connected with appropriate potentials."""
  nimp = len(ImpList)
  H = np.zeros((2*N+nimp,2*N+nimp))
  H[:2*N,:2*N] = HArmStrip(N)
  for i,k in enumerate(ImpList):
    H[2*N+i,2*N+i] = eps_imp
    if k < N:
      for j in range(3) + range(N,N+3):
	H[2*N+i,k+j] = H[k+j,2*N+i] = tau
    if k > N:
      for j in range(3):
	H[2*N+i,k+j] = H[k+j,2*N+i] = tau
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


def VArmStripCenter(N,ImpList):
  """Calculates the LR and RL connection matrices for the armchair strip with center adsorbed impurities"""
  nimp = len(ImpList)
  VLR, VRL = np.zeros([2*N+nimp,2*N]),np.zeros([2*N,2*N+nimp])
  VLR[:2*N,:2*N], VRL[:2*N,:2*N] = VArmStrip(N)
  for i,k in enumerate(ImpList):
    if k > N:
      for j in range(3):
	VLR[2*N+i,k-N+j] = VRL[k-N+j,2*N+i] = tau
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
  gC = gGen(E,HC)	# The advanced GF
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
  gL,gR,VLR,VRL = Leads(N,E-1j*eta)	# Gets the advanced GF
  return Kubo(gL,gR,VLR,VRL)


def KuboSubs(N,E,BigImpList):
  """Calculates the conductance of a GNR with substitutional impurities using the Kubo Formula."""
  # Get Leads
  gL,gR,VLR,VRL = Leads(N,E-1j*eta)	# Gets the advanced GF
  # Build scattering region strip by strip
  for ImpList in BigImpList:
    H = HArmStripSubs(N,ImpList)
    g = gGen(E-1j*eta,H)
    gL = RecAdd(gL,g,VLR,VRL)
  return Kubo(gL,gR,VLR,VRL)


def KuboTop(N,E,BigImpList):
  """Calculates the conductance of a GNR with top-adsorbed impurities using the Kubo Formula."""
  # Get Leads
  gL,gR,VLR,VRL = Leads(N,E-1j*eta)	# Gets the advanced GF
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


def KuboCenter(N,E,BigImpList):
  """Calculates the conductance of a GNR with top-adsorbed impurities using the Kubo Formula."""
  # Leads
  gL,gR,VLR,VRL = Leads(N,E-1j*eta)	# Gets the advanced GF
  for ImpList in BigImpList + [[]]:		# Always add an extra cell.
    H = HArmStripCenter(N,ImpList)
    g = gGen(E-1j*eta,H)
    VLRr = PadZeros(VLR,(gL.shape[0],g.shape[0]))	# resized VLR
    VRLr = PadZeros(VRL,(g.shape[0],gL.shape[0]))
    
    gL = RecAdd(gL,g,VLRr,VRLr)
  
    VLR, VRL = VArmStripCenter(N,ImpList)
  return Kubo(gL,gR,VLR,VRL)



def ConfigAvSubsTotal(N,p,nimp,E):
  """Calculates the Kubo Formula for every possible case of nimp substitutional impurities in a ribbon of (N,p). Averages all cases."""
  KT = 0	# Total of all conductance measurements
  imp_pos = AllPositions(N,p)	# Generates all possible positions in [cell,site] notation
  gL,gR,VLR,VRL = Leads(N,E-1j*eta)	# Gets the advanced GF
  for i in combinations(imp_pos,nimp):	# Gets every possible combination of positions
    BigImpList = ImpListConvert(p,i)	# Conver the list to [[sites][sites]...] notation
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
  gL,gR,VLR,VRL = Leads(N,E-1j*eta)	# Gets the advanced GF
  for i in combinations(imp_pos,nimp):	# Gets every possible combination of positions
    BigImpList = ImpListConvert(p,i)	# Conver the list to [[sites][sites]...] notation
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
  KT = 0	# Total of all conductance measurements
  imp_pos = CenterPositions(N,p)	# Generates all possible positions in [cell,site] notation
  gL,gR,VLR,VRL = Leads(N,E-1j*eta)	# Gets the advanced GF
  for i in combinations(imp_pos,nimp):	# Gets every possible combination of positions
    BigImpList = ImpListConvert(p,i)	# Convert the list to [[sites][sites]...] notation
    GL = gL				# Otherwise this gets overwritten every time
    for ImpList in BigImpList + [[]]:	# Always add an extra cell.
      H = HArmStripCenter(N,ImpList)
      g = gGen(E-1j*eta,H)
      VLRr = PadZeros(VLR,(GL.shape[0],g.shape[0]))	# resized VLR
      VRLr = PadZeros(VRL,(g.shape[0],GL.shape[0]))
      
      GL = RecAdd(GL,g,VLRr,VRLr)
      VLR, VRL = VArmStripCenter(N,ImpList)
    KT += Kubo(GL,gR,VLR,VRL)
  return KT/choose(len(imp_pos),nimp)		# Choose should give the size of our list of combinations



def CASubsRandom(N,p,nimp,niter,E):
  """Calculates the configurational average for nimp substitutional impurities in an armchair nanoribbons (N,p).
  Randomly chooses niter configurations and returns a list of the results of the Kubo Formula applied in these iterations.
  Samples WITH replacement, which is not ideal"""
  gL,gR,VLR,VRL = Leads(N,E-1j*eta)	# Gets the advanced GF
  imp_pos = AllPositions(N,p)	# Generates all possible positions in [cell,site] notation

  Klist = []
  for i in range(niter):	# Repeat niter times
    ImpListMess = random.sample(imp_pos,nimp)		# Get a random sample of positions in [cell,site] notation
    BigImpList = ImpListConvert(p,ImpListMess)	# Convert the list to [[sites][sites]...] notation
    GL = gL				# Otherwise this gets overwritten every time
    for ImpList in BigImpList:
      H = HArmStripSubs(N,ImpList)
      g = gGen(E-1j*eta,H)
      GL = RecAdd(GL,g,VLR,VRL)
    K = Kubo(GL,gR,VLR,VRL)
    Klist.append(K)
  return Klist


def CATopRandom(N,p,nimp,niter,E):
  """Calculates the configurational average for nimp top-adsorbed impurities in an armchair nanoribbons (N,p).
  Randomly chooses niter configurations and returns a list of the results of the Kubo Formula applied in these iterations.
  Samples WITH replacement, which is not ideal"""
  gL,gR,VLR,VRL = Leads(N,E-1j*eta)	# Gets the advanced GF
  imp_pos = AllPositions(N,p)	# Generates all possible positions in [cell,site] notation

  Klist = []
  for i in range(niter):	# Repeat niter times
    ImpListMess = random.sample(imp_pos,nimp)		# Get a random sample of positions in [cell,site] notation
    BigImpList = ImpListConvert(p,ImpListMess)	# Convert the list to [[sites][sites]...] notation
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
    K = Kubo(GL,gR,VLR,VRL)
    Klist.append(K)
  return Klist


def CACenterRandom(N,p,nimp,niter,E):
  """Calculates the configurational average for nimp top-adsorbed impurities in an armchair nanoribbons (N,p).
  Randomly chooses niter configurations and returns a list of the results of the Kubo Formula applied in these iterations.
  Samples WITH replacement, which is not ideal"""
  gL,gR,VLR,VRL = Leads(N,E-1j*eta)	# Gets the advanced GF
  imp_pos = CenterPositions(N,p)	# Generates all possible positions in [cell,site] notation

  Klist = []
  for i in range(niter):	# Repeat niter times
    ImpListMess = random.sample(imp_pos,nimp)		# Get a random sample of positions in [cell,site] notation
    BigImpList = ImpListConvert(p,ImpListMess)	# Convert the list to [[sites][sites]...] notation
    GL = gL				# Otherwise this gets overwritten every time
    for ImpList in BigImpList + [[]]:
      H = HArmStripCenter(N,ImpList)
      g = gGen(E-1j*eta,H)
      VLRr = PadZeros(VLR,(GL.shape[0],g.shape[0]))	# resized VLR
      VRLr = PadZeros(VRL,(g.shape[0],GL.shape[0]))
      
      GL = RecAdd(GL,g,VLRr,VRLr)
      VLR, VRL = VArmStripCenter(N,ImpList)
    K = Kubo(GL,gR,VLR,VRL)
    Klist.append(K)
  return Klist


def ConcentrationPlot(N,p,E):
  """Returns the total configurational average of the conductance at E for increasing concentrations of impurities"""
  max_n = len(CenterPositions(N,p))
  nimpl = range(1,max_n+1)
  
  CAS = [ConfigAvSubsTotal(N,p,nimp,E) for nimp in nimpl]
  CAT = [ConfigAvTopTotal(N,p,nimp,E) for nimp in nimpl]
  CAC = [ConfigAvCenterTotal(N,p,nimp,E) for nimp in nimpl]
  
  conc = [nimp/(2.0*N*p) for nimp in nimpl]
  
  return conc,CAS,CAT,CAC



if __name__ == "__main__":  
  N = 5
  p = 3
  E = 0.0
  #niter = 10000
  #steps = 1
  
  #max_n = len(CenterPositions(N,p))
  #nimpl = range(1,max_n+1,steps)
  
  #CAC = [np.average(CACenterRandom(N,p,nimp,niter,E)) for nimp in nimpl]
  #CAS = [np.average(CASubsRandom(N,p,nimp,niter,E)) for nimp in nimpl]
  #CAT = [np.average(CATopRandom(N,p,nimp,niter,E)) for nimp in nimpl]
  
  #conc = [nimp/(2.0*N*p) for nimp in nimpl]
  
  conc,CAS,CAT,CAC = ConcentrationPlot(N,p,E)
  
  pl.plot(conc,CAS,label='Subs')
  pl.plot(conc,CAT,label='Top')
  pl.plot(conc,CAC,label='Center')
  pl.legend()
  np.savetxt('GCAvconc.dat',zip(conc,CAS,CAT,CAC))
  pl.savefig('GCAvconc.pdf')
  pl.show()

    


	
