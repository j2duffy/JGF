"""Auxiliary routines, mostly for matrix creation, for various systems"""

from GF import *

def Dyson(g,V):
  """Uses Dyson's formula to get the GF mx after a given perturbation."""
  return inv( np.eye(len(g)) - g.dot(V) ).dot(g)	# If V inputted as scalar, automatically multiplies by identity


def ListGF(E,r):
  """Calculates the GF from the Energy and a list.
  Really just a convenience function for all of those Center_gen codes
  Although this is probably how every GF should be done.
  Still weird that Energy comes before position."""
  m,n,s = r
  return gBulk_kZ(m,n,s,E)


def SymSector(r):
  """Puts all position coordinates in the same sector. Used for symmetries."""
  m,n,s = r
  while (m < 0 or n < 0):
    m,n,s = [m+n+s,-m,-s]		# Rotation
  if m<n:		# Reflection
    m,n=n,m
    
  return [m,n,s]


def gBulkSubsMx(m,n,s,E):
  """ Returns the GF matrix for two atomic sites in bulk graphene"""
  g = np.zeros((2,2),dtype=complex)
  g[0,0],g[1,1] = 2*(gBulk_kZ(0,0,0,E),)
  g[0,1],g[1,0] = 2*(gBulk_kZ(m,n,s,E),)
  return g


def gBulkTopMx(m,n,s,E):
  """ Returns the GF matrix for two atomic sites in bulk graphene"""
  g = np.zeros((4,4),dtype=complex)	# Should just call gBulkmx here
  g[0,0],g[1,1] = 2*(gBulk_kZ(0,0,0,E),)
  g[0,1],g[1,0] = 2*(gBulk_kZ(m,n,s,E),)
  
  #Introduce the impurity GFs
  g_impurity = 1.0/(E-eps_imp)
  g[2,2],g[3,3] = 2*(g_impurity,)
  
  # The peturbation connects the impurities to the lattice
  V = np.zeros([4,4],dtype=complex)
  V[2,0],V[0,2],V[1,3],V[3,1] = 4*(tau,)
  
  G = Dyson(g,V)
  
  # Return the part of the matrix that governs the impurity behaviour. 
  return G[2:4,2:4]


def gBulkTopMx3(r1,r2,E):
  """An attempt to do the top adsorbed for 3 impurities.
  Tested to see that it reduces to the 2 impurity case in appropriate conditions.
  No attempt has yet been made to account for symmetries"""
  r0 = np.array([0,0,0])
  r1 = np.array(r1)
  r2 = np.array(r2)
  
  r = [r0,r1,r2]
  
  g = np.zeros((6,6),dtype=complex)
  # This is the least pythonic way in which to do this. You should change it whenever you can think of something better. 
  for i in range(3):
    for j in range(3):
      g[i,j] = ListGF(E,r[j] - r[i])

  #Introduce the impurity GFs
  g_impurity = 1.0/(E-eps_imp)
  g[3,3],g[4,4],g[5,5] = 3*(g_impurity,)
  
  # The peturbation connects the impurities to the lattice
  V = np.zeros([6,6],dtype=complex)
  V[3,0],V[0,3],V[1,4],V[4,1],V[2,5],V[5,2] = 6*(tau,)
  
  G = Dyson(g,V)
  
  # Return the part of the matrix that governs the impurity behaviour. 
  return G[3:6,3:6]


def gGNRSubsMx(nE,m1,n1,m2,n2,s,E):      
  """Just returns the GF matrix for two atomic positions in a graphene GNR.
  Realistically, should be incorporated into far more functions that it is"""
  g = np.zeros((2,2),dtype=complex)
  g[0,0] = gRib_Arm(nE,m1,n1,m1,n1,0,E)
  g[1,1] = gRib_Arm(nE,m2,n2,m2,n2,0,E)
  g[0,1],g[1,0] = 2*(gRib_Arm(nE,m1,n1,m2,n2,s,E),)
  return g


def gGNRTopMx(nE,m1,n1,m2,n2,s,E):      
  """Calculates the appropriate matrix for Top Adsorbed impurities in a GNR.
  Impurities are labelled as 2 and 3. They connect to sites 0 and 1."""
  
  # Introduce the connecting GFs
  g = np.zeros((4,4),dtype=complex)
  g[0,0] = gRib_Arm(nE,m1,n1,m1,n1,0,E)
  g[1,1] = gRib_Arm(nE,m2,n2,m2,n2,0,E)
  g[0,1],g[1,0] = 2*(gRib_Arm(nE,m1,n1,m2,n2,s,E),)
  
  #Introduce the impurity GFs
  g_impurity = 1.0/(E-eps_imp)
  g[2,2],g[3,3] = 2*(g_impurity,)
  
  # The peturbation connects the impurities to the lattice
  V = np.zeros([4,4],dtype=complex)
  V[2,0],V[0,2],V[1,3],V[3,1] = 4*(tau,)
  
  G = Dyson(g,V)
  
  # Return the part of the matrix that governs the impurity behaviour. 
  return G[2:4,2:4]


def gGNRTopMxFast(nE,m1,n1,m2,n2,s,E):      
  """A faster version of gGNRTopMx. 
  Uses Dyson's formula in a more specific way to speed everything up.
  Gets a speed up of a factor of about 2."""
  
  # Introduce the connecting GFs
  gaa = gRib_Arm(nE,m1,n1,m1,n1,0,E)
  gbb = gRib_Arm(nE,m2,n2,m2,n2,0,E)
  gab,gba = 2*(gRib_Arm(nE,m1,n1,m2,n2,s,E),)
  
  #Introduce the impurity GFs
  g_impurity = 1.0/(E-eps_imp)
  gAA,gBB = 2*(g_impurity,)
  
  GAA = (gAA - gAA*gbb*gBB*t**2)/(1 - gbb*gBB*t**2 - gAA*gab*gba*gBB*t**4 + 
  gaa*gAA*t**2*(-1 + gbb*gBB*t**2))
  GBB = (gBB - gaa*gAA*gBB*t**2)/(1 - gbb*gBB*t**2 - gAA*gab*gba*gBB*t**4 + 
  gaa*gAA*t**2*(-1 + gbb*gBB*t**2))
  GAB = -((gAA*gab*gBB*t**2)/(-1 + gbb*gBB*t**2 + gAA*gab*gba*gBB*t**4 + gaa*gAA*(t**2 - gbb*gBB*t**4)))
  
  G = np.zeros((2,2),dtype=complex)
  G[0,0] = GAA
  G[1,1] = GBB
  G[0,1],G[1,0] = 2*(GAB,)
  return G


def gBulkCenterOld(m,n,E):
  """A tidied version of our old Center Code"""
  D = [m,n,0]
  hex1 = np.array([[0,0,0],[0,0,1],[1,0,0],[1,-1,1],[1,-1,0],[0,-1,1]])
  hex2 = hex1 + D
  r = np.concatenate((hex1,hex2))
  rcol = r[:,np.newaxis]
  rij = r-rcol
  
  n = len(rij)
  rijS = [[SymSector(rij[i,j]) for j in range(n)] for i in range(n)]
  
  g_dic = {}
  g_mx = np.zeros([14,14],dtype=complex)
  for i in range(n):
    for j in range(n):
      key = tuple(rijS[i][j])
      try:
	g_mx[i,j] = g_dic[key]
      except KeyError:
	g_mx[i,j] = g_dic[key] = ListGF(E,key)
  
  g_impurity = 1.0/(E-eps_imp)
  g_mx[12,12] = g_mx[13,13] = g_impurity
  
  return g_mx


def gBulkCenterMx(m,n,E):
  """Our Center Code using Dictionary Methods and the lot. I think this is pretty slick."""
  D = [m,n,0]
  hex1 = np.array([[0,0,0],[0,0,1],[1,0,0],[1,-1,1],[1,-1,0],[0,-1,1]])
  hex2 = hex1 + D
  r = np.concatenate((hex1,hex2))
  rcol = r[:,np.newaxis]
  rij = r-rcol
  
  rflat = rij.reshape(144,3)
  rSflat = map(SymSector,rflat)
  rUnique = set(map(tuple,rSflat))
  dic = {k:ListGF(E,k) for k in rUnique}
  gflat = np.array([dic[tuple(r)] for r in rSflat])
  
  g_mx = np.zeros([14,14],dtype=complex)
  g_mx[:12,:12] = gflat.reshape(12,12)
  
  g_impurity = 1.0/(E-eps_imp)
  g_mx[12,12] = g_mx[13,13] = g_impurity
  
  return g_mx


def gBulkCenterMxMulti(m,n,E):
  """Our Center Code using Dictionary Methods and the lot. A LOT of map tricks. Runs in parallel, but slowly. More of a demo than anything else."""
  D = [m,n,0]
  hex1 = np.array([[0,0,0],[0,0,1],[1,0,0],[1,-1,1],[1,-1,0],[0,-1,1]])
  hex2 = hex1 + D
  r = np.concatenate((hex1,hex2))
  rcol = r[:,np.newaxis]
  rij = r-rcol
  
  rflat = rij.reshape(144,3)
  rSflat = map(SymSector,rflat)
  rUnique = list(set(map(tuple,map(SymSector,rflat))))		# Needs to be ordered for the map
  
  g = partial(ListGF,E)
  pool = multiprocessing.Pool()
  gS = pool.map(g,rUnique)
  pool.close()
  pool.join()
   
  dic = {k:v for k,v in zip(rUnique,gS)}
  gflat = np.array([dic[tuple(r)] for r in rSflat])
  
  g_mx = np.zeros([14,14],dtype=complex)
  g_mx[:12,:12] = gflat.reshape(12,12)
  
  g_impurity = 1.0/(E-eps_imp)
  g_mx[12,12] = g_mx[13,13] = g_impurity
  
  return g_mx


def CenterMx(m,n,E):
  """A routine for calculating the 2x2 impurity matrix for Center adsorbed impurities"""
  V = np.zeros([14,14],dtype=complex)
  V[:6,12] = tau
  V[12,:6] = tau

  V[6:12,13] = tau
  V[13,6:12] = tau
  
  g = gBulkCenterMx(m,n,E)

  g_new = Dyson(g,V)

  g_impur =  np.zeros([2,2],dtype=complex)
  g_impur[0,0] = g_new[12,12]
  g_impur[0,1] = g_new[12,13]
  g_impur[1,0] = g_new[13,12]
  g_impur[1,1] = g_new[13,13]

  return g_impur


# Here we have a bunch of attempts at creating a general routine that calculates the matrices for any distribution of impurities in bulk graphene.
def CenterTest(m,n,E):
  """A testing version of our center code. Extended a great deal for generality"""
  D = [m,n,0]
  hex1 = np.array([[0,0,0],[0,0,1],[1,0,0],[1,-1,1],[1,-1,0],[0,-1,1]])
  hex2 = hex1 + D
  r = np.concatenate((hex1,hex2))
  rcol = r[:,np.newaxis]	# Realistically, from this line onward should be in your GF generator
  rij = r-rcol
  
  g_mx = np.zeros([14,14],dtype=complex)
  g_mx[:12,:12] = CenterGen2(rij,E)
  
  g_impurity = 1.0/(E-eps_imp)
  g_mx[12,12] = g_mx[13,13] = g_impurity
  return g_mx


def Top3Test(r1,r2,E):
  """Should calculate the appropriate mx for 3 Top Adsorbed impurities. VERY VERY much in the testing phase"""
  r0 = [0,0,0]
  r1 = r1
  r2 = r2
  
  r = np.array([r0,r1,r2])
  rcol = r[:,np.newaxis]
  rij = r-rcol
  
  g = np.zeros((6,6),dtype=complex)
  g[:3,:3] = CenterGen2(rij,E)

  #Introduce the impurity GFs
  g_impurity = 1.0/(E-eps_imp)
  g[3,3],g[4,4],g[5,5] = 3*(g_impurity,)
  
  # The peturbation connects the impurities to the lattice
  V = np.zeros([6,6],dtype=complex)
  V[3,0],V[0,3],V[1,4],V[4,1],V[2,5],V[5,2] = 6*(tau,)
  
  G = Dyson(g,V)
  
  # Return the part of the matrix that governs the impurity behaviour. 
  return G[3:6,3:6]


def SubsTest(r1,E):
  """Testing function that returns the GF matrix for two atomic sites in bulk graphene"""
  r0 = [0,0,0]
  r1 = r1
  r = np.array([r0,r1])
  rcol = r[:,np.newaxis]
  rij = r-rcol  
  return CenterGen(rij,E)


def CenterGen(rij,E):
  """A function that should calculate the appropriate matrix for a bunch of bulk positions. Very much in the testing phase."""
  n = len(rij)
  rijS = [[SymSector(rij[i,j]) for j in range(n)] for i in range(n)]
  
  g_dic = {}
  g_mx = np.zeros([n,n],dtype=complex)
  for i in range(n):
    for j in range(n):
      key = tuple(rijS[i][j])
      try:
	g_mx[i,j] = g_dic[key]
      except KeyError:
	g_mx[i,j] = g_dic[key] = ListGF(E,key)
  
  return g_mx


def CenterGen2(rij,E):
  """Another function that should calculate the appropriate matrix for a bunch of bulk positions. Very much in the testing phase."""
  n = len(rij)
  rflat = rij.reshape(n*n,3)
  rSflat = map(SymSector,rflat)
  rUnique = set(map(tuple,rSflat))
  dic = {k:ListGF(E,k) for k in rUnique}
  gflat = np.array([dic[tuple(r)] for r in rSflat])
  g_mx = gflat.reshape(n,n)  
  return g_mx



def gTubeSubsMx(nC,m,n,s,E):
  """ Returns the GF matrix for two atomic sites in bulk graphene"""
  g = np.zeros((2,2),dtype=complex)
  g[0,0],g[1,1] = 2*(gTube_Arm(nC,0,0,0,E),)
  g[0,1],g[1,0] = 2*(gTube_Arm(nC,m,n,s,E),)
  return g

if __name__ == "__main__":   
  print gTubeSubsMx(6,4,4,0,1.2+1j*eta)