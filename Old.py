from GFMod import *
from scipy import optimize
from numpy.linalg import norm, det
from functools import partial
import profile
import multiprocessing


global mag_m, band_shift, Vup, Vdown		# Do mag_m and band_shift still need to be here?
# Self-consistency crap
mag_m = 0.9
band_shift = 0.0
ex_split = U*mag_m
hw0 = 1e-3
Vdown = band_shift + (ex_split + hw0)/2.0
Vup = band_shift - (ex_split + hw0)/2.0


def Dyson(g,V):
  """Returns the new Green's function (mx) after a given peturbation"""
  temp1 = inv( np.eye(len(g)) - g.dot(V) )	# If V is inputted as a scalar, automatically multiplies by 2x2 identity
  G_new =  temp1.dot(g) 
  return G_new


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


def ListGF(E,r):
  """Calculates the GF from the Energy and a list.
  Really just a convenience function for all of those Center_gen codes
  Although this is probably how every GF should be done. """
  m = r[0]
  n = r[1]
  s = r[2]
  return gBulk_kZ(m,n,s,E)


def Center_gen_multi(m,n,E):
  D = [m,n,0]
  hex1 = np.array([[0,0,0],[0,0,1],[1,0,0],[1,-1,1],[1,-1,0],[0,-1,1]])
  hex2 = hex1 + D
  atom_pos = np.concatenate((hex1,hex2),axis=0)	# hex1+D
  atom_pos_col = atom_pos[:,np.newaxis]
  
  Vec_mx = np.zeros([12,12,3],dtype=int)
  Vec_mx = atom_pos-atom_pos_col
  
  g_mx = np.zeros([14,14],dtype=complex)
  g = partial(ListGF,E)
  
  pool = multiprocessing.Pool()
  g_mx[:12,:12] = np.array([ pool.map(g,item) for item in Vec_mx ])
  pool.close()
  pool.join()
  
  g_impurity = 1.0/(E-eps_imp)	# 1 line
  g_mx[12,12] = g_impurity
  g_mx[13,13] = g_impurity
  
  return g_mx

    
def Rot(r):	# Put in SymSector
  m = r[0]
  n = r[1]
  s = r[2]
  
  return [m+n+s,-m,-s]

    
def SymSector(r):
  while (r[0] < 0 or r[1] < 0):
    r = Rot(r)		# Rotation
  if r[0]<r[1]:		# Reflection
    r[0],r[1]=r[1],r[0]
    
  return r
  
  
def Center_gen_dic(m,n,E):
  D = [m,n,0]
  hex1 = np.array([[0,0,0],[0,0,1],[1,0,0],[1,-1,1],[1,-1,0],[0,-1,1]])
  hex2 = hex1 + D
  atom_pos = np.concatenate((hex1,hex2),axis=0)
  atom_pos_col = atom_pos[:,np.newaxis]
  
  Vec_mx = np.zeros([12,12,3],dtype=int)
  Vec_mx = atom_pos-atom_pos_col
  
  Vec_mx = np.array([ map(SymSector,item) for item in Vec_mx ])
  
  g_dic = {}
  g_mx = np.zeros([14,14],dtype=complex)
  for i, elem in enumerate( Vec_mx ):
    for j, r in enumerate( elem ):
      key = (r[0],r[1],r[2])
      try:
	g_mx[i,j] = g_dic[key]
      except KeyError:
	g_mx[i,j] = g_dic[key] = ListGF(E,r)
  
  g_impurity = 1.0/(E-eps_imp)
  g_mx[12,12] = g_impurity
  g_mx[13,13] = g_impurity
  
  return g_mx


def Center_paul_dic(m,n,E):
  D = [m,n,0]
  hex1 = np.array([[0,0,0],[0,0,1],[1,0,0],[1,-1,1],[1,-1,0],[0,-1,1]])
  hex2 = hex1 + D
  atom_pos = np.concatenate((hex1,hex2),axis=0)
  atom_pos_col = atom_pos[:,np.newaxis]
  
  Vec_mx = np.zeros([12,12,3],dtype=int)
  Vec_mx = atom_pos-atom_pos_col
  Vec_mx = np.array([ map(SymSector,item) for item in Vec_mx ])
  

  v_dic = {}
  mx_dic = {}
  for xi,x in enumerate( Vec_mx ):
    for yj,y in enumerate (x):
      mx_dic[ ( xi,yj ) ] = (y[0],y[1],y[2])
      v_dic[ (y[0],y[1],y[2]) ] = 0
        
  g = partial(ListGF,E)
  result = np.zeros([len(v_dic)],dtype=complex)
  
  
  pool = multiprocessing.Pool()
  result = pool.map(g,v_dic.keys())
  pool.close()
  pool.join()
    
  
  for x,y in zip(result, v_dic.keys()):
    v_dic[y] = x

  g_mx = np.zeros([14,14],dtype=complex)
  for x in mx_dic.keys():
    g_mx[ x[0], x[1] ] = v_dic[mx_dic[x]]
  

  g_impurity = 1.0/(E-eps_imp)
  g_mx[12,12] = g_impurity
  g_mx[13,13] = g_impurity

  return g_mx


def CenterMx(m,n,E):
  """A routine for calculating the 2x2 impurity matrix for Center adsorbed impurities
  Still very much in its beta phase"""
  V = np.zeros([14,14],dtype=complex)
  V[:6,12] = tau
  V[12,:6] = tau

  V[6:12,13] = tau
  V[13,6:12] = tau
  
  g = Center_gen_dic(m,n,E)

  g_new = Dyson(g,V)

  g_impur =  np.zeros([2,2],dtype=complex)
  g_impur[0,0] = g_new[12,12]
  g_impur[0,1] = g_new[12,13]
  g_impur[1,0] = g_new[13,12]
  g_impur[1,1] = g_new[13,13]

  return g_impur


def JBulkSubs(m,n,s):
  """ The bog standard coupling calculation in bulk graphene."""
  def GF(y):
    return gBulkSubsMx(m,n,s,EF+1j*y)
  def integrand(y):
    return 1.0/pi*log( abs(1.0 + ex_split**2 * Dyson(GF(y),Vup)[1,0] * Dyson(GF(y),Vdown)[0,1])  ).real 
  C = quad(integrand, eta, np.inf, epsabs=0.0e0, epsrel=1.0e-4, limit=200 )
  return C[0]


def JBulkTop(m,n,s):
  """Calculating the Top Adsorbed coupling in bulk graphene. 
  Really, not so different from the substitutional case that it should require its own code."""
  def GF(y):
    return gBulkTopMx(m,n,s,EF+1j*y)
  def integrand(y):
    return 1.0/pi*log( abs(1.0 + ex_split**2 * Dyson(GF(y),Vup)[1,0] * Dyson(GF(y),Vdown)[0,1])  ).real
  C = quad(integrand, eta, np.inf, epsabs=0.0e0, epsrel=1.0e-4, limit=200 )
  return C[0]


def JBulkCenter(m,n):
  """ The bog standard coupling calculation in bulk graphene.
    Done in Python because why not?"""
  def GF(y):
    return CenterMx(m,n,EF+1j*y)
  def integrand(y):
    return 1.0/pi*log( abs(1.0 + ex_split**2 * Dyson(GF(y),Vup)[1,0] * Dyson(GF(y),Vdown)[0,1])  ).real	#Rather ugly way of doing this
  C = quad(integrand, eta, np.inf, epsabs=0.0, epsrel=1.0e-2, limit=200 )
  return C[0]


def JGNRSubs(nE,m1,n1,m2,n2,s):
  """A routine for calculating the coupling in GNRs.
  Has been tested against recursive for bb"""
  def GF(y):
    return gGNRSubsMx(nE,m1,n1,m2,n2,s,EF+1j*y)
  def integrand(y):
    return 1.0/pi*log( abs(1.0 + ex_split**2 * Dyson(GF(y),Vup)[1,0] * Dyson(GF(y),Vdown)[0,1])  ).real	#Ugly, etc.
  C = quad(integrand, eta, np.inf, epsabs=0.0e0, epsrel=1.0e-4, limit=200 ) 
  return C[0]


def JGNRTop(nE,m1,n1,m2,n2,s):
  """A routine for calculating the coupling for Top Adsorbed Impurities in GNRs.
  Is very untested, and relies on an untested subroutine"""
  def GF(y):
    return gGNRTopMx(nE,m1,n1,m2,n2,s,EF+1j*y)
  def integrand(y):
    return 1.0/pi*log( abs(1.0 + ex_split**2 * Dyson(GF(y),Vup)[1,0] * Dyson(GF(y),Vdown)[0,1])  ).real
  C = quad(integrand, eta, np.inf, epsabs=0.0e0, epsrel=1.0e-4, limit=200 ) 
  return C[0]
  
  
def Line_Coupling(DA,s,a=1.0):
  """Finds the coupling between two lines, hopefully"""
  def g_mx(ky,E):
    g_temp = np.zeros((2,2),dtype=complex)
    g_temp[0,0] = gLine_ky(0,ky,0,E,a=a)
    g_temp[1,1] = g_temp[0,0]
    g_temp[0,1] = gLine_ky(DA,ky,s,E,a=a)
    g_temp[1,0] = g_temp[0,1]
    return g_temp
  
  def int_ky(ky,y):
    g = g_mx(ky,EF+1j*y)
    return a/(4*pi**2)*log( abs(1.0 + ex_split**2 * Dyson(g,Vup)[1,0] * Dyson(g,Vdown)[0,1])  ).real
  
  def int_E(y):
    return quad(int_ky, -2*pi/a, 2*pi/a, args=(y,), epsabs=0.0, epsrel=1.0e-2, limit=200 )[0]

  return quad(int_E, eta, np.inf, epsabs=0.0, epsrel=1.0e-2, limit=200 )[0]


def Line_Coupling2(DA,s,a=1.0):
  """Finds the coupling between two lines, hopefully"""
  def g_mx(ky,E):
    g_temp = np.zeros((2,2),dtype=complex)
    g_temp[0,0] = gLine_ky(0,ky,0,E,a=a)
    g_temp[1,1] = g_temp[0,0]
    g_temp[0,1] = gLine_ky(DA,ky,s,E,a=a)
    g_temp[1,0] = g_temp[0,1]
    return g_temp
  
  def int_E(y,ky):
    g = g_mx(ky,EF+1j*y)
    return a/(4*pi**2)*log( abs(1.0 + ex_split**2 * Dyson(g,Vup)[1,0] * Dyson(g,Vdown)[0,1])  ).real
  
  def int_ky(ky):
    return quad(int_E, eta, np.inf, args=(ky,), epsabs=0.0, epsrel=1.0e-2, limit=200 )[0]

  return quad(int_ky, -2*pi/a, 2*pi/a, epsabs=0.0, epsrel=1.0e-2, limit=200 )[0]


def Line_Coupling3(DA,s):
  """A much more efficient code for finding the coupling between two lines"""
  def g_mx(kZ,E):
    g_temp = np.zeros((2,2),dtype=complex)
    g_temp[0,0] = gLine_kZ(0,kZ,0,E)
    g_temp[1,1] = g_temp[0,0]
    g_temp[0,1] = gLine_kZ(DA,kZ,s,E)
    g_temp[1,0] = g_temp[0,1]
    return g_temp
  
  def int_E(y,kZ):
    g = g_mx(kZ,EF+1j*y)
    return 2.0/(pi**2)*log( abs(1.0 + ex_split**2 * Dyson(g,Vup)[1,0] * Dyson(g,Vdown)[0,1])  ).real
  
  def int_kZ(kZ):
    return quad(int_E, eta, np.inf, args=(kZ,), epsabs=0.0, epsrel=1.0e-2, limit=200 )[0]

  return quad(int_kZ, 0, pi/2.0, points=(pi/3.0,), epsabs=0.0, epsrel=1.0e-2, limit=200 )[0]	# YOU HAVE NO JUSTIFICATION FOR THIS POINT, AND THIS DOESN'T WORK AWAY FROM EF=0
  
  
def Line_CouplingRKKY(DA,s):
  """The RKKY approximation, for a pair of lines. 
    IS THIS EVEN THE RKKY APPROXIMATION ANYMORE?"""
  def int_E(y,kZ):
    return (gLine_kZ(DA,kZ,s,EF+1j*y)**2).real		# OBVIOUSLY THE PREFACTOR SHOULD ALWAYS BE TAKEN OUTSIDE THE INTEGRATION TO SAVE TIME
  
  def int_kZ(kZ):
    return quad(int_E, eta, np.inf, args=(kZ,), epsabs=0.0, epsrel=1.0e-2, limit=200 )[0]

  return ex_split**2/(pi**2)*quad(int_kZ, -pi/2.0, pi/2.0, epsabs=0.0, epsrel=1.0e-2, limit=200 )[0]
  
  
def Line_CouplingSPA(DA):
  """The coupling between lines performed with the SPA approximation. Only works for bb
  Also, there is absolutely no reason why this should work. Integrating something that is discontinous on the imaginary axis over the imaginary axis.
  Jesus."""

  def Line_SPA(E):
    """The appropriate SPA approximation for the Line of Impurities
    The current version of the code only works for bb, but really, do we need anything else?"""
    sig = copysign(1,-E.real)
    temp1 = -E**2 *exp(sig*2*1j*abs(DA)*acos((E**2-5.0*t**2)/4.0))/(-E**4+10*E**2 *t**2-9*t**4)**(3.0/4.0)
    temp2 = sqrt( -sig*1j*pi/(abs(DA)*(E**2+3.0*t**2)) )
    ga = temp1*temp2
    
    temp1 = -E**2 *exp(sig*2*1j*abs(DA)*acos(-sqrt(1.0-E**2/t**2)))/(4.0*(E**2 *t**2-E**4)**(3.0/4.0))
    temp2 = sqrt(sig*1j*pi/(abs(DA)*(E**2+3.0*t**2)))
    gb = temp1*temp2
    return ga+2*gb
  
  def integrand(y):
    return ex_split**2/pi**2 *Line_SPA(EF+1j*y).real

  return quad(integrand, eta, np.inf, epsabs=0.0, epsrel=1.0e-2, limit=200 )[0]
  """Gets the RPA spin susceptibility for graphene"""
  X00 = np.zeros((2,2),dtype=complex)
  X00[0,0] = X00HF_GNR(nE,m1,n1,w)
  X00[1,1] = X00HF_GNR(nE,m2,n2,w)
  X00[0,1],X00[1,0] = 2*(XHF_GNR(nE,m1,n1,m2,n2,s,w),)
  
  temp = inv( np.eye(len(X00)) + X00.dot(U) )
  return temp.dot(X00)[0,1]


def JGNRSubstest(nE,m1,n1,m2,n2,s):
  """A routine for calculating the coupling in GNRs, modified to work with matrices.
  Unsurprisingly, takes a bit longer than the usual method."""
  def GF(y):
    return gGNRSubsMx(nE,m1,n1,m2,n2,s,EF+1j*y)
  def integrand(y):
    VRot = np.array([[-2*Vup,0],[0,-2*Vdown]])
    g = np.array([[Dyson(GF(y),Vup)[1,1],0],[0,Dyson(GF(y),Vdown)[1,1]]])
    log(det(np.eye(2)-g.dot(VRot)))
    return 1.0/pi*log(det(np.eye(2)-g.dot(VRot))).real
  C = quad(integrand, eta, np.inf, epsabs=0.0e0, epsrel=1.0e-4, limit=200 ) 
  return C[0]


def JGNRSubstest(nE,m1,n1,m2,n2,s):
  """A routine for calculating the coupling in GNRs, modified to work with matrices.
  Unsurprisingly, takes a bit longer than the usual method."""
  def GF(y):
    return gGNRSubsMx(nE,m1,n1,m2,n2,s,EF+1j*y)
  def integrand(y):
    VRot = np.array([[-2*Vup,0],[0,-2*Vdown]])
    g = np.array([[Dyson(GF(y),Vup)[1,1],0],[0,Dyson(GF(y),Vdown)[1,1]]])
    return 1.0/pi*log(det(np.eye(2)-g.dot(VRot))).real
  C = quad(integrand, eta, np.inf, epsabs=0.0e0, epsrel=1.0e-4, limit=200 ) 
  return C[0]


def JBulkSubstest(m,n,s):
  """A routine for calculating the coupling in Bulk, modified to work with matrices.
  Unsurprisingly, takes a bit longer than the usual method."""
  def GF(y):
    return gBulkSubsMx(m,n,s,EF+1j*y)
  def integrand(y):
    VRot = np.array([[-2*Vup,0],[0,-2*Vdown]])
    g = np.array([[Dyson(GF(y),Vup)[1,1],0],[0,Dyson(GF(y),Vdown)[1,1]]])
    return 1.0/pi*log(det(np.eye(2)-g.dot(VRot))).real
  C = quad(integrand, eta, np.inf, epsabs=0.0e0, epsrel=1.0e-4, limit=200 ) 
  return C[0]


def JBulkTop3(r1,r2):
  """A routine for calculating the Bulk Coupling for top adsorbed impurities with 3 atoms. 
  Been tested a little. Acutally seems alright."""
  def GF(y):
    return gBulkTopMx3(r1,r2,EF+1j*y)
  def integrand(y):
    VRot = np.array([[-2*Vup,0],[0,-2*Vdown]])
    g = np.array([[Dyson(GF(y),Vup)[2,2],0],[0,Dyson(GF(y),Vdown)[2,2]]])
    return 1.0/pi*log(det(np.eye(2)-g.dot(VRot))).real
  C = quad(integrand, eta, np.inf, epsabs=0.0e0, epsrel=1.0e-4, limit=200 ) 
  return C[0]



if __name__ == "__main__":
  m,n,E = -4,10,1.3+1j*eta
  print Center_gen_dic(m,n,E)