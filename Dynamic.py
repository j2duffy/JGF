# A file for just running Dynamic calculations and nothing else.
# Should be the most up to date thing for Dynamic calculations, and the 
from GF import *
from scipy import optimize
from numpy.linalg import norm
from functools import partial
import profile
import multiprocessing
from functionsample import sample_function


# Self-consistency crap
hw0 = 1.0e-3	# A default value for the Zeeman field. Dodgy.


def Dyson1(g,V):
  return g/(1.0-g*V)


def Dyson(g,V):
  """Returns the new Green's function (mx) after a given peturbation"""
  temp1 = inv( np.eye(len(g)) - g.dot(V) )	# If V is inputted as a scalar, automatically multiplies by 2x2 identity
  G_new =  temp1.dot(g) 
  return G_new


def gRib_Armr(nE,r0,r1,E):
  """A temporary function that should be used to convert between old and new notation"""
  m1,n1,s1=r0
  m2,n2,s2=r1
  s=s2-s1
  return gRib_Arm(nE,m1,n1,m2,n2,s,E)


def gRib_mx2(nE,m1,n1,m2,n2,s,E):      
  """Just returns the GF matrix for two atomic positions in a graphene GNR.
  Realistically, should be incorporated into far more functions that it is"""
  g = np.zeros((2,2),dtype=complex)
  g[0,0] = gRib_Arm(nE,m1,n1,m1,n1,0,E)
  g[1,1] = gRib_Arm(nE,m2,n2,m2,n2,0,E)
  g[0,1],g[1,0] = 2*(gRib_Arm(nE,m1,n1,m2,n2,s,E),)		# This can also be done with a = b = f(x)
  return g


def gSImx2(m1,n1,m2,n2,s,E):
  """The appropriate mx for the SI GF. Includes symmetries in a rather ad hoc way"""
  g = np.zeros((2,2),dtype=complex)
  g[0,0] = gSI_kZ(m1,n1,m1,n1,0,E)
  g[1,1]= gSI_kZ(m2,n2,m2,n2,0,E)
  g[0,1] = gSI_kZ(m1,n1,m2,n2,s,E)
  g[1,0] = g[0,1]
  return g


def gRib_mx3(nE,r0,r1,r2,E):      
  """Just returns the GF matrix for three atomic positions in a graphene GNR.
    Tested probably as much as you'd want to."""
  g = np.zeros((3,3),dtype=complex)
  g[0,0] = gRib_Armr(nE,r0,r0,E)
  g[1,1] = gRib_Armr(nE,r1,r1,E)
  g[2,2] = gRib_Armr(nE,r2,r2,E)
  g[0,1] = g[1,0] = gRib_Armr(nE,r0,r1,E)
  g[0,2] = g[2,0] = gRib_Armr(nE,r0,r2,E)
  g[1,2] = g[2,1] = gRib_Armr(nE,r1,r2,E)
  return g


def gRib_mxn(nE,r,E):      
  """The GF matrix for n substitutional impurities. Does not account for symmetries at all"""
  # r is a list of all the relevant position vectors (so a list of lists)
  n = len(r)
  g = np.array([[gRib_Armr(nE,r[i],r[j],E) for j in range(n)] for i in range(n)])
  return g


def gGNRTop1(nE,m,n,E):
  """Returns the GF for the Top Adsorbed impurity in a GNR
  Suffers from a difference of convention with most of your code, where the connecting atoms are labelled first and the impurities last.
  However, this may be a more logical way of doing it (you are more interested in the impurity positions."""
  g00 = 1.0/(E-eps_imp)
  g11 = gRib_Arm(nE,m,n,m,n,0,E)
  G00 = g00/(1.0-tau**2 *g00*g11)
  return G00


def gGNRTopMx2(nE,m1,n1,m2,n2,s,E):      
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


def gGNRTopMxn(nE,r,E): 
  """Calculates the appropriate matrix for an arbitrary number of Top Adsorbed impurities in a GNR.
  Returns the impurity Matrix"""
  
  n = len(r)
  # Creates the Mx of sites we connect to
  gPosMx = gRib_mxn(nE,r,E)
  
  # Createst the Mx of impurities
  g_impurity = 1.0/(E-eps_imp)
  gImpMx = g_impurity*np.eye(n)
  
  # Combination Mx
  gbig = np.zeros([2*n,2*n],dtype=complex)
  gbig[:n,:n] = gPosMx
  gbig[n:,n:] = gImpMx
  
  # Connection Potential
  V = np.zeros([2*n,2*n])
  for i in range(n):
    V[n+i,i] = V[i,n+i] = tau
  
  G = Dyson(gbig,V)
  
  # Return the part of the matrix that governs the impurity behaviour. 
  return G[n:,n:]


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


def gBulk_mx2(m,n,s,E):
  """ Returns the GF matrix for two atomic sites in bulk graphene"""
  g = np.zeros((2,2),dtype=complex)
  g[0,0],g[1,1] = 2*(gBulk_kZ(0,0,0,E),)
  g[0,1],g[1,0] = 2*(gBulk_kZ(m,n,s,E),)
  return g


def ReducedGF(E,L):
  """Calculates the GF from the Energy and a list.
  Really just a convenience function for all of those Center_gen codes"""
  m = L[0]
  n = L[1]
  if L[2] == 0:
    s = 0
  elif L[2] == 1:
    s = 1
  elif L[2] == -1:
    s = -1
  else:
    print 'ReducedGF Error'

  return gBulk_kZ(m,n,s,E)


def Center_gen_multi(m,n,E):	# This is way too fast. Change to only one core and make sure it's slow.
  D = [m,n,0]
  hex1 = np.array([[0,0,0],[0,0,1],[1,0,0],[1,-1,1],[1,-1,0],[0,-1,1]])
  hex2 = hex1 + D
  atom_pos = np.concatenate((hex1,hex2),axis=0)	# hex1+D
  atom_pos_col = atom_pos[:,np.newaxis]
  
  Vec_mx = np.zeros([12,12,3],dtype=int)
  Vec_mx = atom_pos-atom_pos_col
  
  g_mx = np.zeros([14,14],dtype=complex)
  g = partial(ReducedGF,E)
  
  pool = multiprocessing.Pool()
  g_mx[:12,:12] = np.array([ pool.map(g,item) for item in Vec_mx ])
  pool.close()
  pool.join()
  
  g_impurity = 1.0/(E-eps_imp)	# 1 line
  g_mx[12,12] = g_impurity
  g_mx[13,13] = g_impurity
  
  return g_mx

    
def Rot(L):
  m = L[0]
  n = L[1]
  s = L[2]
  
  return [m+n+s,-m,-s]

    
def irr_sector(L):	# Again, best name?
  while (L[0] < 0 or L[1] < 0):
    L = Rot(L)		# Rotation
    
  if L[0]<L[1]:		# Reflection
    L[0],L[1]=L[1],L[0]
    
  return L
  
  
def Center_gen_dic(m,n,E):
  D = [m,n,0]
  hex1 = np.array([[0,0,0],[0,0,1],[1,0,0],[1,-1,1],[1,-1,0],[0,-1,1]])
  hex2 = hex1 + D
  atom_pos = np.concatenate((hex1,hex2),axis=0)
  atom_pos_col = atom_pos[:,np.newaxis]
  
  Vec_mx = np.zeros([12,12,3],dtype=int)
  Vec_mx = atom_pos-atom_pos_col
  
  Vec_mx = np.array([ map(irr_sector,item) for item in Vec_mx ])
  
  g_dic = {}
  g_mx = np.zeros([14,14],dtype=complex)
  for i, elem in enumerate( Vec_mx ):
    for j, L in enumerate( elem ):
      key = (L[0],L[1],L[2])
      try:
	g_mx[i,j] = g_dic[key]
      except KeyError:
	g_mx[i,j] = g_dic[key] = ReducedGF(E,L)
  
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
  Vec_mx = np.array([ map(irr_sector,item) for item in Vec_mx ])
  

  v_dic = {}
  mx_dic = {}
  for xi,x in enumerate( Vec_mx ):
    for yj,y in enumerate (x):
      mx_dic[ ( xi,yj ) ] = (y[0],y[1],y[2])
      v_dic[ (y[0],y[1],y[2]) ] = 0
        
  g = partial(ReducedGF,E)
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



def XHF1(GF,Vup,Vdown,w):
  """Calculates the Hartree-Fock spin susceptibility"""
  def spin_sus_int12(y):
    return hbar/(2.0*pi) *( GF(wf + 1j*y,Vup)*GF(wf + w + 1j*y,Vdown) + GF(wf - 1j*y,Vdown)*GF(wf-w- 1j*y,Vup) )

  def spin_sus_int3(w_dum):
    return - 1j*hbar/(2.0*pi) *GF(w_dum - 1j*eta,Vup)*GF(w + w_dum + 1j*eta,Vdown)
    
  I12 = C_int(spin_sus_int12,eta,np.inf)
  I3 = C_int(spin_sus_int3,wf-w,wf)
    
  return I12 + I3


def XHFBulk1(Vup,Vdown,w):
  """Calculates the Hartree-Fock spin susceptibility in Bulk Graphene"""
  def GF(E,V):
    g = gBulk_kZ(0,0,0,E)
    return Dyson1(g,V)
  return XHF1(GF,Vup,Vdown,w)


def XHF_SI1(m,n,Vup,Vdown,w):
  """Calculates the on-site Hartree-Fock spin susceptibility in Semi-Infinite Graphene."""
  def GF(E,V):
    g = gSI_kZ(m,n,m,n,0,E)
    return Dyson1(g,V)
  return XHF1(GF,Vup,Vdown,w)


def XHF_GNR1(nE,m,n,Vup,Vdown,w):
  """Calculates the Hartree-Fock spin susceptibility in a GNR."""
  def GF(E,V):
    g = gRib_Arm(nE,m,n,m,n,0,E)
    return Dyson1(g,V)
  return XHF1(GF,Vup,Vdown,w)


def XHF_GNRTop1(nE,m,n,Vup,Vdown,w):
  """Calculates the Hartree-Fock spin susceptibility for a Top adsorbed impurity in a GNR."""
  def GF(E,V):
    g = gGNRTop1(nE,m,n,E)
    return Dyson1(g,V)
  return XHF1(GF,Vup,Vdown,w)


def XRPABulk1(Vup,Vdown,w):
  """Calculates the on-site spin susceptibility in the RPA approximation"""
  X0 = XHFBulk1(Vup,Vdown,w)
  return X0/(1.0+U*X0)


def XRPA_GNR1(nE,m,n,Vup,Vdown,w):
  """Calculates the on-site spin susceptibility in the RPA approximation"""
  X0 = XHF_GNR1(nE,m,n,Vup,Vdown,w)
  return X0/(1.0+U*X0)


def XRPA_GNRTop1(nE,m,n,Vup,Vdown,w):
  """Calculates the on-site spin susceptibility in the RPA approximation"""
  X0 = XHF_GNRTop1(nE,m,n,Vup,Vdown,w)
  return X0/(1.0+U*X0)



def XHF(GF,site,Vup,Vdown,w):	# Might be better to include the Dyson function within the calling functions
  """Calculates the Hartree-Fock spin susceptibility"""
  i,j = site
  def spin_sus_int12(y):
    X = hbar/(2.0*pi) *( Dyson(GF(wf + 1j*y),Vup)[j,i]*Dyson(GF(wf + w + 1j*y),Vdown)[i,j] + \
      Dyson(GF(wf - 1j*y),Vdown)[i,j]*Dyson(GF(wf - w - 1j*y),Vup)[j,i] ) # Why the fuck Dyson function?
    return X
  
  def spin_sus_int3(w_dum):
    return - 1j*hbar/(2.0*pi) *Dyson(GF(w_dum - 1j*eta),Vup)[j,i]*Dyson(GF(w + w_dum + 1j*eta),Vdown)[i,j]
  
  I12 = C_int(spin_sus_int12,eta,np.inf)
  I3 = C_int(spin_sus_int3,wf-w,wf)
    
  return I12 + I3


def XHFBulk2(m,n,s,site,Vup,Vdown,w):
  """Calculates the HF spin susceptibility for two substitutional impurities in bulk graphene"""
  def GF(E):
    return gBulk_mx2(m,n,s,E)	# Could maybe include the site term here and save a fucking bunch of time  
  return XHF(GF,site,Vup,Vdown,w)


def XHF_SI2(m1,n1,m2,n2,s,site,Vup,Vdown,w):
  """Calculates the spin susceptibility in the HF approximation for Semi-Infinite Graphene, 
    Requires testing"""
  def GF(E):
    return gSImx2(m1,n1,m2,n2,s,E)
    
  return XHF(GF,site,Vup,Vdown,w)


def XHF_GNR2(nE,m1,n1,m2,n2,s,site,Vup,Vdown,w):
  """Calculates the spin susceptibility in the HF approximation for a GNR"""
  def GF(E):
    g = gRib_mx2(nE,m1,n1,m2,n2,s,E)
    return g  
  return XHF(GF,site,Vup,Vdown,w)


def XGNR_HFTop2(nE,m1,n1,m2,n2,s,site,Vup,Vdown,w):
  """Calculates the spin susceptibility in the HF approximation for Top Adsorbed impurities in a GNR"""
  def GF(E):
    g = gGNRTopMx2(nE,m1,n1,m2,n2,s,E)
    return g   
  return XHF(GF,site,Vup,Vdown,w)


def XRPABulk(m,n,s,Vup,Vdown,w):
  """The RPA susceptibility for Bulk graphene. Has not been tested really at all."""
  n = 2
  X00 = np.array([[XHFBulk2(m,n,s,[i,j],Vup,Vdown,w) for j in range(n)] for i in range(n)])
  temp = inv( np.eye(len(X00)) + X00.dot(U) )
  return temp.dot(X00)


def XRPA_GNR2(nE,m1,n1,m2,n2,s,Vup,Vdown,w):
  """Gets the RPA spin susceptibility for a GNR for 2 atoms.
  Done in a fairly shlick way, to make extensions to more atoms easier."""
  n = 2
  X00 = np.array([[XHF_GNR2(nE,m1,n1,m2,n2,s,[i,j],Vup,Vdown,w) for j in range(n)] for i in range(n)])
  temp = inv( np.eye(len(X00)) + X00.dot(U) )
  return temp.dot(X00)


def XGNR_RPATop2(nE,m1,n1,m2,n2,s,Vup,Vdown,w):
  """Gets the RPA spin susceptibility for a GNR"""
  n = 2
  X00 = np.array([[XGNR_HFTop2(nE,m1,n1,m2,n2,s,[i,j],Vup,Vdown,w) for j in range(n)] for i in range(n)])
  temp = inv( np.eye(len(X00)) + X00.dot(U) )
  return temp.dot(X00)


def XHFGNR3(nE,r0,r1,r2,site,Vup,Vdown,w):
  """The HF spin sus for 3 atoms in a GNR. 
  Tested a little. Also has a much better name."""
  i,j = site
  def GF(E):
    return gRib_mx3(nE,r0,r1,r2,E)
    
  return XHF(GF,site,Vup,Vdown,w)


def XRPAGNR3(nE,r0,r1,r2,Vup,Vdown,w):
  """Gets the RPA spin susceptibility for a GNR for 3 atoms.
    Should be easy to extend to further dimensions. 
    There's also a bunch of obvious symmetries here that we're not exploiting (since the output mx is symmetric (NOT HERMITIAN))"""
  n = 3
  X00 = np.array([[XHFGNR3(nE,r0,r1,r2,[i,j],Vup,Vdown,w) for j in range(n)] for i in range(n)]) 		# Rather slick piece of work here mr. duffy. Apply elsewhere
  temp = inv( np.eye(len(X00)) + X00.dot(U) )
  return temp.dot(X00)


def XHFGNRn(nE,r,site,Vup,Vdown,w):
  """The HF spin sus for n substitutional atoms in a GNR."""
  def GF(E):
    return gRib_mxn(nE,r,E)
  return XHF(GF,site,Vup,Vdown,w)


def XRPAGNRn(nE,r,Vup,Vdown,w):
  """Gets the RPA spin susceptibility for a GNR for n substitutional impurities.
    There are a lot of symmetries not being exploited here (the matrix is symmetric)"""
  n = len(r)
  X00 = np.array([[XHFGNRn(nE,r,[i,j],Vup,Vdown,w) for j in range(n)] for i in range(n)]) 		# Rather slick piece of work here Mr. duffy. Apply elsewhere
  temp = inv( np.eye(len(X00)) + X00.dot(U) )
  return temp.dot(X00)


def XHFGNRTopn(nE,r,site,Vup,Vdown,w):
  """The HF spin sus for n substitutional atoms in a GNR."""
  def GF(E):
    return gGNRTopMxn(nE,r,E)
  return XHF(GF,site,Vup,Vdown,w)


def XRPAGNRTopn(nE,r,Vup,Vdown,w):
  """Gets the RPA spin susceptibility for a GNR for n substitutional impurities.
    There are a lot of symmetries not being exploited here (the matrix is symmetric)"""
  n = len(r)
  X00 = np.array([[XHFGNRTopn(nE,r,[i,j],Vup,Vdown,w) for j in range(n)] for i in range(n)]) 		# Rather slick piece of work here Mr. duffy. Apply elsewhere
  temp = inv( np.eye(len(X00)) + X00.dot(U) )
  return temp.dot(X00)



def SC1(GF,n0=1.0):
  """Calculate Vup/Vdown for a single impurity in graphene."""
  # This returns values for the up/down spin that are only separated by a sign. There is a symmetry here that you are not exploiting.
  mag_m = 0.8
  tolerance = dtol
  delta = 0.0
  
  def n_occ(V):
    integral = quad(GF, eta, np.inf, args=V, epsabs=0.0, epsrel=dtol, limit=200 )
    return 1.0/2.0 + integral[0]/pi
    
  def FZero(delta):
    ex_split = U*mag_m
    Vdown = delta + (ex_split + hw0)/2.0
    Vup = delta - (ex_split + hw0)/2.0
    return n0 - n_occ(Vup) - n_occ(Vdown)

  while True:
    mag_temp = mag_m
    delta = newton(FZero, delta, tol=dtol, maxiter=50)
    ex_split = U*mag_m
    Vdown = delta + (ex_split + hw0)/2.0
    Vup = delta - (ex_split + hw0)/2.0
    mag_m = n_occ(Vup) - n_occ(Vdown)
    if abs(mag_m - mag_temp) <= tolerance:
      break

  return Vup, Vdown


def SCBulkSubs1(n0=1.0):
  def GF(y,V):
    g = gBulk_kZ(0,0,0,EF+1j*y)
    return Dyson1(g,V).real
  return SC1(GF,n0)


def SC_GNRSubs1(nE,m,n,n0=1.0):  
  """Calculates the SC potentials for a GNR with one impurity. Has been tested against Filipe's"""
  def GF(y,V):
    g = gRib_Arm(nE,m,n,m,n,0,EF+1j*y)
    return Dyson1(g,V).real
  return SC1(GF,n0)


def SC_GNRTop1(nE,m,n,n0=1.0):  
  """Calculates the SC potentials for a GNR with one impurity. Has been tested against Filipe's"""
  def GF(y,V):
    g = gGNRTop1(nE,m,n,EF+1j*y)
    return Dyson1(g,V).real
  return SC1(GF,n0)


def SCBulkSubs2(m,n,s,n0=1.0):
  """Calculate Vup/Vdown for a 2 impurities in graphene. Takes advantage of the symmetry a bit."""
  # This return values for the up/down spin that are only separated by a sign. There is a symmetry here that you are not exploiting.
  mag_m = 0.8
  tolerance = dtol
  delta = 0.0
  
  def GF(y,V):	# g_im is a better name
    g = Dyson(gBulk_mx2(m,n,s,EF+1j*y),V)[0,0]
    return g.real
  
  def n_occ(V):
    integral = quad(GF, eta, np.inf, args=V, epsabs=0.0, epsrel=dtol, limit=200 )
    return 1.0/2.0 + integral[0]/pi
    
  def FZero(delta):
    ex_split = U*mag_m
    Vdown = np.eye(2)*(delta + (ex_split + hw0)/2.0)
    Vup = np.eye(2)*(delta - (ex_split + hw0)/2.0)
    return n0 - n_occ(Vup) - n_occ(Vdown)

  while True:
    mag_temp = mag_m
    delta = newton(FZero, delta, tol=dtol, maxiter=50)
    ex_split = U*mag_m
    Vdown = delta + (ex_split + hw0)/2.0
    Vup = delta - (ex_split + hw0)/2.0
    mag_m = n_occ(Vup) - n_occ(Vdown)
    if abs(mag_m - mag_temp) <= tolerance:
      break

  return Vup, Vdown



def SC2(GF,n0):
  """Calculates the self-consistency for 2 substitutional atoms in a GNR.
    Currently the tidiest of all of my codes."""
  tol = dtol
  m = 0.8*np.ones(2)
  delta = np.zeros(2)

  def Vgen(m,delta):
    """Calculates both up and down spin peturbations for given m/delta"""
    ex_split = U*m
    Vup = np.diag([delta[i] - (ex_split[i] + hw0)/2.0 for i in range(2)])
    Vdown = np.diag([delta[i] + (ex_split[i] + hw0)/2.0 for i in range(2)])
    return Vup, Vdown
    
  def g_im(y,V,site):
    """Calcualtes the GF on the imaginary axis"""
    g = Dyson(GF(EF+1j*y),V)[site,site]
    return g.real
  
  def n_occ(V,site):
    """Calculates the occupancy on a given site"""
    integral = quad(g_im, eta, np.inf, args=(V,site), epsabs=0.0, epsrel=dtol, limit=200 )
    return 1.0/2.0 + integral[0]/pi

  def FZero(delta):
    """The difference between the determined and the desired occupancy"""
    Vup, Vdown = Vgen(m,delta)
    nup = np.array([n_occ(Vup,i) for i in range(2)])
    ndown = np.array([n_occ(Vdown,i) for i in range(2)])
    return nup + ndown - n0
    
  while True:
    mtemp = 1.0*m	# Necessary to create a true "copy" and not just a reference
    
    sol = optimize.root(FZero, np.zeros(2), jac=False, method='hybr')
    delta = sol.x
  
    Vup, Vdown = Vgen(m,delta)
    for i in range(2):
      m[i] = n_occ(Vup,i) - n_occ(Vdown,i)
      
    if norm(m - mtemp) <= tol:
      break
      
  return Vup, Vdown


def SC_GNRSubs2(nE,m1,n1,m2,n2,s,n0=1.0):
  """Calculates the self-consistency for 2 substitutional atoms in a GNR."""
  def GF(E):
    return gRib_mx2(nE,m1,n1,m2,n2,s,E)
  return SC2(GF,n0)


def SC_GNRTop2(nE,m1,n1,m2,n2,s,n0=1.0):
  """The Self Consistency performed for a GNR with 2 Top adsorbed impurities."""
  def GF(E):
    return gGNRTopMx2(nE,m1,n1,m2,n2,s,E)
  return SC2(GF,n0)


def SC_GNRSubs3(nE,r0,r1,r2,n0=1.0):
  """Calculates the self consistency for 3 atoms."""
  tol = dtol
  m = 0.8*np.ones(3)
  delta = np.zeros(3)
  
  def Vgen(m,delta):
    """Calculates both up and down spin peturbations for given m/delta"""
    ex_split = U*m
    Vup = np.diag([delta[i] - (ex_split[i] + hw0)/2.0 for i in range(3)])
    Vdown = np.diag([delta[i] + (ex_split[i] + hw0)/2.0 for i in range(3)])
    return Vup, Vdown
    
  def g_im(y,V,site):
    """Calcualtes the GF on the imaginary axis"""
    g = gRib_mx3(nE,r0,r1,r2,EF+1j*y)
    g_new = Dyson(g,V)[site,site]
    return g_new.real
  
  def n_occ(V,site):
    """Calculates the occupancy on a given site"""
    integral = quad(g_im, eta, np.inf, args=(V,site), epsabs=0.0, epsrel=dtol, limit=200 )
    return 1.0/2.0 + integral[0]/pi

  def FZero(delta):
    """The difference between the determined and the desired occupancy"""
    Vup, Vdown = Vgen(m,delta)
    nup = np.array([n_occ(Vup,i) for i in range(3)])
    ndown = np.array([n_occ(Vdown,i) for i in range(3)])
    return nup + ndown - n0
    
  while True:
    mtemp = 1.0*m	# Necessary to create a true "copy" and not just a reference
    
    sol = optimize.root(FZero, [0.,0.,0.], jac=False, method='hybr')		# Might want to replace the zeros here with deltas
    delta = sol.x
  
    Vup, Vdown = Vgen(m,delta)
    for i in range(3):
      m[i] = n_occ(Vup,i) - n_occ(Vdown,i)
      
    if norm(m - mtemp) <= tol:	# Still a bit dubious about this convergence condition
      break
      
  return Vup, Vdown


def SC_GNRSubsn(nE,r,n0=1.0):
  """Calculates the self consistency for n substitutional impurities. Matches in the appropriate places."""
  n = len(r)
  tol = dtol
  m = 0.8*np.ones(n)
  delta = np.zeros(n)
  
  def Vgen(m,delta):
    """Calculates both up and down spin peturbations for given m/delta"""
    ex_split = U*m
    Vup = np.diag([delta[i] - (ex_split[i] + hw0)/2.0 for i in range(n)])
    Vdown = np.diag([delta[i] + (ex_split[i] + hw0)/2.0 for i in range(n)])
    return Vup, Vdown
    
  def g_im(y,V,site):
    """Calcualtes the GF on the imaginary axis"""
    g = gRib_mxn(nE,r,EF+1j*y)
    g_new = Dyson(g,V)[site,site]
    return g_new.real
  
  def n_occ(V,site):
    """Calculates the occupancy on a given site"""
    integral = quad(g_im, eta, np.inf, args=(V,site), epsabs=0.0, epsrel=dtol, limit=200 )
    return 1.0/2.0 + integral[0]/pi

  def FZero(delta):
    """The difference between the determined and the desired occupancy"""
    Vup, Vdown = Vgen(m,delta)
    nup = np.array([n_occ(Vup,i) for i in range(n)])
    ndown = np.array([n_occ(Vdown,i) for i in range(n)])
    return nup + ndown - n0
    
  while True:
    mtemp = 1.0*m	# Necessary to create a true "copy" and not just a reference
    
    sol = optimize.root(FZero, np.zeros(n), jac=False, method='hybr')		# Might want to replace the zeros here with deltas
    delta = sol.x
  
    Vup, Vdown = Vgen(m,delta)
    for i in range(n):
      m[i] = n_occ(Vup,i) - n_occ(Vdown,i)
      
    if norm(m - mtemp) <= tol:
      break
      
  return Vup, Vdown


def SC_GNRTopn(nE,r,n0=1.0):
  """Calculates the self consistency for n top-adorsbed impurities. Matches in the appropriate places."""
  n = len(r)
  tol = dtol
  m = 0.8*np.ones(n)
  delta = np.zeros(n)
  
  def Vgen(m,delta):
    """Calculates both up and down spin peturbations for given m/delta"""
    ex_split = U*m
    Vup = np.diag([delta[i] - (ex_split[i] + hw0)/2.0 for i in range(n)])
    Vdown = np.diag([delta[i] + (ex_split[i] + hw0)/2.0 for i in range(n)])
    return Vup, Vdown
    
  def g_im(y,V,site):
    """Calcualtes the GF on the imaginary axis"""
    g = gGNRTopMxn(nE,r,EF+1j*y)
    g_new = Dyson(g,V)[site,site]
    return g_new.real
  
  def n_occ(V,site):
    """Calculates the occupancy on a given site"""
    integral = quad(g_im, eta, np.inf, args=(V,site), epsabs=0.0, epsrel=dtol, limit=200 )
    return 1.0/2.0 + integral[0]/pi

  def FZero(delta):
    """The difference between the determined and the desired occupancy"""
    Vup, Vdown = Vgen(m,delta)
    nup = np.array([n_occ(Vup,i) for i in range(n)])
    ndown = np.array([n_occ(Vdown,i) for i in range(n)])
    return nup + ndown - n0
    
  while True:
    mtemp = 1.0*m	# Necessary to create a true "copy" and not just a reference
    
    sol = optimize.root(FZero, np.zeros(n), jac=False, method='hybr')		# Might want to replace the zeros here with deltas
    delta = sol.x
  
    Vup, Vdown = Vgen(m,delta)
    for i in range(n):
      m[i] = n_occ(Vup,i) - n_occ(Vdown,i)
      
    if norm(m - mtemp) <= tol:
      break
      
  return Vup, Vdown


def SCBulkCenter(m,n,n0=1.0):
  """Calculate Vup/Vdown for center adsorbed impurities in graphene.
  Assumes that both impurities have equal magnetic moments by symmetry.
  Not converging to high accuracy, possibly because integrating the fucking center adsorbed matrix is a nightmare."""
  mag_m = 0.8
  tolerance = 1.0e-2
  delta = 0.0
  
  def GFImp(y,V):
    g = CenterMx(m,n,EF+1j*y)
    g_new = Dyson(g,V)[0,0]
    return g_new.real
  
  def n_occ(V):
    integral = quad(GFImp, eta, np.inf, args=V, epsabs=0.0, epsrel=dtol, limit=200 )
    return 1.0/2.0 + integral[0]/pi
    
  def FZero(delta):
    ex_split = U*mag_m
    Vdown = np.eye(2)*(delta + (ex_split + hw0)/2.0)
    Vup = np.eye(2)*(delta - (ex_split + hw0)/2.0)
    return n0 - n_occ(Vup) - n_occ(Vdown)

  mag_temp = -999.0	# Really? You can think of nothing better than this?
  while abs(mag_m - mag_temp) >= tolerance:
    mag_temp = mag_m
    delta = newton(FZero, delta, tol=1.0e-2, maxiter=50)	# And this tolerance is a bit arbitrary
    ex_split = U*mag_m
    Vdown = delta + (ex_split + hw0)/2.0
    Vup = delta - (ex_split + hw0)/2.0
    mag_m = n_occ(Vup) - n_occ(Vdown)

  return Vup, Vdown


def GNR_DOS(nE,m1,n1,m2,n2,s,V,E):
  """Temporary function for checking the GNR DOS in the presence of one impurity.
  Probably a bit unnecessary in general and due for retirement after its brief period of service."""
  g_mx = gRib_mx2(nE,m1,n1,m2,n2,s,E)
  g_V = Dyson(g_mx,V)[1,1]
  return -g_V.imag/pi


def SCField(GFup,GFdown,n0=1.0):
  """Calculate Vup/Vdown for a single substitutional impurity in a GNR, assuming that the zeeman field is applied ubiquitously"""
  # This returns values for the up/down spin that are only separated by a sign. There is a symmetry here that you are not exploiting.
  mag_m = 0.8
  tolerance = dtol
  delta = 0.0
  
  def nOccUp(V):
    integral = quad(GFup, eta, np.inf, args=V, epsabs=0.0, epsrel=dtol, limit=200 )
    return 1.0/2.0 + integral[0]/pi
  
  def nOccdown(V):
    integral = quad(GFdown, eta, np.inf, args=V, epsabs=0.0, epsrel=dtol, limit=200 )
    return 1.0/2.0 + integral[0]/pi
    
  def FZero(delta):
    ex_split = U*mag_m
    Vdown = delta + ex_split/2.0
    Vup = delta - ex_split/2.0
    return n0 - nOccUp(Vup) - nOccdown(Vdown)

  while True:
    mag_temp = mag_m
    delta = newton(FZero, delta, tol=dtol, maxiter=50)
    ex_split = U*mag_m
    Vdown = delta + ex_split/2.0
    Vup = delta - ex_split/2.0
    mag_m = nOccUp(Vup) - nOccdown(Vdown)
    if abs(mag_m - mag_temp) <= tolerance:
      break

  return Vup, Vdown


def SCGNRField(nE,m,n,hw0=hw0,n0=1.0):
  def GFup(y,V):
    g = gRib_Arm(nE,m,n,m,n,0,EF+1j*y,E0=-hw0/2.0)
    return Dyson1(g,V).real
  def GFdown(y,V):
    g = gRib_Arm(nE,m,n,m,n,0,EF+1j*y,E0=hw0/2.0)
    return Dyson1(g,V).real
  return SCField(GFup,GFdown,n0)


if __name__ == "__main__":
  nE, r = 6,[[1,0,0],[11,10,0],[21,20,0]]
  Vup, Vdown = SC_GNRTopn(nE,r)
  fXr = np.vectorize(lambda w: XRPAGNRTopn(nE,r,Vup,Vdown,w)[0,0].real)
  fXi = np.vectorize(lambda w: XRPAGNRTopn(nE,r,Vup,Vdown,w)[0,0].imag)
  wrlist, Xrtemp = sample_function(fXr, [0.0,0.00122,0.5e-2], tol=1e-3)
  wilist, Xitemp = sample_function(fXi, [0.0,0.00122,0.5e-2], tol=1e-3)
  Xrlist = Xrtemp[0]
  Xilist = Xitemp[0]
  pl.plot(wrlist,Xrlist)
  pl.plot(wilist,Xilist)
  pl.show()

  #nE,r = 6, [[1,0,0],[2,0,0],[4,0,0]]
  #Vup,Vdown = SC_GNRSubsn(nE,r)
  #fXi = np.vectorize(lambda w: XRPAGNRn(nE,r,Vup,Vdown,w)[0,0].imag)
  #wilist, Xitemp = sample_function(fXi, [0.0,1.0e-1], tol=1e-3)
  #Xilist = Xitemp[0]
  #pl.plot(wilist,Xilist)
  #pl.show()