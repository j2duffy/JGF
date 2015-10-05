from GFRoutines import *
from numpy.linalg import norm
from Recursive import *


def gSubs1(nE,m,n,E):
  s = 0
  g = gRib_Arm(nE,m,n,m,n,s,E)
  return Dyson1(g,eps_imp)


def GMx2Subs(nE,mI,nI,mP,nP,s,E):
  """Creates the GF matrix for a single subsitutional impurity. Order is Probe -> subsitutional"""
  g = gMx2GNR(nE,mP,nP,mI,nI,s,E)
  V = [[0,0],[0,eps_imp]]
  return Dyson(g,V)



def gTop1(nE,m,n,E):      
  """Calculates the GF of a top adsorbed impurity in a GNR"""
  # Introduce the connecting GFs
  g = np.zeros((2,2),dtype=complex)
  g[0,0] = gRib_Arm(nE,m,n,m,n,0,E)

  #Introduce the impurity GFs
  g[1,1] = 1.0/(E-eps_imp)
  
  # The peturbation connects the impurities to the lattice
  V = np.zeros([2,2],dtype=complex)
  V[1,0] = V[0,1] = tau
  
  G = Dyson(g,V)
  
  # Return the part of the matrix that governs the impurity behaviour. 
  return G


def gMxGNRgamma(nE,mC,nC,E):  
  """Returns the GF Mx of all of the connecting elements to the center adsorbed impurity.
  mC, nC give the location of the center-adsorbed impurity"""
  rC = np.array([mC,nC,0])	# Position of center adsorbed impurity (bottom left site)
  rHex = np.array([[0,0,0],[0,0,1],[1,0,0],[1,-1,1],[1,-1,0],[0,-1,1]])		# All of the sites of a hexagon (w.r.t bottom left)  
  return  gMxnGNR(nE,rHex+rC,E)


def gMxGNRgammaProbe(nE,mC,nC,mP,nP,sP,E):
  rC = np.array([mC,nC,0])	# Position of center adsorbed impurity (bottom left site)
  rHex = np.array([[0,0,0],[0,0,1],[1,0,0],[1,-1,1],[1,-1,0],[0,-1,1]])		# All of the sites of a hexagon (w.r.t bottom left)  
  r = np.concatenate(([[mP,nP,sP]],rHex + rC))	# Our complete list of positions, probe site, hexagon
  return  gMxnGNR(nE,r,E)


def GMx1Center(nE,mC,nC,E):  
  """Gets the GF a Center Adsorbed impurity."""
  n = 7		# Total number of sites (hexagon + impurity)
  
  rC = np.array([mC,nC,0])	# Position of center adsorbed impurity (bottom left site)
  rHex = np.array([[0,0,0],[0,0,1],[1,0,0],[1,-1,1],[1,-1,0],[0,-1,1]])		# All of the sites of a hexagon (w.r.t bottom left)
  r = rHex + rC		# Our complete list of positions, probe site, hexagon
  
  gMx = np.zeros((n,n),dtype=complex)
  gMx[:n-1,:n-1] = gMxnGNR(nE,r,E)	# First n-1 elements are the hexagon sites
  gMx[n-1,n-1] = 1.0/(E-eps_imp)	# Final site is that of the impurity
    
  V = np.zeros([n,n],dtype=complex)
  V[:n-1,n-1] = tau
  V[n-1,:n-1] = tau
  
  GMx = Dyson(gMx,V)
  
  return GMx


def GMx1CenterProbe(nE,mC,nC,mP,nP,sP,E):  
  """Returns the GF Mx of the center adosrbed impurity and also a single probe site.
  The site order is probe,hexagon,impurity site."""
  n = 8		# Total number of sites (hexagon + impurity + probe)
  
  rC = np.array([mC,nC,0])	# Position of center adsorbed impurity (bottom left site)
  rHex = np.array([[0,0,0],[0,0,1],[1,0,0],[1,-1,1],[1,-1,0],[0,-1,1]])		# All of the sites of a hexagon (w.r.t bottom left)
  r = np.concatenate(([[mP,nP,sP]],rHex + rC))	# Our complete list of positions, probe site, hexagon
  
  gMx = np.zeros((n,n),dtype=complex)
  gMx[:n-1,:n-1] = gMxnGNR(nE,r,E)
  gMx[n-1,n-1] = 1.0/(E-eps_imp)
    
  V = np.zeros([n,n],dtype=complex)
  V[1:n-1,n-1] = tau		# 1 -> n-2 sites are hexagon sites, they connect to n-1, the impurity site
  V[n-1,1:n-1] = tau
  
  GMx = Dyson(gMx,V)
  
  return GMx


#def GMxSubsRec(N,ImpList,E):
  #"""Calculates the GF mx of a strip in an AGNR in the presence of subsitutional impurities"""
  #gL,gR,VLR,VRL = Leads(N,E)
  #H = HArmStrip(N,SubsList=ImpList)
  #gC = gGen(E,H)
  #gL = RecAdd(gL,gC,VLR,VRL)
  #g = RecAdd(gR,gL,VRL,VLR)
  #return g


#def GMxCenterRec(N,ImpList,E):
  #gL,gR,VLR,VRL = Leads(N,E)
  #H = HArmStrip(N,CenterList=ImpList)
  #gC = gGen(E,H)[:2*N,:2*N]
  #gL = RecAdd(gL,gC,VLR,VRL)
  #g = RecAdd(gR,gL,VRL,VLR)
  #return g


#def GMxCenterRec2(N,p,ImpList,E):
  #"""Calculates the GF of a strip in an AGNR in the presence of center adsorbed impurities.
  #Doesn't work for p = 2 or more because of the way your potentials work"""
  #nimp = len(ImpList)
  #gL,gR,VLR,VRL = Leads(N,E)		# Get Leads
  #H = HArmStrip(N,p,CenterList=ImpList)	# Hamiltonian with Center adsorbed impurities
  #gC = gGen(E,H)
  #sizeP = gL.shape[0]
  #sizeI = gC.shape[0]
  #VLRb, VRLb = PadZeros(VLR,(sizeP,sizeI)), PadZeros(VRL,(sizeI,sizeP))	# To get the full GF, VLR and VRL must be padded to match left and right cells.
  #gL = RecAdd(gL,gC,VLRb,VRLb)
  #VLRb, VRLb = PadZeros(VLR,(sizeI,sizeP)), PadZeros(VRL,(sizeP,sizeI))
  #g = RecAdd(gR,gL,VRLb,VLRb)
  #return g


def GFTest(N,E):
  gL,gR,VLR,VRL = Leads(N,E)
  gR = RecAdd(gL,gR,VLR,VRL)
  return gR


def GammaBulk(m,n,E):
  """The Gamma function in bulk graphene. 
  Only one integral, and not checked enough.
  Should not work around zero"""
  def int_temp(kZ):
    q = acos( (E**2 - t**2 - 4.0*t**2 *cos(kZ)**2)/(4.0*t**2 *cos(kZ) ) )
    if q.imag < 0.0: q = -q
    
    sig = copysign(1,m+n)
    f = t*( 1.0 + 2.0*cos(kZ)*exp(sig*1j*q) )
    ft = t*( 1.0 + 2.0*cos(kZ)*exp(-sig*1j*q) )
    
    Const = 1j/(4*pi*t**2)
    Den = cos(kZ)*sin(q)
    
    temp1 = 2.0*E**3/t**2 + t*exp(-2.0*1j*q)*f**3 + t*exp(2.0*1j*q)*ft**3
    temp2 = exp( 1j*(sig*q*(m+n) + kZ*(m-n) ) )
    
    return Const*temp1*temp2/Den

  return C_int(int_temp,-pi/2,pi/2)


def gMx2BulkCenter(m,n,E):
  """Calculates the matrix of connecting sites for a center adsorbed impurity in bulk"""
  D = [m,n,0]
  hex1 = np.array([[0,0,0],[0,0,1],[1,0,0],[1,-1,1],[1,-1,0],[0,-1,1]])
  hex2 = hex1 + D
  r = np.concatenate((hex1,hex2))
  
  return BulkMxGen(r,E)


if __name__ == "__main__":  
  #m,n = 3,0
  #Elist = np.linspace(-4.0+1j*eta,4.0+1j*eta,201)
  #Glist = np.array([GammaBulk(m,n,E) for E in Elist])
  #pl.plot(Elist.real,Glist.real)
  #pl.plot(Elist.real,Glist.imag)
  #pl.show()
  
  m,n = 3,0
  E = 1.1+1j*eta
  print gMx2BulkCenter(m,n,E)[6:,6:].sum()
  

     

  

