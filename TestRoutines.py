from GFRoutines import *
from numpy.linalg import norm
from Recursive import *


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
  V[1:n-1,n-1] = tau
  V[n-1,1:n-1] = tau
  
  GMx = Dyson(gMx,V)
  
  return GMx


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
  V[1:n-1,n-1] = tau
  V[n-1,1:n-1] = tau
  
  GMx = Dyson(gMx,V)
  
  return GMx


def test(E):
  N = nE - 1
  gL,gR,VLR,VRL = Leads(N,E+1j*eta)
  H = HArmStrip(N,CenterList=[0])
  gC = gGen(E,H)[:2*N,:2*N]
  gL = RecAdd(gL,gC,VLR,VRL)
  g = RecAdd(gR,gL,VRL,VLR)
  return g[0,0]


def test2(E):
  N = nE - 1
  gL,gR,VLR,VRL = Leads(N,E+1j*eta)
  H = HArmStrip(N,CenterList=[0])
  gC = gGen(E,H)
  VLRtemp,VRLtemp = np.zeros((2*N,2*N+1)), np.zeros((2*N+1,2*N))
  VLRtemp[:2*N,:2*N], VRLtemp[:2*N,:2*N] = VLR, VRL
  gL = RecAdd(gL,gC,VLRtemp,VRLtemp)
  VLRtemp,VRLtemp = np.zeros((2*N+1,2*N)), np.zeros((2*N,2*N+1))
  VLRtemp[:2*N,:2*N], VRLtemp[:2*N,:2*N] = VLR, VRL
  g = RecAdd(gR,gL,VRLtemp,VLRtemp)
  return g[0,0]


if __name__ == "__main__":  
  nE = 6
  mC = 1
  nC = 0

  #Elist = np.linspace(-3.0+1j*eta,3.0+1j*eta,201)
  #Glist = [GMx1Center(nE,mC,nC,E)[0,0] for E in Elist]
  #pl.plot(Elist,Glist)
  #pl.show()
  
  Elist = np.linspace(-3.0+1j*eta,3.0+1j*eta,201)
  T1list = [test(E) for E in Elist]
  T2list = [test(E) for E in Elist]
  pl.plot(Elist,T1list)
  pl.plot(Elist,T2list)
  pl.show()
  

