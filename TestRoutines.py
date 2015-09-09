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
  V[:n-1,n-1] = tau
  V[n-1,:n-1] = tau
  
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
  V[:n-1,n-1] = tau
  V[n-1,:n-1] = tau
  
  GMx = Dyson(gMx,V)
  
  return GMx


def GMxSubsRec(N,ImpList,E):
  """Calculates the GF mx of a strip in an AGNR in the presence of subsitutional impurities"""
  gL,gR,VLR,VRL = Leads(N,E)
  H = HArmStrip(N,SubsList=ImpList)
  gC = gGen(E,H)
  gL = RecAdd(gL,gC,VLR,VRL)
  g = RecAdd(gR,gL,VRL,VLR)
  return g


def GMxCenterRec(N,ImpList,E):
  gL,gR,VLR,VRL = Leads(N,E)
  H = HArmStrip(N,CenterList=ImpList)
  gC = gGen(E,H)[:2*N,:2*N]
  gL = RecAdd(gL,gC,VLR,VRL)
  g = RecAdd(gR,gL,VRL,VLR)
  return g


def GMxCenterRec2(N,ImpList,E):
  """Calculates the GF of a strip in an AGNR in the presence of center adsorbed impurities"""
  nimp = len(ImpList)
  gL,gR,VLR,VRL = Leads(N,E)		# Get Leads
  H = HArmStrip(N,CenterList=ImpList)	# Hamiltonian with Center adsorbed impurities
  gC = gGen(E,H)
  VLRb, VRLb = PadZeros(VLR,(2*N,2*N+nimp)), PadZeros(VRL,(2*N+nimp,2*N))	# To get the full GF, VLR and VRL must be padded to match left and right cells.
  gL = RecAdd(gL,gC,VLRb,VRLb)
  VLRb, VRLb = PadZeros(VLR,(2*N+nimp,2*N)), PadZeros(VRL,(2*N,2*N+nimp))
  g = RecAdd(gR,gL,VRLb,VLRb)
  return g


if __name__ == "__main__":  
  nE = 9
  mI,nI = 1,0
  mP,nP = 3,0
  s = 0
  E = -1j*eta
  
  print 
  
  Dlist = range(1,60)
  glist = [GMx2Subs(nE,mI,nI,mP+D,nP+D,s,E)[0,0].imag for D in Dlist]
  print glist[-1]
  #pl.plot(Dlist,glist)
  #pl.savefig("plot.png")
  #pl.show()

  
  #nE = 9
  #mC,nC = 1,0
  #sP = 0
  #E = 1j*eta
  
  #Dlist = range(1,60)
  #m = 1
  #glist = [GMx1CenterProbe(nE,mC,nC,m+D,0+D,sP,E)[0,0].imag for D in Dlist]
  #pl.plot(Dlist,glist)
  #pl.savefig("plot.png")
  #pl.show()



  

