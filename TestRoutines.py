from GFRoutines import *
from numpy.linalg import norm
from Recursive import *


def GMxTest(nE,mC,nC,mP,nP,sP,E):  
  """Probes the GF of a given position in the presence of a center adorbed impurity.
  May have to be some distance from the hexagon, since otherwise we're adding sites unnecessarily"""
  n = 8		# Total number of sites (hexagon + impurity + probe)
  
  rC = np.array([mC,nC,0])	# Position of center adsorbed impurity (bottom left site)
  rHex = np.array([[0,0,0],[0,0,1],[1,0,0],[1,-1,1],[1,-1,0],[0,-1,1]])		# All of the sites of a hexagon (w.r.t bottom left)
  r = np.concatenate(([[mP,nP,sP]],rHex + rC))	# Our complete list of positions, probe site, hexagon
  
  gMx = np.zeros((8,8),dtype=complex)
  gMx[:n-1,:n-1] = gMxnGNR(nE,r,E)
  gMx[7,7] = 1.0/(E-eps_imp)
    
  V = np.zeros([n,n],dtype=complex)
  V[1:n-1,n-1] = tau
  V[n-1,1:n-1] = tau
  
  GMx = Dyson(gMx,V)
  
  return GMx[0,0]


if __name__ == "__main__":  
  nE = 6
  mC = 1
  nC = 0
  
  mP = 1
  nP = 0
  sP = 0
  E = 1.2 + 1j*eta
  print GMxTest(nE,mC,nC,mP,nP,sP,E)
  
  N = nE - 1
  gL,gR,VLR,VRL = Leads(N,E)
  H = HArmStrip(N,CenterList=[0])
  gC = gGen(E,H)
