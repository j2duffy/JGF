"""Coupling routines for various systems"""

from GFRoutines import *
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


def J(GF):
  """A general routine for calculating the coupling between two impurities"""
  def integrand(y):
    return 1.0/pi*log( abs(1.0 + ex_split**2 * Dyson(GF(y),Vup)[1,0] * Dyson(GF(y),Vdown)[0,1])  ).real 
  C = quad(integrand, eta, np.inf, epsabs=0.0e0, epsrel=1.0e-4, limit=200 )
  return C[0]


def JBulkSubs(m,n,s):
  """The Coupling calculation in bulk graphene for substitutional impurities."""
  def GF(y):
    return gMx2Bulk(m,n,s,EF+1j*y)
  return J(GF)


def JBulkTop(m,n,s):
  """The coupling for top-adsorbed impurities in bulk graphene"""
  def GF(y):
    return gBulkTopMx(m,n,s,EF+1j*y)
  return J(GF)


def JBulkCenter(m,n):
  """The coupling for center-adsorbed impurities in bulk graphene"""
  def GF(y):
    return gBulkCenterMx(m,n,EF+1j*y)
  return J(GF)


def JGNRSubs(nE,m1,n1,m2,n2,s):
  """The coupling for substitutional impurities in a GNR"""
  def GF(y):
    return gMx2GNR(nE,m1,n1,m2,n2,s,EF+1j*y)
  return J(GF)


def JGNRTop(nE,m1,n1,m2,n2,s):
  """The coupling for top-adsorbed impurities in a GNR"""
  def GF(y):
    return gGNRTopMx(nE,m1,n1,m2,n2,s,EF+1j*y)
  return J(GF)
  
  
def JTubeSubs(nC,m,n,s):
  """The coupling for substitutional impurities in a nanotube. Not tested a whole bunch"""
  def GF(y):
    return gTubeSubsMx(nC,m,n,s,EF+1j*y)
  return J(GF)


def JGNRSubsMx(nE,m1,n1,m2,n2,s):
  """A routine for calculating the coupling in GNRs, modified to work with matrices.
  Unsurprisingly, takes a bit longer than the usual method."""
  def GF(y):
    return gMx2GNR(nE,m1,n1,m2,n2,s,EF+1j*y)
  def integrand(y):
    VRot = np.array([[-2*Vup,0],[0,-2*Vdown]])
    g = np.array([[Dyson(GF(y),Vup)[1,1],0],[0,Dyson(GF(y),Vdown)[1,1]]])
    return 1.0/pi*log(det(np.eye(2)-g.dot(VRot))).real
  C = quad(integrand, eta, np.inf, epsabs=0.0e0, epsrel=1.0e-4, limit=200 ) 
  return C[0]


def JBulkSubsMx(m,n,s):
  """A routine for calculating the coupling in Bulk, modified to work with matrices.
  Unsurprisingly, takes a bit longer than the usual method."""
  def GF(y):
    return gMx2Bulk(m,n,s,EF+1j*y)
  def integrand(y):
    VRot = np.array([[-2*Vup,0],[0,-2*Vdown]])
    g = np.array([[Dyson(GF(y),Vup)[1,1],0],[0,Dyson(GF(y),Vdown)[1,1]]])
    return 1.0/pi*log(det(np.eye(2)-g.dot(VRot))).real
  C = quad(integrand, eta, np.inf, epsabs=0.0e0, epsrel=1.0e-4, limit=200 ) 
  return C[0]


def JBulkTop3(r0,r1,r2):
  """A routine for calculating the Bulk Coupling for top adsorbed impurities with 3 atoms. 
  Been tested a little. Acutally seems alright."""
  def GF(y):
    return gBulkTop3Mx(r0,r1,r2,EF+1j*y)
  def integrand(y):
    VRot = np.array([[-2*Vup,0],[0,-2*Vdown]])
    g = np.array([[Dyson(GF(y),Vup)[2,2],0],[0,Dyson(GF(y),Vdown)[2,2]]])
    return 1.0/pi*log(det(np.eye(2)-g.dot(VRot))).real
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


if __name__ == "__main__":
  r0,r1,r2 = [[0,0,0],[1,0,0],[5,0,1]]
  print JBulkTop3(r0,r1,r2)
