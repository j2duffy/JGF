"""Green's functions for various systems"""
import FMod
from scipy.integrate		import quad
from scipy.optimize		import newton
import numpy 			as np
import pylab 			as pl
from numpy.linalg		import inv
from cmath			import sin, cos, log, acos, asin, sqrt, exp, pi
from math			import copysign
from functools import partial
import sys
import time

global EF
# math parameters
hbar = 1.0
eta = 1.0e-4
# material parameters
t = -1.0
EF = 0.0
wf=EF	# This needs a lot of thought/work
eps_imp = 1.0
tau = -1.0
U = 10.0
hw0 = 1.0e-3	# A default value for the Zeeman field. 
dtol = 1.0e-6		# Stands for "default tolerance". Could be better


def C_int(f,lim1,lim2):
  int_re = quad(lambda x: f(x).real, lim1, lim2, epsabs=0.0, epsrel=1.0e-4, limit=200 )
  int_im = quad(lambda x: f(x).imag, lim1, lim2, epsabs=0.0, epsrel=1.0e-4, limit=200 )
  return int_re[0] + 1j*int_im[0]


def gLin_Chain(n,E):
  """ A GF for the Linear Chain. You have not taken the sign of n into account.
    You have also just borrowed the whole expression from Mauro's notes.
    You terrible terrible cat hiss"""
  
  return 1j*exp(1j*abs(n)*acos(E/(2.0*t)))/(t*sqrt(4.0-E**2/t**2))


def gBulk_kZ(m,n,s,E,E0=0.0):
  """The Graphene Green's function
    The kZ integration is performed last
    The E0 is actually kind of useless and starting to annoy me"""
  GF = partial(FMod.gbulk_kz_int,m,n,s,E,E0,t)
  return C_int(GF,-pi/2,pi/2)


def gBulk_kA(m,n,s,E):
  """The Graphene Green's function
      The kA integration is performed last"""
  def int_temp(kA):
    qp = acos( 0.5*(-cos(kA) + sqrt( (E**2/t**2) - (sin(kA))**2 ) )  )
    qm = acos( 0.5*(-cos(kA) - sqrt( (E**2/t**2) - (sin(kA))**2 ) )  ) 

    if qp.imag < 0.0: qp = -qp
    if qm.imag < 0.0: qm = -qm

    sig = copysign(1,m-n)
    const = 1j/(4.0*pi*t**2)
    
    if s == 0:
      return const*E*( exp( 1j*( kA*(m+n) + sig*qp*(m-n) )  ) / ( sin(2*qp) + sin(qp)*cos(kA)  )		\
	+ exp( 1j*( kA*(m+n) + sig*qm*(m-n) )  ) / ( sin(2*qm) + sin(qm)*cos(kA)  )  )
    elif s == 1:
      fp = t*( 1.0 + 2.0*cos(qp)*exp(1j*kA)  )
      fm = t*( 1.0 + 2.0*cos(qm)*exp(1j*kA)  )
      
      return const*( exp( 1j*( kA*(m+n) + sig*qp*(m-n) )  )*fp / ( sin(2*qp) + sin(qp)*cos(kA)  )  	\
	+ exp( 1j*( kA*(m+n) + sig*qm*(m-n) )  )*fm / ( sin(2*qm) + sin(qm)*cos(kA)  )  ) 
    elif s == -1:
      ftp = t*( 1.0 + 2.0*cos(qp)*exp(-1j*kA)  )
      ftm = t*( 1.0 + 2.0*cos(qm)*exp(-1j*kA)  )

      return const*( exp( 1j*( kA*(m+n) + sig*qp*(m-n) )  )*ftp / ( sin(2*qp) + sin(qp)*cos(kA)  )  	\
      + exp( 1j*( kA*(m+n) + sig*qm*(m-n) )  )*ftm / ( sin(2*qm) + sin(qm)*cos(kA)  )  ) 
    else: print "Sublattice error in gBulk_kA"
  
  return C_int(int_temp,-pi/2,pi/2)


def gRib_Arm(nE,m1,n1,m2,n2,s,E,E0=0.0):
  """An interace to the FORTRAN armchair ribbon GF.
  The E0 is starting to annoy me."""
  GF = FMod.grib_arm(nE,m1,n1,m2,n2,s,E,E0,t)
  return GF


def gTube_Arm(nC,m,n,s,E):
  """The Green's Function of a Carbon Nanotube
    Problem: k is a really weird choice for index given that it alludes to the Fermi wavevector"""
  return FMod.gtube_arm(nC,m,n,s,E,t)


def gBulk_Lin(E,a=1.0):		# Seems to be off by a factor of 2
  def int_qxqy(qx,qy,Re):
    eps2 = 3.0*t**2  *a**2/4.0 *(qx**2 + qy**2)
    integrand = 1.0/(E**2 - eps2)
    if Re:
      return integrand.real
    else:
      return integrand.imag
  
  def int_qy(qy,Re): 
    lim = 2*pi/(sqrt(3.0)*a).real
    return quad(int_qxqy, -lim, lim, args=(qy,Re), epsabs=0.0, epsrel=1.0e-2, limit=200 )[0]
  
  g_re = quad(int_qy, -pi/a, pi/a, args=(True), epsabs=0.0, epsrel=1.0e-2, limit=200 )[0]
  g_im = quad(int_qy, -pi/a, pi/a, args=(False), epsabs=0.0, epsrel=1.0e-2, limit=200 )[0]
  return 2*E*sqrt(3.0)*a**2/(8.0*pi**2) *(g_re + 1j*g_im)


def gBulkArmSPA(DA,E):
  sig = copysign(1.0,-E.real)

  temp_a1 = 1j*sig*E*exp( sig*2*1j*abs(DA)*acos( (E**2 - 5*t**2)/(4*t**2) ) ) 
  temp_a2 = ( (E**2-9*t**2)*(-E**2+t**2) )**(1.0/4.0)
  temp_a3 = sqrt( -sig*( 1j/(abs(DA)*pi*(3*t**2 + E**2) )  )  )
  ga = (temp_a1/temp_a2)*temp_a3

  temp_b1 = 1j*sig*( E*exp(sig*2*1j*abs(DA)*acos( -sqrt( 1 - E**2/t**2 ) ) ) )
  temp_b2 = sqrt(3*t**2 + E**2)*( E**2*(t**2 - E**2) )**(1.0/4.0)
  temp_b3 = sqrt( sig*(1j/(pi*abs(DA))) )
  gb = (temp_b1/temp_b2)*temp_b3

  g = ga + gb

  return g


def gFakeSPA(DA,E):
  """ An SPA thing that I worked out in Mathematica.
    Probably totally wrong."""
  A = 1j*sqrt(1j/DA)*sqrt(E)/(sqrt(3.0*pi)*(t+0j)**(3.0/2.0))
  #EQ = exp(sig*2*1j*abs(DA)*acos( -sqrt( 1 - E**2/t**2 ) ) )

  return A#*EQ
  

def gLine_ky(DA,ky,s,E,a=1.0):
  """The Graphene Green's function, projected onto the k_y, x basis,
    used in the calculation of infinite lines of impurities"""
  q = acos( (E**2 - t**2 - 4.0*t**2 *cos(ky*a/2)**2)/(4.0*t**2 *cos(ky*a/2) ) )
  if q.imag < 0.0: q = -q

  Const = 1j/(4*t**2)
  Den = cos(ky*a/2)*sin(q)
  
  if s == 0:
    sig = copysign(1,DA)	# You haven't made sure DA goes here
    return Const*E*exp( 1j*sig*q*DA )/ Den 
  elif s == 1: 
    sig = copysign(1,DA)
    f = t*( 1.0 + 2.0*cos(ky*a/2)*exp( 1j*sig*q ) )
    return Const*f*exp( 1j*sig*q*DA )/Den  
  elif s == -1:
    sig = copysign(1,DA-1)
    ft = t*( 1.0 + 2.0*cos(ky*a/2)*exp(-sig*1j*q) )
    return Const*ft*exp( 1j*sig*q*DA )/Den 
  else: print "Sublattice error in gLine_ky"
  

def gLine_kZ(DA,kZ,s,E):	# This really needs to be tested against something
  """The Graphene Green's function, in the k_y, x basis"""
  q = acos((E**2 - t**2 - 4.0*t**2 *cos(kZ)**2)/(4.0*t**2 *cos(kZ)))
  if q.imag < 0.0: q = -q

  Const = 1j/(4*t**2)
  Den = cos(kZ)*sin(q)
  
  if s == 0:
    sig = copysign(1,DA)	# You haven't made sure DA goes here
    return Const*E*exp(1j*sig*q*DA)/Den 
  elif s == 1: 
    sig = copysign(1,DA)
    f = t*(1.0 + 2.0*cos(kZ)*exp(1j*sig*q))
    return Const*f*exp( 1j*sig*q*DA )/Den  
  elif s == -1:
    sig = copysign(1,DA-1)
    ft = t*(1.0 + 2.0*cos(kZ)*exp(-sig*1j*q))
    return Const*ft*exp(1j*sig*q*DA)/Den 
  else: print "Sublattice error in gLine_kZ"


def gSI_kZ(m1,n1,m2,n2,s,E):
  """The Semi-Infinite Graphene Green's function
      The kZ integration is performed last"""
  GF = partial(FMod.gsi_kz_int,m1,n1,m2,n2,s,E,t)
  return C_int(GF,-pi/2,pi/2)


def gSIZigtest(DA1,DA2,DZ,s_lat,E):		# You need to change the sublattice notation. Which will be hard
  """The Graphene Green's function
      The kA integration is performed last"""
  def int_temp(kA):
    qp = acos( 0.5*(-cos(kA) + sqrt( (E**2/t**2) - (sin(kA))**2 ) )  )
    qm = acos( 0.5*(-cos(kA) - sqrt( (E**2/t**2) - (sin(kA))**2 ) )  ) 

    if qp.imag < 0.0: qp = -qp
    if qm.imag < 0.0: qm = -qm

    sig = copysign(1,DZ)		# Check this versus zigzag SI (should be same though, on exponential)
    const = 1j/(2.0*pi*t**2)
    
    if s_lat == 'bb':
      return const*E*exp(1j*sig*qp*(DZ))*sin(kA*DA1)*sin(kA*DA2) / ( sin(2*qp) + sin(qp)*cos(kA) )		\
	+ const*E*exp(1j*sig*qm*(DZ))*sin(kA*DA1)*sin(kA*DA2) / ( sin(2*qm) + sin(qm)*cos(kA)  ) 
    elif s_lat == 'ww':
      fp = 1.0 + 2.0*cos(qp)*exp(1j*kA)
      fm = 1.0 + 2.0*cos(qm)*exp(1j*kA)
      ftp = 1.0 + 2.0*cos(qp)*exp(-1j*kA)
      ftm = 1.0 + 2.0*cos(qm)*exp(-1j*kA)
      fpasq = 1.0 + 4.0*cos(qp)*cos(kA) + 4.0*cos(qp)**2
      fmasq = 1.0 + 4.0*cos(qm)*cos(kA) + 4.0*cos(qm)**2
      Np = ( ftp*exp(-1j*kA*DA1) - fp*exp(1j*kA*DA1) )*( fp*exp(1j*kA*DA2) - ftp*exp(-1j*kA*DA2) )/fpasq
      Nm = ( ftm*exp(-1j*kA*DA1) - fm*exp(1j*kA*DA1) )*( fm*exp(1j*kA*DA2) - ftm*exp(-1j*kA*DA2) )/fmasq
      return 0.25*const*E*exp(1j*sig*qp*DZ)*Np / ( sin(2*qp) + sin(qp)*cos(kA) )		\
	+ 0.25*const*E*exp(1j*sig*qm*DZ)*Nm / ( sin(2*qm) + sin(qm)*cos(kA)  ) 
    else: print 's_lat not a valid character'
  
  return C_int(int_temp,-pi/2,pi/2)



if __name__ == "__main__":   
  nE,m1,n1,m2,n2,s,E,E0 = 6,3,0,5,2,1,1.2+1j*eta,1.0
  print gRib_Arm(nE,m1,n1,m2,n2,s,E,E0=E0)
