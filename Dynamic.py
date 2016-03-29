# A file for just running Dynamic calculations and nothing else.
# Should be the most up to date thing for Dynamic calculations, and the 
from GFRoutines import *
from scipy import optimize
from numpy.linalg import norm
from functools import partial
import profile
import multiprocessing
from functionsample import sample_function
from scipy.interpolate import UnivariateSpline


def X1HF(GF,Vup,Vdown,w):
  """Calculates the Hartree-Fock spin susceptibility for a general GF"""
  def spin_sus_int12(y):
    return hbar/(2.0*pi) *( GF(wf + 1j*y,Vup)*GF(wf + w + 1j*y,Vdown) + GF(wf - 1j*y,Vdown)*GF(wf - w - 1j*y,Vup) )

  def spin_sus_int3(w_dum):
    return - 1j*hbar/(2.0*pi) *GF(w_dum - 1j*eta,Vup)*GF(w + w_dum + 1j*eta,Vdown)
    
  I12 = C_int(spin_sus_int12,eta,np.inf)
  I3 = C_int(spin_sus_int3,wf-w,wf)
    
  return I12 + I3


def X1HFBulkSubs(Vup,Vdown,w):
  """Calculates the Hartree-Fock spin susceptibility for a substitutional impurity Bulk Graphene"""
  def GF(E,V):
    g = gBulk_kZ(0,0,0,E)
    return Dyson1(g,V)
  return X1HF(GF,Vup,Vdown,w)


def X1HFSISubs(m,n,Vup,Vdown,w):
  """Calculates the on-site Hartree-Fock spin susceptibility for a substitutional impurity in Semi-Infinite Graphene."""
  def GF(E,V):
    g = gSI_kZ(m,n,m,n,0,E)
    return Dyson1(g,V)
  return X1HF(GF,Vup,Vdown,w)


def X1HFGNRSubs(nE,m,n,Vup,Vdown,w):
  """Calculates the Hartree-Fock spin susceptibility for a substitutional impurity in a GNR."""
  def GF(E,V):
    g = gRib_Arm(nE,m,n,m,n,0,E)
    return Dyson1(g,V)
  return X1HF(GF,Vup,Vdown,w)


def X1HFGNRTop(nE,m,n,Vup,Vdown,w):
  """Calculates the Hartree-Fock spin susceptibility for a top adsorbed impurity in a GNR."""
  def GF(E,V):
    g = g1GNRTop(nE,m,n,E)
    return Dyson1(g,V)
  return X1HF(GF,Vup,Vdown,w)


def X1RPABulkSubs(Vup,Vdown,w):
  """Calculates the on-site RPA spin susceptibility for a substitutional impurity in bulk graphene"""
  X0 = X1HFBulkSubs(Vup,Vdown,w)
  return X0/(1.0+U*X0)


def X1RPAGNRSubs(nE,m,n,Vup,Vdown,w):
  """Calculates the on-site RPA spin susceptibility for a substitutional impurity in a GNR"""
  X0 = X1HFGNRSubs(nE,m,n,Vup,Vdown,w)
  return X0/(1.0+U*X0)


def X1RPAGNRTop(nE,m,n,Vup,Vdown,w):
  """Calculates the on-site RPA spin susceptibility for a top adsorbed impurity in a GNR"""
  X0 = X1HFGNRTop(nE,m,n,Vup,Vdown,w)
  return X0/(1.0+U*X0)



def XHF(GF,site,Vup,Vdown,w):	# Might be better to include the Dyson function within the calling functions
  """Calculates the Hartree-Fock spin susceptibility between two sites"""
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


def X2HFBulkSubs(m,n,s,site,Vup,Vdown,w):
  """Calculates the HF spin susceptibility for two substitutional impurities in bulk graphene"""
  def GF(E):
    return gMx2Bulk(m,n,s,E)	# Could maybe include the site term here and save a fucking bunch of time  
  return XHF(GF,site,Vup,Vdown,w)


def X2HFSISubs(m1,n1,m2,n2,s,site,Vup,Vdown,w):
  """Calculates the HF spin susceptibility for two substitutional impurities in semi-infinite graphene, 
    Requires testing"""
  def GF(E):
    return gMx2SI(m1,n1,m2,n2,s,E)
    
  return XHF(GF,site,Vup,Vdown,w)


def X2HFGNRSubs(nE,m1,n1,m2,n2,s,site,Vup,Vdown,w):
  """Calculates the HF spin susceptibility for two substitutional impurities in a GNR"""
  def GF(E):
    g = gGNRSubsMx(nE,m1,n1,m2,n2,s,E)
    return g  
  return XHF(GF,site,Vup,Vdown,w)


def X2HFGNRTop(nE,m1,n1,m2,n2,s,site,Vup,Vdown,w):
  """Calculates the HF spin susceptibility for two top-adsorbed impurities in a GNR"""
  def GF(E):
    g = gGNRTopMx(nE,m1,n1,m2,n2,s,E)
    return g   
  return XHF(GF,site,Vup,Vdown,w)


def X2RPABulkSubs(m,n,s,Vup,Vdown,w):
  """Calculates the RPA spin susceptibility for two substitutional impurities in bulk graphene"""
  n = 2
  X00 = np.array([[X2HFBulkSubs(m,n,s,[i,j],Vup,Vdown,w) for j in range(n)] for i in range(n)])
  temp = inv( np.eye(len(X00)) + X00.dot(U) )
  return temp.dot(X00)


def X2RPAGNRSubs(nE,m1,n1,m2,n2,s,Vup,Vdown,w):
  """Calculates the RPA spin susceptibility for two substitutional impurities in a GNR"""
  n = 2
  X00 = np.array([[X2HFGNRSubs(nE,m1,n1,m2,n2,s,[i,j],Vup,Vdown,w) for j in range(n)] for i in range(n)])
  temp = inv( np.eye(len(X00)) + X00.dot(U) )
  return temp.dot(X00)


def X2RPAGNRTop(nE,m1,n1,m2,n2,s,Vup,Vdown,w):
  """Calculates the RPA spin susceptibility for two top-adsorbed impurities in a GNR"""
  n = 2
  X00 = np.array([[X2HFGNRTop(nE,m1,n1,m2,n2,s,[i,j],Vup,Vdown,w) for j in range(n)] for i in range(n)])
  temp = inv( np.eye(len(X00)) + X00.dot(U) )
  return temp.dot(X00)


def X3HFGNRSubs(nE,r0,r1,r2,site,Vup,Vdown,w):
  """Calculates the HF spin susceptibility for 3 substitutional impurities in a GNR"""
  i,j = site
  def GF(E):
    return gMx3GNR(nE,r0,r1,r2,E)
    
  return XHF(GF,site,Vup,Vdown,w)


def X3RPAGNRSubs(nE,r0,r1,r2,Vup,Vdown,w):
  """Gets the RPA spin susceptibility for a GNR for 3 atoms.
    Should be easy to extend to further dimensions. 
    There's also a bunch of obvious symmetries here that we're not exploiting (since the output mx is symmetric (NOT HERMITIAN))"""
  n = 3
  X00 = np.array([[X3HFGNRSubs(nE,r0,r1,r2,[i,j],Vup,Vdown,w) for j in range(n)] for i in range(n)]) 		# Rather slick piece of work here mr. duffy. Apply elsewhere
  temp = inv( np.eye(len(X00)) + X00.dot(U) )
  return temp.dot(X00)



def XnHFGNRSubs(nE,r,site,Vup,Vdown,w):
  """The HF spin sus for n substitutional atoms in a GNR."""
  def GF(E):
    return gMxnGNR(nE,r,E)
  return XHF(GF,site,Vup,Vdown,w)


def XRPAGNRn(nE,r,Vup,Vdown,w):
  """Gets the RPA spin susceptibility for a GNR for n substitutional impurities.
    There are a lot of symmetries not being exploited here (the matrix is symmetric)"""
  n = len(r)
  X00 = np.array([[XnHFGNRSubs(nE,r,[i,j],Vup,Vdown,w) for j in range(n)] for i in range(n)]) 		# Rather slick piece of work here Mr. duffy. Apply elsewhere
  temp = inv( np.eye(len(X00)) + X00.dot(U) )
  return temp.dot(X00)


def XnHFGNRTop(nE,r,site,Vup,Vdown,w):
  """The HF spin sus for n substitutional atoms in a GNR."""
  def GF(E):
    return gMxnGNRTop(nE,r,E)
  return XHF(GF,site,Vup,Vdown,w)


def XnRPAGNRTop(nE,r,Vup,Vdown,w):
  """Gets the RPA spin susceptibility for a GNR for n substitutional impurities.
    There are a lot of symmetries not being exploited here (the matrix is symmetric)"""
  n = len(r)
  X00 = np.array([[XnHFGNRTop(nE,r,[i,j],Vup,Vdown,w) for j in range(n)] for i in range(n)]) 		# Rather slick piece of work here Mr. duffy. Apply elsewhere
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


def SC1BulkSubs(n0=1.0):
  def GF(y,V):
    g = gBulk_kZ(0,0,0,EF+1j*y)
    return Dyson1(g,V).real
  return SC1(GF,n0)


def SC1GNRSubs(nE,m,n,n0=1.0):  
  """Calculates the SC potentials for a GNR with one impurity. Has been tested against Filipe's"""
  def GF(y,V):
    g = gRib_Arm(nE,m,n,m,n,0,EF+1j*y)
    return Dyson1(g,V).real
  return SC1(GF,n0)


def SC1GNRTop(nE,m,n,n0=1.0):  
  """Calculates the SC potentials for a GNR with one impurity. Has been tested against Filipe's"""
  def GF(y,V):
    g = g1GNRTop(nE,m,n,EF+1j*y)
    return Dyson1(g,V).real
  return SC1(GF,n0)




def SC2(GF,n0):
  """Calculates the self-consistency for 2 substitutional atoms in a GNR.
  Is this specific to GNRs?"""
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


def SC2BulkSubs(m,n,s,n0=1.0):
  """Calculate Vup/Vdown for a 2 impurities in graphene. Takes advantage of the symmetry a bit."""
  # This return values for the up/down spin that are only separated by a sign. There is a symmetry here that you are not exploiting.
  mag_m = 0.8
  tolerance = dtol
  delta = 0.0
  
  def GF(y,V):	# g_im is a better name
    g = Dyson(gMx2Bulk(m,n,s,EF+1j*y),V)[0,0]
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


def SC2GNRSubs(nE,m1,n1,m2,n2,s,n0=1.0):
  """Calculates the self-consistency for 2 substitutional atoms in a GNR."""
  def GF(E):
    return gGNRSubsMx(nE,m1,n1,m2,n2,s,E)
  return SC2(GF,n0)


def SC2GNRTop(nE,m1,n1,m2,n2,s,n0=1.0):
  """The Self Consistency performed for a GNR with 2 Top adsorbed impurities."""
  def GF(E):
    return gGNRTopMx(nE,m1,n1,m2,n2,s,E)
  return SC2(GF,n0)


def SC3GNRSubs(nE,r0,r1,r2,n0=1.0):
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
    g = gMx3GNR(nE,r0,r1,r2,EF+1j*y)
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


def SCnGNRSubs(nE,r,n0=1.0):
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
    g = gMxnGNR(nE,r,EF+1j*y)
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


def SCnGNRTop(nE,r,n0=1.0):
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
    g = gMxnGNRTop(nE,r,EF+1j*y)
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


def SC2BulkCenter(m,n,n0=1.0):
  """Calculate Vup/Vdown for center adsorbed impurities in graphene.
  Assumes that both impurities have equal magnetic moments by symmetry.
  Not converging to high accuracy, possibly because integrating the fucking center adsorbed matrix is a nightmare."""
  mag_m = 0.8
  tolerance = 1.0e-2
  delta = 0.0
  
  def GFImp(y,V):
    g = gMx2BulkCenter(m,n,EF+1j*y)
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


def gImpTest():
  def GF(y,V):
    g = 1.0/(EF+1j*y-eps_imp)
    return Dyson1(g,V).real
  return SC1(GF)


def X1ImpTest(Vup,Vdown,w):
  """Calculates the Hartree-Fock spin susceptibility for a substitutional impurity Bulk Graphene"""
  def GF(E,V):
    g = 1.0/(E-eps_imp)
    return Dyson1(g,V)
  return X1HF(GF,Vup,Vdown,w)


def X1RPATest(Vup,Vdown,w):
  """Calculates the on-site RPA spin susceptibility for a substitutional impurity in bulk graphene"""
  X0 = X1ImpTest(Vup,Vdown,w)
  return X0/(1.0+U*X0)


if __name__ == "__main__":
  #Single Impurity
  nE = 6
  m,n = 3,0
  Vup, Vdown = SC1GNRTop(nE,m,n)
  fXi = np.vectorize(lambda w: X1RPAGNRTop(nE,m,n,Vup,Vdown,w).imag)
  wilist, Xitemp = sample_function(fXi,[0,1.0e-2], tol=1e-3)
  Xilist = Xitemp[0]
  Xilist = Xilist - Xilist.min()/2.0
  #spline = UnivariateSpline(wilist,Xilist)
  #roots = spline.roots()
  #print roots[1]-roots[0]
  pl.plot(wilist,Xilist)
  pl.savefig("Dynamic.png")
  pl.show()
  #np.savetxt("1XV1.dat", zip(wilist,Xilist))
  
  #Flist = []
  #for m in range(1,9):
    #x,y = np.loadtxt("Dynamic_%g.dat" % (m,)).T
    #y = y - y.min()/2.0
    #spline = UnivariateSpline(x,y)
    #roots = spline.roots()
    #FWHM = roots[1] - roots[0]
    #Flist.append(FWHM)
  #pl.plot(range(1,9),Flist)
  #pl.savetxt("X1FWHM.dat",zip(range(1,9),Flist))
  #pl.show()

  #Double Impurity
  #nE,m1,n1 = 6,2,0
  #m2,n2,s = 6,3,0
  #Vup, Vdown = SC2GNRTop(nE,m1,n1,m2,n2,s)
  #fXi = np.vectorize(lambda w: X2RPAGNRTop(nE,m1,n1,m2,n2,s,Vup,Vdown,w)[0,0].imag)
  #wilist, Xitemp = sample_function(fXi, [0.0,0.002], tol=1e-4)
  #Xilist = Xitemp[0]
  #pl.plot(wilist,Xilist)
  #pl.show()
  #np.savetxt("2XV1_2.dat", zip(wilist,Xilist))
    
  #Triple Impurity
  #nE, r = 6,[[1,0,0],[11,10,0],[6,3,0]]
  #Vup, Vdown = SCnGNRTop(nE,r)
  #fXr = np.vectorize(lambda w: XnRPAGNRTop(nE,r,Vup,Vdown,w)[0,0].real)
  #fXi = np.vectorize(lambda w: XnRPAGNRTop(nE,r,Vup,Vdown,w)[0,0].imag)
  #wilist, Xitemp = sample_function(fXi, [0.0,1.0e-2], tol=1e-4)
  #Xilist = Xitemp[0]
  #np.savetxt("triple.dat",zip(wilist,Xilist))
  #pl.plot(wilist,Xilist)
  #pl.show()
    
  #Dlist = range(1000,3000,50)
  #Flist = []
  #for D in Dlist:
    #x, y = np.loadtxt("Dynamic_%g.dat" % (D,)).T
    #y = y - y.min()/2.0 
    #spline = UnivariateSpline(x,y)
    ##pl.plot(x,y)
    ##pl.plot(x,spline(x),'o')
    ##pl.show()
    #roots = spline.roots()
    #FWHM = roots[1] - roots[0]
    #Flist.append(FWHM)
  #pl.plot(Dlist,Flist)
  #np.savetxt("XFWHM_DZ12.dat",zip(Dlist,Flist))
  #pl.show()
  
  #x, y = np.loadtxt("test.dat").T
  #y = y - y.min()/2.0 
  #spline = UnivariateSpline(x,y)
  #pl.plot(x,y)
  #pl.plot(x,spline(x),'o')
  #pl.show()
  #roots = spline.roots()
  #FWHM = roots[1] - roots[0]
  #print FWHM

      