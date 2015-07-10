import numpy as np
from numpy import dot
from GF import *
import collections
import functools
from operator import mul
import random


class memoized(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   '''
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args)
      if args in self.cache:
         return self.cache[args]
      else:
         value = self.func(*args)
         self.cache[args] = value
         return value
   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)


def HArmStrip(N):
  """Creates the Hamiltonian of an armchair strip (building block of a nanoribbon.
  Uses the N numbering convention."""
  H = np.zeros([2*N,2*N])
  for i in range(N-1):
    H[i,i+1], H[i+1,i] = 2*(t,)
  for i in range(N,2*N-1):
    H[i,i+1], H[i+1,i] = 2*(t,)
  for i in range(0,N,2):
    H[i,N+i], H[N+i,i] = 2*(t,)
  return H


def HArmStripOld(N):
  """Creates the Hamiltonian of an armchair strip (building block of a nanoribbon).
  Uses the S numbering convention."""
  H = np.zeros([2*N,2*N])
  for i in range(N-1):
    H[i,i+1], H[i+1,i] = 2*(t,)
  for i in range(N,2*N-1):
    H[i,i+1], H[i+1,i] = 2*(t,)
  for i in range(0,N,2):
    H[i,2*N-1-i], H[2*N-1-i,i] = 2*(t,)
  return H


def gArmStrip(E,N):
  """Calculates the GF of an armchair strip.
  Uses the N numbering convention"""
  return inv(E*np.eye(2*N) - HArmStrip(N))


def gArmStripOld(E,N):
  """Calculates the GF of an armchair strip.
  Uses the S numbering convention"""
  return inv(E*np.eye(2*N) - HArmStripOld(N))


def HBigArmStrip(N,p):
  """Creates the Hamiltonian for an armchair strip N atoms across and p "unit cells" in width.
  It uses the N numbering convention."""
  H = np.zeros((2*N*p,2*N*p))
  for j in range(1,2*p+1):
    for i in range((j-1)*N,j*N-1):
      H[i,i+1], H[i+1,i] = 2*(t,)
  for i in range(0,2*N*p-N,2):
    H[i,i+N], H[i+N,i] = 2*(t,)
    
  return H


def HBigArmStripSubs(N,p,k):
  """Creates the Hamiltonian for an armchair strip N atoms across and p "unit cells" in width.
  Populates the strip with k impurities at randomly chosen sites.
  It uses the N numbering convention."""
  H = np.zeros((2*N*p,2*N*p))
  for j in range(1,2*p+1):
    for i in range((j-1)*N,j*N-1):
      H[i,i+1], H[i+1,i] = 2*(t,)
  for i in range(0,2*N*p-N,2):
    H[i,i+N], H[i+N,i] = 2*(t,)
  
  for i in random.sample(range(2*N*p),k):
    H[i,i] = 1.0	# Should replace this with some constant
    
  return H


def HBigArmStripTop(N,p,k):
  """Creates the Hamiltonian for an armchair strip N atoms across and p "unit cells" in width.
  Populates the strip with k Top-Adsorbed impurities which connect to randomly chosen sites.
  It uses the N numbering convention."""
  H = np.zeros((2*N*p+k,2*N*p+k))
  for j in range(1,2*p+1):
    for i in range((j-1)*N,j*N-1):
      H[i,i+1], H[i+1,i] = 2*(t,)
  for i in range(0,2*N*p-N,2):
    H[i,i+N], H[i+N,i] = 2*(t,)
  
  for i,k in enumerate(random.sample(range(2*N*p),k)):
    H[2*N*p+i,k] = t
    H[k,2*N*p+i] = t
    H[2*N*p+i,2*N*p+i] = 1.0	# Again that random constant
    
  return H


def gBigArmStrip(E,N,p):
  """Calculates the GF for the large armchair strip"""
  return inv(E*np.eye(2*N*p) - HBigArmStrip(N,p))


def gBigArmStripSubs(E,N,p,k):
  """Calculates the GF for a large armchair strip with k randomly distributed substitutional impurities"""
  return inv(E*np.eye(2*N*p) - HBigArmStrip(N,p))


def VArmStrip(N):
  """Calculates the LR and RL connection matrices for the armchair strip.
  Uses the N numbering convention."""
  VLR, VRL = np.zeros([2*N,2*N]),np.zeros([2*N,2*N])	# Done this way for a reason, using tuple properties produces views, not copies.
  for i in range(1,N,2):
    VLR[N+i,i], VRL[i,N+i] = 2*(t,)
  return VLR, VRL


def VArmStripOld(N):
  """Calculates the LR and RL connection matrices for the armchair strip.
  Uses the S numbering convention"""
  VLR, VRL = np.zeros([2*N,2*N]),np.zeros([2*N,2*N])	# Done this way for a reason, using tuple properties produces views, not copies.
  for i in range(1,N,2):	# Think this should be N-1
    VLR[2*N-1-i,i], VRL[i,2*N-1-i] = 2*(t,)
  return VLR, VRL


def RecAdd(g00,g11,V01,V10):
  """Add a cell g11 to g00 recursively using the Dyson Formula, get a cell G11"""
  return dot(inv(np.eye(2*N)-dot(dot(g11,V10), dot(g00,V01))),g11)


def gEdge(gC,V01,V10,tol=1.0e-4):
  """Obtain a semi-infinite lead by simple recusive addition (in the 1 direction)"""
  g11 = gC.copy()
  gtemp = np.zeros(2*N)
  while np.linalg.norm(g11 - gtemp)/np.linalg.norm(g11) > tol:
    gtemp = g11.copy()
    g11 = RecAdd(g11,gC,V01,V10)
  return g11


def gOffDiagonal(g22,g11,g10,g01,g00,V12,V21):
  """Connects a cell 2 to a lead ending in cell 1 and connected to cell 0.
  Returns off diagonal elements G22, G20, G02, G00 """
  G22 = dot(inv(np.eye(g22.shape[0])  - dot(dot(g22,V21),dot(g11,V12)) ),g22)
  G20 = dot(G22,dot(V21,g10))
  G00 = g00 + dot(g01,dot(V12,G20))
  G02 = dot(g01,dot(V12,G22))
  return G22, G20, G02, G00


def RubioSancho(g00,V01,V10,tol=1.0e-4):
  """Uses the Rubio Sancho method to get the GF at the end of a semi-infinite lead.
  Borrowed directly from FORTRAN."""
  smx = dot(g00,V01)
  tmx = dot(g00,V10)
  identity = np.eye(g00.shape[0]) 
  sprod = identity

  Transf = tmx.copy()
  Transf_Old = np.zeros(g00.shape[0])	# Broadcasts to correct shape
  while np.linalg.norm(Transf - Transf_Old)/np.linalg.norm(Transf) > tol:	# this condition is well dodge, surely you want, like, the relative error?
    # Update t
    temp = inv( identity - dot(tmx,smx) - dot(smx,tmx))
    tmx = dot( temp, dot(tmx,tmx) )
    
    # Update sprod
    sprod = dot(sprod,smx)
  
    # Update Transfer mx
    Transf_Old = Transf.copy()
    Transf = Transf + dot(sprod,tmx)

    # Update s
    smx = dot( temp,  dot(smx,smx) )
  return dot( inv( identity - dot( dot(g00,V01) , Transf)) ,  g00 ) 


def RubioSanchoTest(g00,V01,V10):
  """An absolutely insane verison of the Rubio Sancho method that implements all kinds of recursive/dynamic programming madness."""
  Imx = np.eye(g00.shape[0]) 
  
  @memoized
  def s(i):
    if i == 0:
      return dot(g00,V01)
    else:
      s0 = s(i-1)
      t0 = t(i-1)
      return dot( inv( Imx - dot(t0,s0) - dot(s0,t0)) , dot(s0,s0) )
      
  @memoized
  def t(i):
    if i == 0:
      return dot(g00,V10)
    else:
      s0 = s(i-1)
      t0 = t(i-1)
      return dot( inv( Imx - dot(t0,s0) - dot(s0,t0)) , dot(t0,t0) )
    
  #@memoized
  def T(n):
    if n==0:
      return t(0)
    else:
      sprod = s(0)
      for i in range(1,n):
	sprod = dot(sprod,s(i))
      return dot(sprod,t(n))
    
  Transf = T(0)
  T(1),T(2),T(3)
  Transf_Old = 0.0
  i = 1
  while np.linalg.norm(Transf - Transf_Old) > 1.0e-4:
    #print i, T(i)
    Transf_Old = Transf.copy()
    Transf += T(i)
    i += 1
    
  return dot(inv(Imx-dot(g00,dot(V01,Transf))),g00)


def gRibArmRecursive(N,E):
  """Calculates the GF (in a single unit cell) recursively. 
  Probably should be thrown out once you have better routines"""
  VLR, VRL = VArmStrip(N)	# Takes up a little bit of space since this has to be done every E, but really shouldn't have much effect on performance
  gC = gArmStrip(E,N)
  gR = RubioSancho(gC,VRL,VLR)	# The RIGHTMOST cell, not the cell on the right when we add
  gL = RubioSancho(gC,VLR,VRL)
  return RecAdd(gR,gL,VLR,VRL)


if __name__ == "__main__":
  N = 3
  p = 2
  k = 2
  
  print HBigArmStrip(N,p)
