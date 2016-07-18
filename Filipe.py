# First of all, let's import everything from the Dynamic File
from Dynamic import *

# Now let's create a function that plots things for a single impurity
def plot1(nE,m,n,interval=[0,1.0e-2]):
  """Calculates the susceptibility against w for a single impurity.
  Optionally allows the user to specify an interval on which the susceptibility will be plotted.
  nE: Ribbon Width
  m1,n1: Location of first impurity
  s: Sublattice (0:bb,1:bw,-1:wb)
  interval: plotting range"""
  
  Vup, Vdown = SC1GNRTop(nE,m,n)	# Loads the appropriate petrubations mxs.
  fXi = np.vectorize(lambda w: X1RPAGNRTop(nE,m,n,Vup,Vdown,w).imag)	# Just creates a vectorized version of the susceptibility
  wilist, Xitemp = sample_function(fXi,interval, tol=1e-3)
  Xilist = Xitemp[0]
  pl.plot(wilist,Xilist)
  pl.savefig("Dynamic.png")
  pl.show()  
  
def plot2(nE,m1,n1,m2,n2,s,interval=[0.0,0.002]):
  """Plots the susceptibility for two impurities.
  nE: Ribbon Width
  m1,n1: Location of first impurity
  s: Sublattice (0:bb,1:bw,-1:wb)
  interval: plotting range"""

  Vup, Vdown = SC2GNRTop(nE,m1,n1,m2,n2,s)
  fXi = np.vectorize(lambda w: X2RPAGNRTop(nE,m1,n1,m2,n2,s,Vup,Vdown,w)[0,0].imag)
  wilist, Xitemp = sample_function(fXi,interval, tol=1e-4)
  Xilist = Xitemp[0]
  pl.plot(wilist,Xilist)
  pl.show()

if __name__ == "__main__":
  # And let's start by performing a simple plot for two impurities.
  nE,m1,n1 = 6,2,0
  m2,n2,s = 6,3,0
  plot2(nE,m1,n1,m2,n2,s)
