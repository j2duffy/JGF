# Currently working on some shit with my Dynamic code
from Dynamic import *

if __name__ == "__main__":    
  nE,m,n = 6,3,0
  Vup, Vdown = SC1GNRTop(nE,m,n)
  fXi = np.vectorize(lambda w: X1HFGNRTop(nE,m,n,Vup,Vdown,w).imag)
  wilist, Xitemp = sample_function(fXi,[0,1.0e-2], tol=1e-3)
  Xilist = Xitemp[0]
  pl.plot(wilist,Xilist)
  pl.savefig("test.png")
  pl.show()