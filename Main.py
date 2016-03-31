# Currently working on some shit with my Dynamic code
from Dynamic import *

def plot(nE, DZ,interval):
  Vup, Vdown = SC1GNRTop(nE,DZ,0)
  fXi = np.vectorize(lambda w: X1RPAGNRTop(nE,DZ,0,Vup,Vdown,w).imag)
  wilist, Xitemp = sample_function(fXi,interval, tol=1e-3)
  Xilist = Xitemp[0]
  return wilist, Xilist

def humanplot(nE, DZ):
  # Initial plot
  w,X = plot(nE, DZ,[0,1.0e-2])
  pl.plot(w,X)
  pl.show()
  # Should start your loop here
  while True:
    # Get Center
    imin = X.argmin()
    wmin = w[imin]
    #Request range
    wrange = eval(raw_input("Select a range:"))
    # Replot with new range 
    w,X = plot(nE, DZ,[wmin-wrange,wmin+wrange])
    pl.plot(w,X)
    pl.show()
    # If range works out, save and exit
    text = raw_input("Press enter to iterate, type 'save' to save:")
    if text != "":
      np.savetxt("X_nE%i_DZ%i.dat" % (nE,DZ) ,(w,X))
      return
    
if __name__ == "__main__":   
  #nE,DZ = 12,1
  #Vup, Vdown = SC1GNRTop(nE,DZ,0)
  #fXi = np.vectorize(lambda w: X1RPAGNRTop(nE,DZ,0,Vup,Vdown,w).imag)
  #wilist, Xitemp = sample_function(fXi,[0,1.0e-2], tol=1e-3)
  #Xilist = Xitemp[0]
  #np.savetxt("X_nE%i_DZ%i.dat" % (nE,DZ) ,(wilist,Xilist))
  #pl.plot(wilist,Xilist)
  #pl.show()
  
  # Until decided
  nE = 12
  DZlist = range(1,nE)
  for DZ in DZlist:
    humanplot(nE, DZ)

  #nE = 9
  #minlist = []
  #DZlist = range(1,nE)
  #for DZ in DZlist:
    #if DZ % 3 == 0:
      #DZlist.remove(DZ)
      
  #for DZ in DZlist:
    #w,X = np.loadtxt("X_nE%i_DZ%i.dat" % (nE,DZ))
    #minlist.append(X.min())
    
  #pl.plot(DZlist,minlist,'o')
  #pl.savefig("test.png")
  #pl.show()