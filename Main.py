# Currently working on some shit with my Dynamic code
from Dynamic import *

def Xplot(nE,DZ,interval):
  Vup, Vdown = SC1GNRTop(nE,DZ,0)
  fXi = np.vectorize(lambda w: X1RPAGNRTop(nE,DZ,0,Vup,Vdown,w).imag)
  wilist, Xitemp = sample_function(fXi,interval, tol=1e-3)
  Xilist = Xitemp[0]
  return wilist, Xilist


def AdjustPlot(nE,DZ):
  """Plots the susceptibility at various human defined ranges until it finds a well resolved peak.
  Save the appropriate values"""
  # Initial plot
  w,X = Xplot(nE,DZ,[0,1.0e-2])
  pl.plot(w,X)
  pl.show()
  # Until you give the OK
  while True:
    # Get Center
    imin = X.argmin()
    wmin = w[imin]
    #Request range
    wrange = eval(raw_input("Select a range:"))
    # Replot with new range 
    w,X = Xplot(nE,DZ,[wmin-wrange,wmin+wrange])
    pl.plot(w,X)
    pl.show()
    # If range works out, save and exit
    text = raw_input("Save plot? (y or n):")
    if text == "y":
      np.savetxt("data/X_nE%i_DZ%i.dat" % (nE,DZ) ,(w,X))
      return
    
    
def AdjustPlotEF(nE,DZ):
  """Exactly the same as AdjustPlot, except labels the file with EF"""
  # Initial plot
  w,X = Xplot(nE,DZ,[0,1.0e-2])
  pl.plot(w,X)
  pl.show()
  # Until you give the OK
  while True:
    # Get Center
    imin = X.argmin()
    wmin = w[imin]
    #Request range
    wrange = eval(raw_input("Select a range:"))
    # Replot with new range 
    w,X = Xplot(nE,DZ,[wmin-wrange,wmin+wrange])
    pl.plot(w,X)
    pl.show()
    # If range works out, save and exit
    text = raw_input("Save plot? (y or n):")
    if text == "y":
      np.savetxt("data/X_nE%i_DZ%i_EF%.1f.dat" % (nE,DZ,EF) ,(w,X))
      return
    
    
def PeakHeight():
  nE = 30
  DZlist = range(1,nE)
  for DZ in DZlist:
    AdjustPlot(nE, DZ)
    
  nE = 30
  minlist = []
  DZlist = range(1,nE)
  for DZ in DZlist:
    if DZ % 3 == 0:
      DZlist.remove(DZ)
      
  for DZ in DZlist:
    w,X = np.loadtxt("data/X_nE%i_DZ%i.dat" % (nE,DZ))
    minlist.append(X.min())
    
  pl.plot(DZlist,minlist)
  pl.savefig("test.png")
  pl.show()
  
  
if __name__ == "__main__":   
  nE,DZ = 6,1
  EF = 0.0
  w,X = np.loadtxt("data/X_nE%i_DZ%i_EF%.1f.dat" % (nE,DZ,EF))
  X = X - X.min()/2.0
  spline = UnivariateSpline(w,X)
  #r1, r2 = spline.roots() # find the roots
  pl.plot(w,X)
  pl.plot(w,spline(w),'o')
  pl.show()


