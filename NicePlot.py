import numpy as np
import pylab as pl
import seaborn as sns
from Dynamic import *

if __name__ == "__main__":   
  sns.set(font_scale=1.5)
  
  nE,DZ = 6,3
  EFlist = np.arange(0,1.2,0.1)
  FWHMlist = []
  for EF in EFlist:
    w,X = np.loadtxt("data/X_nE%i_DZ%i_EF%.1f.dat" % (nE,DZ,EF))
    X = X - X.min()/2.0
    spline = UnivariateSpline(w,X)
    r1, r2 = spline.roots() # find the roots
    FWHMlist.append(r2 - r1)
    
  pl.xlabel(r"$E_F$")
  pl.ylabel(r"$FWHM(E_F)$")

  pl.plot(EFlist,FWHMlist)
  pl.ticklabel_format(style='sci',scilimits=(0,0))
  pl.savefig("test.png")
  pl.show()

  
  #for EF in [0.0,0.5,0.7,0.8,0.9]:
    #w,X = np.loadtxt("data/X_nE6_DZ1_EF%.1f.dat" % (EF,))
    #pl.plot(w,X,label="EF=%.1f" % (EF,))
    
  #pl.xlabel(r"$\omega$")
  #pl.ylabel(r"$\chi(\omega)$")
  #pl.legend(frameon=True,framealpha=0.5)
  
  #pl.xlim(0.9e-3,1.04e-3)
  #pl.ticklabel_format(style='sci',scilimits=(0,0))
  #pl.tight_layout()
  
  #pl.savefig("test.png")
  #pl.show()