import numpy as np
import pylab as pl
import seaborn as sns
from Dynamic import *

def X1plot():
  sns.set(font_scale=1.5)
  
  pl.figure(figsize=(12,6))
  
  pl.subplot(1,2,1)
  w,X = np.loadtxt("X1DZ1re.dat").T
  pl.plot(w,X)
  w,X = np.loadtxt("X1DZ1im.dat").T
  pl.plot(w,X)
  pl.xlim([0.85e-3,1.1e-3])
  
  pl.xlabel(r"$\omega$")
  pl.ylabel(r"$\chi(\omega)$")
  
  pl.ticklabel_format(style='sci',scilimits=(0,0))
  
  pl.subplot(1,2,2)
  w,X = np.loadtxt("X1DZ3re.dat").T
  pl.plot(w,X)
  w,X = np.loadtxt("X1DZ3im.dat").T
  pl.plot(w,X)
  pl.xlim([9.694658e-4,9.694661e-4])
  
  pl.xlabel(r"$\omega$")
  pl.ylabel(r"$\chi(\omega)$")
  
  pl.ticklabel_format(style='sci',scilimits=(0,0))
  
  pl.tight_layout()
  pl.savefig("X1.pdf")
  pl.show()
  
  
def FWHMvsDZplot():
  sns.set(font_scale=1.5)
  
  DZ, FWHM = np.loadtxt("PaperData/FWHMvDZ.dat")
  pl.plot(DZ, FWHM)
  pl.xlabel(r"$D_Z$")
  pl.ylabel(r"$FWHM(D_Z)$")
  pl.tight_layout()
  pl.savefig("X1FWHM.pdf")
  pl.show()

if __name__ == "__main__": 
  sns.set(font_scale=1.5)

  #EFlist = [0.0,0.5,0.7,0.8,0.9]
  #for EF in EFlist:
    #w,X = np.loadtxt("data/X_nE6_DZ1_EF%.1f.dat" % (EF,))
    #pl.plot(w,X)
  #pl.xlim([9e-4,1e-3])
  #pl.ticklabel_format(style='sci',scilimits=(0,0))
    
  #pl.axes([0.15,0.15,0.4,0.4])
  #x,y = np.loadtxt("PaperData/DOS_DZ1.dat")
  #pl.plot(x,y)
  #x,y = np.loadtxt("PaperData/DOSpoints_DZ1.dat")
  #pl.plot(x,y,'o')
  #pl.tick_params(labelbottom=False,labelleft=False) # labels along the bottom edge are off

  #pl.savefig("plotEF.pdf")
  #pl.show()

  
  EFlist,FWHM = np.loadtxt("PaperData/FWHM2vEF_DZ3.dat")
  pl.plot(EFlist,FWHM,'-o')
  
  pl.xlabel(r"$E_F$")
  pl.ylabel(r"$FWHM(E_F)$")
  
  pl.ticklabel_format(style='sci',scilimits=(0,0))
  
  pl.axes([0.15,0.5,0.4,0.4])
  x,y = np.loadtxt("PaperData/DOS_DZ3.dat")
  pl.plot(x,y)
  
  pl.tick_params(labelbottom=False,labelleft=False) # labels along the bottom edge are off
  
  pl.tight_layout()
  
  pl.savefig("FWHMvEF_DZ3.pdf")
  pl.show()