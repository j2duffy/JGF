import numpy as np
import pylab as pl
import seaborn as sns
from Dynamic import *

if __name__ == "__main__":   
  sns.set(font_scale=1.5)

  
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