import numpy as np
import matplotlib
import pylab as pl
import seaborn as sns

if __name__ == "__main__":
  pl.figure(figsize=(14,6))
  sns.set(font_scale=1.8)
  
  pl.subplot(1,2,1)
  E,gre,gim = np.loadtxt("gSI1.dat").T
  pl.plot(E,gre,label="Re")
  pl.plot(E,gim,label="Im")
  
  E,gre,gim = np.loadtxt("gSI_SPA1.dat").T
  pl.plot(E,gre,'o',markevery=3)
  pl.plot(E,gim,'o',markevery=3)
  pl.xlabel(r"$E$")
  pl.ylabel(r"$g(E)$")
  pl.legend()

  pl.subplot(1,2,2)
  E,gre,gim = np.loadtxt("gSI2.dat").T
  pl.plot(E,gre,label="Re")
  pl.plot(E,gim,label="Im")
  
  E,gre,gim = np.loadtxt("gSI_SPA2.dat").T
  pl.plot(E,gre,'o',markevery=3)
  pl.plot(E,gim,'o',markevery=3)
  pl.xlabel(r"$E$")
  pl.ylabel(r"$g(E)$")
  pl.legend()

  pl.tight_layout()
  pl.savefig("gBulkSPA.pdf")
  pl.show()