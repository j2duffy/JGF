import numpy as np
import matplotlib
import pylab as pl
import seaborn as sns

if __name__ == "__main__":
  pl.figure(figsize=(12,9))
  sns.set(font_scale=2.2)
  
  E,G = np.loadtxt("GSubs0.dat").T
  pl.plot(E,G,label=r"$D_Z = 0$")
  
  E,G = np.loadtxt("GSubs1.dat").T
  pl.plot(E,G,label=r"$D_Z = 1$")
  
  E,G = np.loadtxt("GSubs2.dat").T
  pl.plot(E,G,label=r"$D_Z = 2$")
  
  pl.xlabel(r"$E_F$")
  pl.ylabel(r"$\Gamma(E_F)$")

  pl.legend(frameon=True,framealpha=0.5)
  pl.savefig("GSubs.pdf")
  pl.show()