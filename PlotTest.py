import numpy as np
import matplotlib
import pylab as pl
import seaborn as sns

if __name__ == "__main__":
  #E, S, T, C = np.loadtxt("test.dat").T
  
  #sns.set_style("white")

  #sns.set_palette(sns.mpl_palette("YlGnBu_d"))
  #pl.plot(E,S,E,T,E,C)
  #pl.xlabel("E")
  #sns.despine()
  #pl.show()
  
  sns.set(font_scale=1.5)
  pl.figure(figsize=(16,6))

  pl.subplot(1,2,1)
  pl.ylabel("J")
  pl.xlabel("DA")
  D,J = np.loadtxt("JBulkSubsAC.dat").T
  pl.plot(D,J)
  
  pl.subplot(1,2,2)
  pl.xlabel("DA")
  D,J = np.loadtxt("JBulkSubsZZ.dat").T
  pl.plot(D,J)
  
  pl.savefig("JBulkSubs.pdf")
  pl.show()
  
