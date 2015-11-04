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
  
  
  #pl.xlabel("D")
  #pl.ylabel("J(D)")
  #Dlist,Jlist = np.loadtxt("JBulkSubsAC.dat").T
  #pl.plot(Dlist,Jlist,label="AC")

  #Dlist,Jlist = np.loadtxt("JBulkSubsZZ.dat").T
  #pl.plot(Dlist,Jlist)
  #pl.legend()
  #pl.savefig("JBulkSubsLog.pdf")
  #pl.show()
  
  pl.figure(figsize=(16,6))
  sns.set(font_scale=1.5)
  
  pl.subplot(1,3,1)
  pl.xlabel("E")
  pl.ylabel("J")
  D,J = np.loadtxt("JSISubsDZ1.dat").T
  pl.plot(D,J)
  pl.subplot(1,3,2)
  pl.xlabel("E")
  D,J = np.loadtxt("JSISubsDZ2.dat").T
  pl.plot(D,J)
  pl.subplot(1,3,3)
  pl.xlabel("E")
  D,J = np.loadtxt("JSISubsDZ3.dat").T
  pl.plot(D,J)
  pl.show()
  
