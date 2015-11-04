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
  
  pl.subplot(1,2,1)
  pl.xlabel("E")
  pl.ylabel("g(E)")
  E,gRe,gIm = np.loadtxt("gBulkAC.dat").T
  pl.plot(E,gRe,E,gIm)
  E,gSPARe,gSPAIm = np.loadtxt("gBulkAC.dat").T
  pl.plot(E,gSPARe,'o',E,gSPAIm,'o')
  
  pl.subplot(1,2,2)
  pl.xlabel("E")
  E,gRe,gIm = np.loadtxt("gBulkAC.dat").T
  pl.plot(E,gRe,E,gIm)
  E,gSPARe,gSPAIm = np.loadtxt("gBulkAC.dat").T
  pl.plot(E,gSPARe,'o',E,gSPAIm,'o')
  
  pl.savefig("gBulk.pdf")
  pl.show()
  
