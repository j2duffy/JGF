import numpy as np
import pylab as pl
import seaborn as sns

if __name__ == "__main__":
  E, S, T, C = np.loadtxt("test.dat").T
  
  sns.set_style("white")
  pl.plot(E,S,E,T,E,C)
  pl.xlabel("E")
  sns.despine()
  pl.show()
  print pl.s