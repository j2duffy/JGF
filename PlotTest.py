import numpy as np
import matplotlib
import pylab as pl
import seaborn as sns

from math import pi, sin

if __name__ == "__main__":
  #pl.figure(figsize=(12,6))
  sns.set(font_scale=1.5)
  
  D,J = np.loadtxt("JLinebbLog.dat").T
  sns.regplot(D,J,label="bb")
  D,J = np.loadtxt("JLinebwLog.dat").T
  sns.regplot(D,J,label="bw")
  D,J = np.loadtxt("JLinewbLog.dat").T
  sns.regplot(D,J,label="wb")
  
  pl.xlabel(r"$\log(D)$")
  pl.ylabel(r"$\log(J(D))$")
  pl.legend()
  
  #pl.ticklabel_format(style="sci",scilimits=(0,100))

  pl.tight_layout()
  pl.savefig("JLineLog.pdf")
  pl.show()