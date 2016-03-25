import numpy as np
import matplotlib
import pylab as pl
import seaborn as sns

from math import pi, sin

if __name__ == "__main__":
  #pl.figure(figsize=(12,6))
  sns.set(font_scale=1.4)
  
  w,X = np.loadtxt("3XV1.dat").T
  pl.plot(w,X,label="3")
  w,X = np.loadtxt("2XV0comp3.dat").T
  pl.plot(w,X,"--",label="2")
  
  pl.xlim(0.0006,0.0018)
  pl.ylim(-400000,100000)
  pl.ticklabel_format(style="sci",scilimits=(0,0))
  
  pl.xlabel(r"$\omega$")
  pl.ylabel(r"$\chi(\omega)$")
  
  pl.legend()
  pl.tight_layout()
  pl.savefig("3XV1.pdf")
  pl.show()