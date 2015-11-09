import numpy as np
import matplotlib
import pylab as pl
import seaborn as sns

if __name__ == "__main__":
  sns.set(font_scale=1.5)
  DZlist, alist = np.loadtxt("JSISubs_Decay.dat").T
  pl.plot(DZlist,alist)
  pl.ylabel(r"$\alpha$")
  pl.xlabel(r"$D_Z$")
  pl.savefig("JSISubs_Decay.pdf")
  pl.show()
