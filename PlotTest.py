import numpy as np
import matplotlib
import pylab as pl
import seaborn as sns

if __name__ == "__main__":
  pl.figure(figsize=(16,8))
  sns.set(font_scale=2)

  pl.subplot(1,2,1)
  E,GS,GT,GC = np.loadtxt("GImpComp.dat").T
  pl.plot(E,GS,label="Subs")
  pl.plot(E,GT,label="Top")
  pl.plot(E,GC,label="Center")
  pl.xlabel(r"$E$")
  pl.ylabel(r"$G(E)$")
  pl.legend(frameon=True,framealpha=0.7)
  
  pl.subplot(2,2,2)
  E,GS,GT,GC = np.loadtxt("GImpComp1.dat").T
  pl.plot(E,GS)
  pl.plot(E,GT)
  pl.plot(E,GC)
  pl.xlabel(r"$E$")
  pl.ylabel(r"$G(E)$")
  
  pl.subplot(2,2,4)
  E,GS,GT,GC = np.loadtxt("GImpComp2.dat").T
  pl.plot(E,GS)
  pl.plot(E,GT)
  pl.plot(E,GC)
  pl.xlabel(r"$E$")
  pl.ylabel(r"$G(E)$")
  
  pl.tight_layout()
  pl.savefig("GImpComp.pdf")
  pl.show()