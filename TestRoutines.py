from GFRoutines import *
from numpy.linalg import norm

if __name__ == "__main__":  
  m1 = 1
  n1 = 0
  
  r1 = np.array([m1,n1,0])
  hex1 = np.array([[0,0,0],[0,0,1],[1,0,0],[1,-1,1],[1,-1,0],[0,-1,1]])
  r = hex1 + r1
  
  
  #rcol = r[:,np.newaxis]
  #rij = r-rcol
  #n = len(rij)
  #rflat = rij.reshape(n*n,3)
  #rSflat = map(SymSector,rflat)
  #rUnique = set(map(tuple,rSflat))
  #dic = {k:gBulkList(E,k) for k in rUnique}
  #gflat = np.array([dic[tuple(r)] for r in rSflat])
  #g_mx = gflat.reshape(n,n)  
  #return g_mx
  
  #g_mx = np.zeros([14,14],dtype=complex)
  #g_mx[:12,:12] = BulkMxGen(r,E)
  #g_impurity = 1.0/(E-eps_imp)
  #g_mx[12,12] = g_mx[13,13] = g_impurity