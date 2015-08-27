from Dynamic import *

print "Dynamic Tests"

nE,m,n = 6,1,0
Vup,Vdown = SC_GNRTop1(nE,m,n)
X = XRPA_GNRTop1(nE,m,n,Vup,Vdown,0.3)

if (Vup,Vdown) == (-5.859200848518203, 3.859200848518169) :
  print "Test Passed"
else:
  print SC_GNRTop1(6,1,0)
  print (-5.859200848518203, 3.859200848518169)
  
if X == (3.14609188034-0.011961195901j):
  print "Test Passed"
else:
  print X
  print (3.14609188034-0.011961195901j)