from Dynamic import *
from numpy.testing import assert_allclose

print "Dynamic Tests"

#nE,m,n = 6,1,0
#Vup,Vdown = SC_GNRTop1(nE,m,n)
#X = XRPA_GNRTop1(nE,m,n,Vup,Vdown,0.3)
#assert_allclose((Vup,Vdown),(-5.859200848518203, 3.859200848518169))
#assert_allclose(X,(3.14609188034-0.011961195901j))


m,n = 3,2
assert_allclose((-3.7636285014641575, 4.775961447320155), SCBulkCenter(m,n))
print "All Tests Passed!"