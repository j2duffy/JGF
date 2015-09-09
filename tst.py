from Dynamic import *
from Coupling import *
from Recursive import *
from numpy.testing import assert_allclose


print "Recursive Tests"
N = 5
p = 2
E = 1.4
ImpList = [1,5,10,15]
assert_allclose( KuboSubs(N,p,E,ImpList), 1.15328297698 )

N,p,nimp,E = 10,2,1,1.2
assert_allclose( ConfigAvCenterTotal(N,p,nimp,E), 4.77837126094 )


print "Coupling Tests"
nE,m1,n1,m2,n2,s = 9,1,0,5,2,1
assert_allclose(JGNRSubs(nE,m1,n1,m2,n2,s),8.9053121966e-07)
print "Passed coupling tests"

m,n,s = 3,1,0
assert_allclose( JBulkSubs(3,1,0),-0.000871492243211 )

print "Dynamic Tests"

print "Testing SC1GNRTop and X1RPAGNRTop"
nE,m,n = 6,1,0
Vup,Vdown = SC1GNRTop(nE,m,n)
X = X1RPAGNRTop(nE,m,n,Vup,Vdown,0.3)
assert_allclose((Vup,Vdown),(-5.859200848518203, 3.859200848518169))
assert_allclose(X,(3.14609188034-0.011961195901j))


print "Testing SC3GNRSubs and X3RPAGNRSubs"
nE = 8
r0 = [1,0,0]
r1 = [2,0,0]
r2 = [4,1,1]

Vup,Vdown = SC3GNRSubs(nE,r0,r1,r2)
assert_allclose(
      (Vup,Vdown), (np.array([[-4.38213749,  0.        ,  0.        ],
       [ 0.        , -4.06898994,  0.        ],
       [ 0.        ,  0.        , -4.10707937]]), np.array([[ 4.38213749,  0.        ,  0.        ],
       [ 0.        ,  4.06898994,  0.        ],
       [ 0.        ,  0.        ,  4.10707937]]))
       )

assert_allclose( X3RPAGNRSubs(nE,r0,r1,r2,Vup,Vdown,0.3),
[[ -9.90294033 -1.65273553e-03j,  20.51468263 +2.06285294e-03j,
    0.33150831 -4.60325320e-05j],
 [ 20.51468263 +2.06285294e-03j, -18.21750620 -4.45142722e-03j,
   -0.11331096 -3.02317554e-04j],
 [  0.33150831 -4.60325320e-05j,  -0.11331096 -3.02317554e-04j,
    8.33187485 -1.82502590e-04j]]	)



print "Testing SC2BulkCenter"
m,n = 3,2
assert_allclose((-3.7636285014641575, 4.775961447320155), SC2BulkCenter(m,n))


print "Testing SCnGNRSubs"
nE,r = 8,[[1,0,0],[5,1,1],[3,2,0],[10,5,0]]
Vup,Vdown = SCnGNRSubs(nE,r)
assert_allclose( (Vup,Vdown),
(np.array([[-4.45740789,  0.        ,  0.        ,  0.        ],
[ 0.        , -4.44635686,  0.        ,  0.        ],
[ 0.        ,  0.        , -4.63386753,  0.        ],
[ 0.        ,  0.        ,  0.        , -4.11426048]]), np.array([[ 4.45740789,  0.        ,  0.        ,  0.        ],
[ 0.        ,  4.44635686,  0.        ,  0.        ],
[ 0.        ,  0.        ,  4.63386753,  0.        ],
[ 0.        ,  0.        ,  0.        ,  4.11426048]]))		)
print "Testing XnHFGNRSubs"
assert_allclose( XnHFGNRSubs(nE,r,[1,0],Vup,Vdown,0.3),(-0.000549720171215+1.05613653854e-06j) )


print "Testing SCnGNRTop"
nE,r = 5,[[1,0,0],[2,1,1],[0,-2,0],[10,9,1]]
assert_allclose( SCnGNRTop(nE,r),
(np.array([[-5.8605745 ,  0.        ,  0.        ,  0.        ],
[ 0.        , -5.86060756,  0.        ,  0.        ],
[ 0.        ,  0.        , -5.87387876,  0.        ],
[ 0.        ,  0.        ,  0.        , -5.85995626]]), np.array([[ 3.8605745 ,  0.        ,  0.        ,  0.        ],
[ 0.        ,  3.86060756,  0.        ,  0.        ],
[ 0.        ,  0.        ,  3.87387876,  0.        ],
[ 0.        ,  0.        ,  0.        ,  3.85995626]]))		)
# This will evaluate incorrectly unless this test is done on its own. Which is disturbing.
print "Testing XnHFGNRTop"
assert_allclose( XnHFGNRTop(nE,r,[1,1],Vup,Vdown,0.1),(-0.108515546142-5.7631916817e-12j) )



print "All Tests Passed!"