from Dynamic import *
from numpy.testing import assert_allclose

print "Dynamic Tests"

#print "Testing SC_GNRTop1 and XRPA_GNRTop1"
#nE,m,n = 6,1,0
#Vup,Vdown = SC_GNRTop1(nE,m,n)
#X = XRPA_GNRTop1(nE,m,n,Vup,Vdown,0.3)
#assert_allclose((Vup,Vdown),(-5.859200848518203, 3.859200848518169))
#assert_allclose(X,(3.14609188034-0.011961195901j))


#print "Testing SC_GNRSubs3 and XRPAGNR3"
#nE = 8
#r0 = [1,0,0]
#r1 = [2,0,0]
#r2 = [4,1,1]

#Vup,Vdown = SC_GNRSubs3(nE,r0,r1,r2)
#assert_allclose(
      #(Vup,Vdown), (np.array([[-4.38213749,  0.        ,  0.        ],
       #[ 0.        , -4.06898994,  0.        ],
       #[ 0.        ,  0.        , -4.10707937]]), np.array([[ 4.38213749,  0.        ,  0.        ],
       #[ 0.        ,  4.06898994,  0.        ],
       #[ 0.        ,  0.        ,  4.10707937]]))
       #)

#assert_allclose( XRPAGNR3(nE,r0,r1,r2,Vup,Vdown,0.3),
#[[ -9.90294033 -1.65273553e-03j,  20.51468263 +2.06285294e-03j,
    #0.33150831 -4.60325320e-05j],
 #[ 20.51468263 +2.06285294e-03j, -18.21750620 -4.45142722e-03j,
   #-0.11331096 -3.02317554e-04j],
 #[  0.33150831 -4.60325320e-05j,  -0.11331096 -3.02317554e-04j,
    #8.33187485 -1.82502590e-04j]]	)



#print "Testing SCBulkCenter"
#m,n = 3,2
#assert_allclose((-3.7636285014641575, 4.775961447320155), SCBulkCenter(m,n))



#print "All Tests Passed!"