module shared_data
implicit none
  save
  ! Mathematical parameters
  complex(8), parameter :: im = (0.0d0,1.0d0)
  real(8), parameter :: pi = acos(-1.0d0)
  ! Material parameters
  real(8), parameter :: t = -1.0d0
  ! Other parameters
  complex(8) :: E
  character(2) :: s_lat
  integer :: m, n

end module

function integrand (kZ,a)
use shared_data
implicit none
  real(8) :: integrand
  real(8), intent(in) :: kZ
  real(8), intent(in) :: a

  integrand = t*a*cos(kZ)**2

end function integrand 
