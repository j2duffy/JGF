# JGF
A collection of Green's Functions, mostly for graphene and its allotropes

Where possible, the lowest level routines have been written in FORTRAN.
Numerical integration is done in python, since I could not find a FORTRAN version that wasn't awful.
Compile instructions for the FORTRAN module (which is linked with f2py) can be found in the log file. Although this may realistically be a better place for them.
GF.py contains basic Green's Functions.
GFRoutines.py constains utility routines for use with them (generally mx population routines).
Coupling.py has a bunch of functions for calculating the IEC in various different cases.


