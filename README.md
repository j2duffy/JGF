# JGF
A collection of Green's Functions, mostly for graphene and its allotropes

Where possible, the lowest level routines have been written in FORTRAN.
Numerical integration is done in python, since I could not find a FORTRAN version that wasn't awful.

GF.py contains basic Green's Functions.
GFRoutines.py constains utility functions (generally mx population routines).
Coupling.py has a bunch of functions for calculating the IEC for different systems and impurity conformations..

The FORTRAN module is compiled using
f2py --fcompiler=gfortran --opt=-O -c -m FMod FMod.f95 
Bear in mind that capital letters do not work in f2py, or at least the version that I'm using.
