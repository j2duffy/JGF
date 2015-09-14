# JGF
A collection of Green's Functions, mostly for graphene and its allotropes

Where possible, the lowest level routines have been written in FORTRAN.
Numerical integration is done in python, since I could not find a FORTRAN version that wasn't awful.

GF.py contains basic Green's Functions.
GFRoutines.py constains utility functions (generally mx population routines).
Coupling.py has a bunch of functions for calculating the IEC for different systems and impurity conformations.
Recursive.py builds armchair nanoribbons recursively, and also calculates the conductance for same.
Dynamic.py calculates spin susceptibilities in the presense of peturbed impurities in the aformentioned systems.
TestRoutines stores routines that I am working on at a certain moment, but do not expect to stand the test of time. 
tst.py is a unit testing module. 
functionsample.py provides a way to plot functions adaptively. 

The FORTRAN module is compiled using
f2py --fcompiler=gfortran --opt=-O -c -m FMod FMod.f95 
I avoid capital letters while using f2py, since this seems to avoid the name mangling issue.
