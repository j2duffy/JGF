# JGF
Routines for calculating various physical properties of graphene and related materials, mainly using Green's Functions.
Low level routines are in FORTRAN.
Higher level routines and numerical integration are in Python.

## Modules
*GF.py* contains basic Green's Functions for different systems. Often these will just be wrappers for the FORTRAN functions.
*GFRoutines.py* constains utility functions (generally mx population routines).
*Coupling.py* has a bunch of functions for calculating the IEC for different systems and impurity conformations.
*Recursive.py* builds armchair nanoribbons recursively, and also calculates the conductance for same.
*Dynamic.py* calculates spin susceptibilities in the presense of peturbed impurities in the aformentioned systems.

## Main Files
*Main.py* contains whatever projects I am working on at the moment. Normally it will import a bunch of modules from above
*NicePlot.py* uses seaborn to create what **I** think are presentable plots. It will normally use data from the */data* folder

## Other Files
*tst.py* is a unit testing module. 
I haven't fully completed this, and it suffers from the normal floating point problems, and the fact that I would expect some things to change after program changes.
TestRoutines is a legacy file that stores a bunch of routines that I never though would stand the test of time.
functionsample.py provides a way to plot functions adaptively, and is essential when working with the susceptibility.

## Running
The FORTRAN module is compiled using
f2py --fcompiler=gfortran --opt=-O -c -m FMod FMod.f95 
I avoid capital letters while using f2py, since this seems to avoid the name mangling issue.

Python files are run in the usual manner of *python $file*.
