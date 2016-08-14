"""Converts row arranged files to column arranges files.
python fconvert.py file_in file_out"""

import sys
import numpy as np

def get_col(f1,f2):
  """Read a file in row order, converts it to a user specified file in column order"""
  with open(f1,'r') as f:
    data = np.loadtxt(f1)
  with open(f2,'w') as f:
    for i in zip(*data):
      fmt = len(i)*'{:<18.12}' + '\n'
      f.write( fmt.format(*i) )


if __name__ == "__main__":
  f1,f2 = sys.argv[1:]
  get_col(f1,f2)

