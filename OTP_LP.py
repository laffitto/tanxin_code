# Optimal Transportation Problem with Linear Programming
from scipy.optimize import linprog
import numpy as np

# C = [5, 6, 4, 6, 3, 7]
# A_ub = [[1, 1, 1, 0, 0, 0],
#      [0, 0, 0, 1, 1, 1],
#      [1, 0, 0, 1, 0, 0],
#      [0, 1, 0, 0, 1, 0],
#      [0, 0, 1, 0, 0, 1]]
# b_ub = [300, 500, 200, 300, 250]
# A_eq = [[1]*6]
# b_eq = [min(300+500,200+300+250)]
# X_lb = 0
# X_ub = None
# res = linprog(C,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=(X_lb,X_ub))
# print(res)

# con: array([1.91433293e-05])
# fun: 3049.999925005495
# message: 'Optimization terminated successfully.'
# nit: 6
# slack: array([7.64736461e-06, 5.00000115e+01, 5.09651198e-06, 7.66618263e-06,
#               6.38063460e-06])
# status: 0
# success: True
# x: array([4.99999987e+01, 4.82740451e-07, 2.49999993e+02, 1.49999996e+02,
#           2.99999992e+02, 4.53980581e-07])
# x = [50, 0, 250, 150, 300, 0]


C = [5, 6, 4, 6, 3, 7]
A_ub = [[1, 1, 1, 0, 0, 0],
     [0, 0, 0, 1, 1, 1],
     [1, 0, 0, 1, 0, 0],
     [0, 1, 0, 0, 1, 0],
     [0, 0, 1, 0, 0, 1],
     [1, 1, 1, 1, 1, 1],
     [-1,-1,-1,-1,-1,-1]]
b_ub = [300, 500, 200, 300, 250, 750, -750]
res = linprog(C,A_ub=A_ub,b_ub=b_ub)
print(res)

# con: array([], dtype=float64)
# fun: 3049.999972847826
# message: 'Optimization terminated successfully.'
# nit: 6
# slack: array([2.73738743e-06, 5.00000041e+01, 1.81903616e-06, 2.73697560e-06,
#               2.27812262e-06, 6.83413430e-06, -6.83413430e-06])
# status: 0
# success: True
# x: array([4.99999997e+01, 4.41209764e-08, 2.49999997e+02, 1.49999998e+02,
#           2.99999997e+02, 2.24855832e-07])
# x = [50, 0, 250, 150, 300, 0]


# dual-form
C2 =  np.array(b_ub)
A_ub2 =  - np.array(A_ub).T
b_ub2 =  np.array(C)
res2 = linprog(C2,A_ub=A_ub2,b_ub=b_ub2)
print(res2)

# con: array([], dtype=float64)
# fun: -3049.999959683235
# message: 'Optimization terminated successfully.'
# nit: 6
# slack: array([1.11393774e-07, 4.00000008e+00, 4.82714517e-08, 6.77400323e-08,
#               3.85791310e-08, 2.00000000e+00])
# status: 0
# success: True
# x: array([1.00000006e+00, 1.88892064e-08, 7.73928569e+01, 8.03928569e+01,
#           7.83928569e+01, 7.00809525e+01, 1.53473809e+02])
# x = [50, 0, 250, 150, 300, 0]
# x = [1, 0, 77, 80, 78, 70, 153]