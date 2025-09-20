import numpy as np

# Written all the constraint functions in a matrix by converting 
# them in to the form of <= 0
def constraint_functions(x):
    y = np.array([4*x[0]**2 + x[1]**2 - 16, 
                  3*x[0] + 5*x[1] - 4, 
                  -x[0], 
                  -x[1]])
    return y