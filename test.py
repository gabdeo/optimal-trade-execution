import numpy as np
from sympy import symbols, sqrt, hessian, Matrix

def objective(x, c):
    return sum(x[s] * x[t] * (min(s+1, t+1) + sqrt(x[s] * x[t]) * c[s] * c[t]) 
               for s in range(len(x)) 
               for t in range(len(x))
            )

def check_convexity(n):

    x = symbols('x1:%d' % (n+1))
    c = symbols('c1:%d' % (n+1))
    func = objective(x, c)

    # Compute the Hessian matrix
    H = hessian(func, x)

    # Check if the Hessian matrix is positive semi-definite
    # (i.e. all eigenvalues are non-negative)
    eigenvalues = H.eigenvals()
    if np.all(eigenvalues >= 0):
        return True
    
    return False


if __name__ == '__main__':
    check_convexity(13)
    exit(1)

