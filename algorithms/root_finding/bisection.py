import numpy as np

def solve(f, a, b, tol=1e-6, max_iter=100):
    """
    Bisection method for root finding.
    
    Args:
        f: Function for which we want to find the root
        a: Left bound of the interval
        b: Right bound of the interval
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
        
    Returns:
        tuple: (root, iterations, errors)
            - root: Approximate root
            - iterations: List of all x values at each iteration
            - errors: List of error estimates at each iteration
    """
    # Check if the function has opposite signs at the endpoints
    fa = f(a)
    fb = f(b)
    
    if fa * fb > 0:
        raise ValueError("Function must have opposite signs at the endpoints")
    
    iterations = []
    errors = []
    
    for i in range(max_iter):
        # Calculate midpoint
        c = (a + b) / 2
        fc = f(c)
        
        # Store iteration data
        iterations.append(c)
        current_error = (b - a) / 2
        errors.append(current_error)
        
        # Check for convergence
        if abs(fc) < tol or current_error < tol:
            break
        
        # Update interval
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    
    return iterations[-1], iterations, errors 