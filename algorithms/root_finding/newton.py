import numpy as np

def solve(f, df, x0, tol=1e-6, max_iter=100):
    """
    Newton's method for root finding.
    
    Args:
        f: Function for which we want to find the root
        df: Derivative of the function
        x0: Initial guess
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
        
    Returns:
        tuple: (root, iterations, errors)
            - root: Approximate root
            - iterations: List of all x values at each iteration
            - errors: List of error estimates at each iteration
    """
    iterations = [x0]
    errors = [abs(f(x0))]
    
    x = x0
    
    for i in range(max_iter):
        # Function and derivative value
        fx = f(x)
        dfx = df(x)
        
        # Check for zero derivative
        if abs(dfx) < 1e-14:
            raise ValueError("Derivative is close to zero - Newton's method failed")
        
        # Newton's update
        x_new = x - fx / dfx
        
        # Store iteration data
        iterations.append(x_new)
        
        # Compute error estimate
        error = abs(x_new - x)
        errors.append(error)
        
        # Update x
        x = x_new
        
        # Check for convergence
        if error < tol or abs(f(x)) < tol:
            break
    
    return iterations[-1], iterations, errors 