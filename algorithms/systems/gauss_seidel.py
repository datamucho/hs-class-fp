import numpy as np

def solve(A, b, x0=None, tol=1e-6, max_iter=100):
    """
    Gauss-Seidel iterative method for solving linear systems.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        x0: Initial guess (default: zeros)
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
        
    Returns:
        tuple: (x, iterations, errors)
            - x: Solution vector
            - iterations: List of solution vectors at each iteration
            - errors: List of error estimates at each iteration
    """
    # Convert inputs to numpy arrays for consistent handling
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    # Get dimensions
    n = len(b)
    
    # Initialize x if not provided
    if x0 is None:
        x0 = np.zeros(n)
    
    # Convert x0 to numpy array
    x0 = np.array(x0, dtype=float)
    
    # Initialize lists to store iterations and errors
    iterations = [x0.copy()]
    errors = [np.linalg.norm(np.dot(A, x0) - b)]
    
    # Current solution
    x = x0.copy()
    
    # Main iteration loop
    for k in range(max_iter):
        x_new = x.copy()  # Start with current values but will update in-place
        
        # Gauss-Seidel update formula (uses latest values immediately)
        for i in range(n):
            s1 = sum(A[i, j] * x_new[j] for j in range(i))  # Use already updated values
            s2 = sum(A[i, j] * x[j] for j in range(i + 1, n))  # Use old values
            
            if abs(A[i, i]) < 1e-14:
                raise ValueError(f"Zero diagonal element at position {i}")
                
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        
        # Store iteration
        iterations.append(x_new.copy())
        
        # Compute error
        error = np.linalg.norm(x_new - x)
        errors.append(error)
        
        # Update x
        x = x_new.copy()
        
        # Check convergence
        if error < tol:
            break
    
    return x, iterations, errors 