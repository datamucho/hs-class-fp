import numpy as np

def solve(A, b, pivoting=True):
    """
    Gaussian elimination with optional pivoting.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        pivoting: Whether to use partial pivoting
        
    Returns:
        tuple: (x, steps)
            - x: Solution vector
            - steps: List of steps for visualization
    """
    # Convert to numpy arrays for consistent handling
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    # Create copies to avoid modifying the original arrays
    A_work = A.copy()
    b_work = b.copy()
    
    n = len(b)
    
    # Store steps for visualization (each step is a tuple of A and b)
    steps = [(A_work.copy(), b_work.copy())]
    
    # Forward elimination
    for k in range(n-1):
        # Partial pivoting
        if pivoting:
            # Find index of maximum element in current column
            i_max = np.argmax(np.abs(A_work[k:, k])) + k
            
            # Swap rows if needed
            if i_max != k:
                A_work[[k, i_max]] = A_work[[i_max, k]]
                b_work[[k, i_max]] = b_work[[i_max, k]]
                
                # Record step after pivoting
                steps.append((A_work.copy(), b_work.copy()))
        
        # Check for zero pivot
        if abs(A_work[k, k]) < 1e-10:
            raise ValueError(f"Zero pivot encountered at position ({k},{k})")
        
        # Elimination for all rows below pivot
        for i in range(k+1, n):
            factor = A_work[i, k] / A_work[k, k]
            
            # Update row i
            A_work[i, k:] = A_work[i, k:] - factor * A_work[k, k:]
            b_work[i] = b_work[i] - factor * b_work[k]
        
        # Record step after row operations
        steps.append((A_work.copy(), b_work.copy()))
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        # Calculate sum of known terms
        s = 0
        for j in range(i+1, n):
            s += A_work[i, j] * x[j]
        
        # Solve for x[i]
        x[i] = (b_work[i] - s) / A_work[i, i]
    
    # Record final solution
    steps.append((A_work.copy(), b_work.copy(), x.copy()))
    
    return x, steps 