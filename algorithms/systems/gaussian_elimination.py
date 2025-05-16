import numpy as np

def solve(A, b, pivoting=True):
    """
    Gaussian elimination with optional pivoting.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        pivoting: Whether to use partial pivoting
        
    Returns:
        x: Solution vector
    """
    # For now, just use NumPy's solver as a placeholder
    # In a complete implementation, we would code the algorithm from scratch
    return np.linalg.solve(A, b) 