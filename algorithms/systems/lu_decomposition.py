import numpy as np

def decompose(A):
    """
    LU Decomposition without pivoting.
    
    Args:
        A: Square matrix to decompose
        
    Returns:
        tuple: (L, U, steps)
            - L: Lower triangular matrix
            - U: Upper triangular matrix
            - steps: List of steps showing the decomposition process
    """
    # Convert to numpy array for consistent handling
    A = np.array(A, dtype=float)
    n = A.shape[0]
    
    # Check if matrix is square
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square for LU decomposition")
    
    # Initialize L and U matrices
    L = np.eye(n)  # Identity matrix for L
    U = np.zeros((n, n))
    
    # Store steps for visualization
    steps = []
    
    # Calculate U's first row
    U[0, :] = A[0, :]
    steps.append(("Initialize U's first row", L.copy(), U.copy()))
    
    # Check for zero pivot
    if abs(U[0, 0]) < 1e-10:
        raise ValueError("Zero pivot encountered at position (0,0)")
    
    # Calculate L's first column
    L[1:, 0] = A[1:, 0] / U[0, 0]
    steps.append(("Calculate L's first column", L.copy(), U.copy()))
    
    # Fill in remaining elements
    for i in range(1, n):
        # Calculate U's current row
        for j in range(i, n):
            s = 0
            for k in range(i):
                s += L[i, k] * U[k, j]
            U[i, j] = A[i, j] - s
        
        steps.append((f"Calculate U's row {i}", L.copy(), U.copy()))
        
        # Check for zero pivot
        if abs(U[i, i]) < 1e-10:
            raise ValueError(f"Zero pivot encountered at position ({i},{i})")
            
        # Calculate L's current column (if not the last row)
        if i < n - 1:
            for j in range(i + 1, n):
                s = 0
                for k in range(i):
                    s += L[j, k] * U[k, i]
                L[j, i] = (A[j, i] - s) / U[i, i]
            
            steps.append((f"Calculate L's column {i}", L.copy(), U.copy()))
    
    return L, U, steps

def solve(A, b):
    """
    Solve a system Ax = b using LU decomposition.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        
    Returns:
        tuple: (x, steps)
            - x: Solution vector
            - steps: List of steps for visualization
    """
    # Convert to numpy arrays for consistent handling
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    # Perform LU decomposition
    L, U, decomp_steps = decompose(A)
    
    # Store steps for visualization
    steps = [("LU Decomposition", L.copy(), U.copy())]
    
    # Forward substitution (Ly = b)
    y = np.zeros(n)
    for i in range(n):
        s = 0
        for j in range(i):
            s += L[i, j] * y[j]
        y[i] = b[i] - s
        steps.append(("Forward Substitution", i, y.copy()))
    
    # Back substitution (Ux = y)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        s = 0
        for j in range(i+1, n):
            s += U[i, j] * x[j]
        x[i] = (y[i] - s) / U[i, i]
        steps.append(("Back Substitution", i, x.copy()))
    
    return x, steps 