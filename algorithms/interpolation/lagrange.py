import numpy as np

def interpolate(x_points, y_points, x_eval):
    """
    Lagrange polynomial interpolation.
    
    Args:
        x_points: Array of x coordinates of data points
        y_points: Array of y coordinates of data points
        x_eval: Points at which to evaluate the interpolating polynomial
        
    Returns:
        tuple: (y_eval, basis_polynomials)
            - y_eval: Interpolated values at x_eval points
            - basis_polynomials: List of Lagrange basis polynomials for visualization
    """
    x_points = np.array(x_points)
    y_points = np.array(y_points)
    x_eval = np.array(x_eval)
    
    n = len(x_points)
    if n != len(y_points):
        raise ValueError("x_points and y_points must have the same length")
    
    # Initialize result array
    if np.isscalar(x_eval):
        y_eval = 0.0
    else:
        y_eval = np.zeros_like(x_eval, dtype=float)
    
    # Store basis polynomials for visualization
    basis_polynomials = []
    
    # Calculate Lagrange polynomial
    for i in range(n):
        # Initialize basis polynomial
        basis = np.ones_like(x_eval, dtype=float)
        
        # Calculate basis polynomial L_i(x)
        for j in range(n):
            if i != j:
                basis *= (x_eval - x_points[j]) / (x_points[i] - x_points[j])
        
        # Store basis polynomial for visualization
        basis_polynomials.append(basis.copy())
        
        # Add contribution from this point
        y_eval += y_points[i] * basis
    
    return y_eval, basis_polynomials

def evaluate_at_point(x_points, y_points, x):
    """Evaluate Lagrange polynomial at a single point"""
    x_points = np.array(x_points)
    y_points = np.array(y_points)
    
    n = len(x_points)
    result = 0.0
    
    for i in range(n):
        # Calculate basis polynomial L_i(x)
        basis = 1.0
        for j in range(n):
            if i != j:
                basis *= (x - x_points[j]) / (x_points[i] - x_points[j])
        
        # Add contribution from this point
        result += y_points[i] * basis
    
    return result 