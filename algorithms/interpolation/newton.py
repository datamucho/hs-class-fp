import numpy as np

def divided_differences(x, y):
    """
    Calculate divided differences table for Newton's interpolation.
    
    Args:
        x: Array of x coordinates of data points
        y: Array of y coordinates of data points
        
    Returns:
        coefs: Coefficients of the Newton form
        dd_table: Full divided differences table for visualization
    """
    n = len(x)
    dd_table = np.zeros((n, n))
    
    # Fill in the first column with y values
    dd_table[:, 0] = y
    
    # Calculate the divided differences
    for j in range(1, n):
        for i in range(n - j):
            dd_table[i, j] = (dd_table[i+1, j-1] - dd_table[i, j-1]) / (x[i+j] - x[i])
    
    # The coefficients are the first row of the table
    coefs = dd_table[0, :]
    
    return coefs, dd_table

def interpolate(x_points, y_points, x_eval):
    """
    Newton's divided differences interpolation.
    
    Args:
        x_points: Array of x coordinates of data points
        y_points: Array of y coordinates of data points
        x_eval: Points at which to evaluate the interpolating polynomial
        
    Returns:
        tuple: (y_eval, dd_table)
            - y_eval: Interpolated values at x_eval points
            - dd_table: Divided differences table for visualization
    """
    x_points = np.array(x_points)
    y_points = np.array(y_points)
    x_eval = np.array(x_eval)
    
    n = len(x_points)
    if n != len(y_points):
        raise ValueError("x_points and y_points must have the same length")
    
    # Calculate divided differences coefficients
    coefs, dd_table = divided_differences(x_points, y_points)
    
    # Initialize result array
    if np.isscalar(x_eval):
        y_eval = coefs[0]
        
        # Add contributions from each term
        for i in range(1, n):
            term = coefs[i]
            for j in range(i):
                term *= (x_eval - x_points[j])
            y_eval += term
    else:
        y_eval = np.full_like(x_eval, coefs[0], dtype=float)
        
        # Add contributions from each term
        for i in range(1, n):
            term = np.full_like(x_eval, coefs[i], dtype=float)
            for j in range(i):
                term *= (x_eval - x_points[j])
            y_eval += term
    
    return y_eval, dd_table

def evaluate_at_point(x_points, y_points, x):
    """Evaluate Newton polynomial at a single point"""
    coefs, _ = divided_differences(x_points, y_points)
    
    n = len(x_points)
    result = coefs[0]
    
    for i in range(1, n):
        term = coefs[i]
        for j in range(i):
            term *= (x - x_points[j])
        result += term
    
    return result 